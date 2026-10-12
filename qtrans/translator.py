"""Module translation: a Python file -> one self-contained C++ header.

    python module                 C++ header
    -------------                 ----------
    imports                   ->  used to resolve np.exp, norm.cdf, ... (nothing emitted)
    NAME = constant           ->  inline const double NAME = ...;
    def f(x: float) -> float  ->  inline double f(double x) { ... }
    anything else             ->  skipped, and listed in the report
"""
from __future__ import annotations

import ast
import copy
import inspect
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

from . import library as lib
from .errors import Unsupported
from .expressions import P_ATOM, Expr, ExprTranslator, cpp_ident
from . import classes as cls_
from .functions import FunctionTranslator, is_docstring, python_reads, source_mark, strip_marks, subscript_root, target_leaves, view_root
from .report import FunctionReport, ModuleReport
from .types import (BOOL, FLOAT, INT, MAT, NONE, RNG, STR, VEC, VOID, ArrayType, OptionalType, RngType, Scalar,
                    StructType, TupleType, assignable, is_array)

VERSION = "0.9.0"

# Notes about behaviour that differs only on overflow or error paths, shared by most numeric
# code. Reported once per module instead of under every function.
GENERAL_NOTES = {"int64", "div0", "powoverflow"}
RUNTIME_HEADER = Path(__file__).parent / "runtime" / "qtrans_runtime.hpp"
ARRAYS_HEADER = Path(__file__).parent / "runtime" / "qtrans_arrays.hpp"
RANDOM_HEADER = Path(__file__).parent / "runtime" / "qtrans_random.hpp"


@dataclass
class Param:
    name: str
    cpp: str
    type: object
    default_code: str | None
    # how an array parameter is passed: "cref" (read only), "value" (rebound in the body),
    # "ref" (modified in place, so the caller sees the change, as in Python)
    mode: str = "value"


def _optional_array(t) -> bool:
    return isinstance(t, OptionalType) and is_array(t.inner)


@dataclass
class Signature:
    name: str
    cpp: str
    params: list
    ret: object
    lineno: int
    doc: str | None
    owner: str | None = None        # the class, for a method
    kind: str = "function"          # or "instance", "property", "static", "class", "post_init"
    const: bool = False             # a method that doesn't modify self
    module: str | None = None       # imported from this module of the project (multi-module)


@dataclass
class Constant:
    cpp: str
    type: object
    code: str
    external: bool = False          # imported from another module of the project: not rendered here
    lineno: int = 0                 # where it is assigned (its comments are carried over)


@dataclass
class TranslationResult:
    module: str
    header: str            # the generated C++ header text
    report: ModuleReport
    signatures: dict       # name -> Signature, translated functions only
    uses_runtime: bool
    uses_arrays: bool = False
    uses_random: bool = False
    classes: dict = None   # name -> classes.ClassInfo, translated classes only (stage 5)
    namespace: str = ""    # the C++ namespace
    deps: list = None      # modules of the project this one imports (their headers are included)
    line_map: list = None  # per header line: the (first, last) Python lines it came from, or None
    line_notes: dict = None  # function -> [((first, last), key, note)]: where each note applies


def _collect_comments(source):
    """Python's AST drops comments; tokenize keeps them, so they can be carried over."""
    full, inline = {}, {}
    cols = {}  # line -> (column of its end-of-line comment, the line that comment belongs to)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return full, inline
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        text = tok.string[1:]
        text = text[1:] if text.startswith(" ") else text
        text = text.rstrip().rstrip("\\")  # a trailing backslash would splice lines in C++
        line = tok.start[0]
        if line == 1 and text.startswith("!"):
            continue  # shebang
        if tok.line[:tok.start[1]].strip() == "":
            prev = cols.get(line - 1)
            if prev is not None and prev[0] == tok.start[1]:
                # lined up under the previous line's end-of-line comment: its continuation
                inline[prev[1]] += " " + text
                cols[line] = (tok.start[1], prev[1])
                continue
            full[line] = text
        else:
            inline[line] = text
            cols[line] = (tok.start[1], line)
    return full, inline


class _ConstTranslator(ExprTranslator):
    """Expressions evaluated at module level: constants and default arguments."""


def _used_params(fdef, body_lines, sig) -> set:
    """Parameters the C++ body uses: read in Python, and still named in the generated body
    (comments aside). The others get [[maybe_unused]]: a parameter only assigned, or used
    only in code the translation dropped (the else of `if True:`), would draw a warning."""
    loads = python_reads(fdef)
    loads |= {x.target.id for x in ast.walk(fdef) if isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name)}  # a *= k
    code = [re.sub(r"//.*", "", line) for line in body_lines]

    def read_in_cpp(ident):
        # every mention but the left side of a plain `p = ...;` (a read only in dropped code,
        # the `if False:` branch, leaves just the writes: -Wunused-but-set-parameter)
        word, write = re.compile(rf"\b{re.escape(ident)}\b"), re.compile(rf"^\s*{re.escape(ident)} = ")
        return sum(len(word.findall(line)) - (1 if write.match(line) else 0) for line in code) > 0
    return {p.name for p in sig.params if p.name in loads and read_in_cpp(p.cpp)}


class ModuleTranslator:
    def __init__(self, source: str, module_name: str, filename: str = "<input>", deps: dict | None = None):
        """deps: the already-translated modules of the project (name -> ModuleTranslator) that
        this one may import from; see project.py."""
        self.source = source
        self.module_name = module_name
        self.filename = filename
        self.tree = ast.parse(source, filename)
        self.full_comments, self.inline_comments = _collect_comments(source)
        self.imports: dict = {}
        self.array_users: dict = {}  # function -> whether its translation used any array value
        self.line_notes: dict = {}   # function -> [((first, last) Python lines, key, text)] (for explain)
        self.constants: dict = {}
        self.skipped_globals: dict = {}
        self.sigs: dict = {}
        self.fdefs: dict = {}
        self.rejected: dict = {}
        self.report = ModuleReport(module_name, Path(filename).name)
        self.doc = None
        self.type_aliases: set = set()     # module-level `Vec = ...` / `Mat = ...`
        self.returns_alias: dict = {}      # function -> indices of array params it may return
        self.errstate_funcs: set[str] = set()
        self.classes: dict = {}            # name -> classes.ClassInfo (stage 5)
        # multi-module projects (0.8)
        self.deps: dict = dict(deps or {})
        ns = cpp_ident(re.sub(r"\W", "_", module_name) or "module")
        self.ns = "m_" + ns if ns[0].isdigit() else ns
        self.imported: dict = {}           # local name -> (owner module, name there, kind)
        self.module_aliases: dict = {}     # `import bl_core as C` -> {"C": "bl_core"}
        self.project_deps: list = []       # project modules imported (their headers are included)
        self.ext_modified: dict = {}       # imported function / method -> parameters it modifies
        self.ext_math_domain: set = set()  # imported functions that may raise a math domain error
        self.class_owner: dict = {}        # class name -> namespace of the module defining it
        self.translated: set = set()       # after translate(): the names that got a body

    # --- annotations -----------------------------------------------------------------
    def parse_annotation(self, node, allow_void=False):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node = ast.parse(node.value, mode="eval").body
        if isinstance(node, ast.Constant) and node.value is None:
            if allow_void:
                return VOID
        elif (opt := self._optional_inner(node)) is not None:
            inner = self.parse_annotation(opt)
            if not isinstance(inner, (Scalar, ArrayType)):
                raise Unsupported(node, f"'{ast.unparse(node)}': only a number or an array can be optional")
            return OptionalType(inner)
        elif isinstance(node, ast.Name) and node.id in ("float", "int", "bool", "str"):
            return {"float": FLOAT, "int": INT, "bool": BOOL, "str": STR}[node.id]
        elif isinstance(node, ast.Name) and self.imports.get(node.id) == "numpy.random.Generator":
            return RNG
        elif isinstance(node, ast.Name) and node.id in self.classes:
            c = self.classes[node.id]
            if c.rejected is not None:
                raise Unsupported(node, f"class {c.name} could not be translated: {c.rejected.message}")
            return c.type
        elif isinstance(node, ast.Name) and node.id in ("Vec", "Mat") and (
                self.imports.get(node.id) == f"qtrans.typing.{node.id}" or node.id in self.type_aliases):
            return VEC if node.id == "Vec" else MAT
        elif isinstance(node, ast.Attribute):
            q = _ConstTranslator(self).qualify(node)
            m = {"numpy.float64": FLOAT, "numpy.int64": INT, "numpy.bool_": BOOL,
                 "qtrans.typing.Vec": VEC, "qtrans.typing.Mat": MAT, "numpy.random.Generator": RNG}
            if q in m:
                return m[q]
        elif isinstance(node, ast.Subscript):
            base = node.value
            bname = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if bname in ("tuple", "Tuple"):
                elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                ts = tuple(self.parse_annotation(e) for e in elts)
                if all(isinstance(t, (Scalar, ArrayType)) or t == STR for t in ts) and len(ts) >= 1:
                    return TupleType(ts)
        text = ast.unparse(node)
        if "ndarray" in text or "NDArray" in text:
            raise Unsupported(node, f"type annotation '{text}' doesn't say whether the array is 1-D or 2-D",
                              "annotate with Vec (1-D) or Mat (2-D): `from qtrans.typing import Vec, Mat`")
        raise Unsupported(node, f"type annotation '{text}' is not supported",
                          "use float, int, bool, Vec, Mat, or tuple[...] of those")

    # --- classes (stage 5) -------------------------------------------------------------
    def _register_class(self, node):
        self.class_owner[node.name] = self.module_name
        if any(isinstance(b, ast.Name) and (b.id.endswith("Error") or b.id.endswith("Exception")) for b in node.bases):
            self.report.module_notes.append(f"line {node.lineno}: exception class `{node.name}` skipped "
                                            "(raise ValueError / RuntimeError instead)")
            return
        try:
            is_dc, frozen, eq, nt = cls_.dataclass_options(self, node)
        except Unsupported as e:
            info = cls_.ClassInfo(node.name, node, rejected=e)
            self.classes[node.name] = info
            return
        if not (is_dc or nt):
            self.classes[node.name] = cls_.ClassInfo(node.name, node, eq=False, plain=True)
            return
        self.classes[node.name] = cls_.ClassInfo(node.name, node, frozen=frozen, eq=eq, namedtuple=nt)

    def _collect_classes(self):
        """Fields, defaults and method signatures of every class, once all names are known."""
        own = [i for i in self.classes.values() if not i.external]
        for info in own:
            if info.rejected is None:
                try:
                    cls_.collect_class_body(self, info)
                except Unsupported as e:
                    info.rejected = e
        for info in own:
            if info.rejected is None:
                try:
                    cls_.check_field_types(self, info)
                except Unsupported as e:
                    info.rejected = e
        for info in own:  # method signatures (a plain class's __init__ types its fields)
            if info.rejected is not None:
                continue
            try:
                for m in info.methods.values():
                    if m.name in cls_.DUNDERS_SKIPPED:
                        info.notes.append(f"{m.name} skipped: it only formats text")
                        continue
                    self.fdefs[m.key] = m.fdef
                    self.sigs[m.key] = self.method_signature(info, m)
            except Unsupported as e:
                info.rejected = e
        for info in own:
            if info.rejected is None:
                try:
                    if info.plain:
                        self.infer_init_fields(info)
                    self._field_defaults(info)
                except Unsupported as e:
                    info.rejected = e
        try:
            self.class_order = cls_.struct_order({n: c for n, c in self.classes.items() if not c.external and
                                                  c.rejected is None and all(f.type is not None for f in c.fields)})
        except Unsupported as e:
            for c in own:
                c.rejected = c.rejected or e
            self.class_order = []
        changed = True
        while changed:  # a class holding a rejected class is rejected too
            changed = False
            for info in own:
                if info.rejected is not None:
                    continue
                for f in info.fields:
                    t = f.type.inner if isinstance(f.type, OptionalType) else f.type
                    if isinstance(t, StructType) and self.classes[t.name].rejected is not None:
                        info.rejected = Unsupported(info.node, f"field '{f.name}' holds a {t.name}, which could "
                                                               f"not be translated: {self.classes[t.name].rejected.message}")
                        changed = True
                        break
        for info in own:
            if info.rejected is not None:
                for m in info.methods.values():
                    self.fdefs.pop(m.key, None)
                    self.sigs.pop(m.key, None)

    def infer_init_fields(self, info):
        """Types of a plain class's fields from __init__: each `self.x` is read as a local
        `x` and the usual type inference runs over __init__ (so `self.n = len(w)` is an int,
        `self.total = 0.0` a float). A field it can't type needs an annotation."""
        m = info.methods["__init__"]
        missing = [f for f in info.fields if f.type is None]
        if not missing:
            return
        alias = {f.name: f"qtrans_self_{f.name}" for f in info.fields}

        class AsLocals(ast.NodeTransformer):
            def visit_Attribute(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in alias:
                    return ast.copy_location(ast.Name(id=alias[node.attr], ctx=node.ctx), node)
                return node

        tree = AsLocals().visit(copy.deepcopy(m.fdef))
        sig = self.sigs[m.key]
        flat = Signature(sig.name, sig.cpp, sig.params, VOID, sig.lineno, None)
        tr = FunctionTranslator(self, tree, flat)
        reason = None
        try:
            tr.scan(tr.body)
            tr.check_loop_vars()
            tr.finish_scan()
            tr.infer_types()
        except Unsupported as e:
            reason = e.message
        for f in missing:
            t = tr.local_types.get(alias[f.name])
            if t is None or t == VOID:
                raise Unsupported(m.fdef, f"can't tell the type of field '{f.name}' from __init__"
                                          + (f" ({reason})" if reason else ""),
                                  f"annotate it: self.{f.name}: float = ... (or declare `{f.name}: float` in the class)")
            if t == NONE:
                raise Unsupported(m.fdef, f"field '{f.name}' starts as None", f"annotate it: self.{f.name}: float | None = None")
            f.type = t

    def _field_defaults(self, info):
        tr = _ConstTranslator(self)
        for f in info.fields:
            node = f.default if f.default is not None else f.factory
            if node is None:
                continue
            e = tr.expr_raw(node) if isinstance(f.type, OptionalType) else tr.expr(node)
            if not assignable(f.type, e.type):
                raise Unsupported(node, f"the default for field '{f.name}' is a {e.type} but the field is {f.type}")
            if isinstance(f.type, OptionalType):
                e = tr.to_optional(e, f.type)
            f.default_code = tr.as_double(e).code if f.type == FLOAT else e.code

    def method_signature(self, info, m):
        f = m.fdef
        args = f.args.args[1:] if m.kind != "static" else f.args.args
        stub = ast.FunctionDef(name=f.name, args=ast.arguments(posonlyargs=[], args=args, vararg=None,
                               kwonlyargs=[], kw_defaults=[], kwarg=None,
                               defaults=list(f.args.defaults)[-len(args):] if args and f.args.defaults else []),
                               body=f.body, decorator_list=[], returns=f.returns, lineno=f.lineno)
        if m.kind in ("post_init", "init") and f.returns is None:
            stub.returns = ast.Constant(None)
        sig = self.signature(stub)
        sig.owner, sig.kind = info.name, m.kind
        py_names = {n.id for n in ast.walk(f) if isinstance(n, ast.Name)} | {a.arg for a in f.args.args}
        for prm in sig.params:  # a parameter named like a field would shadow it
            prm.cpp = cls_.member_safe_name(prm.name, info, py_names)
        if m.kind in ("post_init", "init") and sig.ret != VOID:
            raise Unsupported(f, f"{m.name} must return None")
        return sig

    def _rejected_class_in(self, sig):
        for t in [p.type for p in sig.params] + [sig.ret]:
            for x in (t.elts if isinstance(t, TupleType) else [t]):
                x = x.inner if isinstance(x, OptionalType) else x
                if isinstance(x, StructType) and self.classes[x.name].rejected is not None:
                    return Unsupported(self.fdefs[sig.name] if sig.name in self.fdefs else None,
                                       f"uses class {x.name}, which could not be translated: "
                                       f"{self.classes[x.name].rejected.message}")
        if sig.owner and self.classes[sig.owner].rejected is not None:
            return self.classes[sig.owner].rejected
        return None

    def _optional_inner(self, node):
        """X | None, None | X, Optional[X], Union[X, None] -> the X node; otherwise None."""
        def is_none(n):
            return isinstance(n, ast.Constant) and n.value is None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if is_none(node.right) and not is_none(node.left):
                return node.left
            if is_none(node.left) and not is_none(node.right):
                return node.right
        if isinstance(node, ast.Subscript):
            q = self._qualify_path(node.value) if isinstance(node.value, ast.Attribute) else \
                self.imports.get(node.value.id) if isinstance(node.value, ast.Name) else None
            if q == "typing.Optional":
                return node.slice
            if q == "typing.Union" and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                a, b = node.slice.elts
                if is_none(b) and not is_none(a):
                    return a
                if is_none(a) and not is_none(b):
                    return b
        return None

    # --- multi-module projects (0.8) -------------------------------------------------------
    def _import_project(self) -> set:
        """Names imported from other modules of the project: `from m import f, g as h`,
        `from m import *`, `import m` / `import m as alias` (then `alias.f`), `from . import m`,
        `from .m import f`. Each becomes a function, class or constant of that module, seen
        from here: C++ calls go to m::f, types are m::Option. Returns the import statements
        handled here (the others are library imports)."""
        handled = set()
        # names defined here: a namespace of the same name is then reached as ::name
        self._own_top = {n.name for n in self.tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
            t.id for n in self.tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in self.deps:
                        self.module_aliases[a.asname or a.name] = a.name
                        self._dep(a.name)
            elif isinstance(node, ast.ImportFrom) and node.level <= 1:
                if node.level == 1 and node.module is None:  # from . import m
                    mods = [a for a in node.names if a.name in self.deps]
                    if mods and len(mods) == len(node.names):
                        handled.add(id(node))
                        for a in mods:
                            self.module_aliases[a.asname or a.name] = a.name
                            self._dep(a.name)
                    continue
                if node.module not in self.deps:
                    continue
                handled.add(id(node))
                self._dep(node.module)
                for a in node.names:
                    if a.name == "*":
                        self._import_star(node.module, node)
                    else:
                        self._import_name(a.asname or a.name, node.module, a.name, node)
        if self.module_aliases:
            self._rewrite_module_attributes()
        return handled

    def _q(self, ns):
        """How this module names another module's namespace."""
        return f"::{ns}" if ns in getattr(self, "_own_top", ()) else ns

    def _dep(self, module):
        if module not in self.project_deps:
            self.project_deps.append(module)

    def _import_star(self, module, node):
        dep = self.deps[module]
        names = dep.public_names()
        for name in names:
            self._import_name(name, module, name, node)
        for alias, q in dep.imports.items():  # what it imported from libraries comes along too
            if not alias.startswith("_") and (dep.all_names is None or alias in dep.all_names):
                self.imports.setdefault(alias, q)
        for alias, m in dep.module_aliases.items():
            if not alias.startswith("_"):
                self.module_aliases.setdefault(alias, m)

    def _import_name(self, local, module, name, node):
        entry = self.deps[module].namespace.get(name)
        if entry is None:
            self.report.module_notes.append(f"line {node.lineno}: {module} has no translatable name '{name}'")
            return
        kind, owner, oname = entry
        o = self.deps[owner]
        self.class_owner.update(o.class_owner)
        if kind == "class":
            if local != oname:
                self.skipped_globals[local] = f"class {oname} is imported under another name; import it as {oname}"
                return
            self._import_class(o, oname)
        elif kind == "function":
            if oname in o.translated:
                sig = self._from_module(o.sigs[oname], o, qualify=True)
                sig.module = owner
                self.sigs[local] = sig
                self.ext_modified[local] = set(o.modified.get(oname, ()))
                if oname in o.returns_alias:
                    self.returns_alias[local] = set(o.returns_alias[oname])
                if oname in o.math_domain:
                    self.ext_math_domain.add(local)
            else:
                e = o.rejected.get(oname)
                why = e.message if e is not None else "it could not be translated"
                self.rejected[local] = Unsupported(None, f"{oname} (in {owner}) could not be translated: {why}")
        elif kind == "constant":
            c = o.constants[oname]
            self.constants[local] = Constant(f"{self._q(o.ns)}::{c.cpp}", self._view(c.type), c.code, external=True)
        else:
            self.skipped_globals[local] = f"{o.skipped_globals.get(oname, 'not translated')} (in {owner})"
        self.imported[local] = (owner, oname, kind)

    def _import_class(self, o, name):
        """A class of module o, as seen here: m::Option, with the methods' signatures, and any
        class it holds as a field."""
        if name in self.classes:
            return
        info = o.classes[name]
        c = copy.copy(info)
        c.ns, c.external, c.notes = self._q(o.ns), True, []
        c.fields = []
        for f in info.fields:
            g = copy.copy(f)
            g.type = self._view(f.type) if f.type is not None else None
            g.default_code = o.qualify_code(f.default_code, self._q(o.ns))
            c.fields.append(g)
        c.methods = dict(info.methods)
        self.classes[name] = c
        if info.rejected is not None:
            return
        for m in info.methods.values():
            if m.key in o.translated:
                self.sigs[m.key] = self._from_module(o.sigs[m.key], o, qualify=False)
                self.ext_modified[m.key] = set(o.modified.get(m.key, ()))
        for f in info.fields:  # a class held as a field: its fields and methods are reachable here
            t = f.type.inner if isinstance(f.type, OptionalType) else f.type
            if isinstance(t, StructType) and t.name not in self.classes:
                self._import_class(self.deps[self.class_owner[t.name]], t.name)

    def _view(self, t):
        """A type of another module as seen from here: its classes qualified with their namespace."""
        if isinstance(t, StructType):
            owner = self.class_owner.get(t.name, self.module_name)
            if owner == self.module_name:
                return StructType(t.name, cpp_ident(t.name))
            return StructType(t.name, f"{self._q(self.deps[owner].ns)}::{cpp_ident(t.name)}")
        if isinstance(t, TupleType):
            return TupleType(tuple(self._view(e) for e in t.elts))
        if isinstance(t, OptionalType):
            return OptionalType(self._view(t.inner))
        return t

    def _from_module(self, sig, o, qualify):
        new = copy.deepcopy(sig)
        for p in new.params:
            p.type = self._view(p.type)
            p.default_code = o.qualify_code(p.default_code, self._q(o.ns))
        new.ret = self._view(new.ret)
        if qualify:
            new.cpp = f"{self._q(o.ns)}::{sig.cpp}"
        return new

    def qualify_code(self, code, prefix=None):
        """C++ written for inside this module's namespace, made valid outside it: its own
        constants, classes and functions get the namespace (string literals untouched)."""
        if code is None:
            return None
        names = {c.cpp for c in self.constants.values() if not c.external}
        names |= {cpp_ident(n) for n, i in self.classes.items() if not i.external}
        names |= {self.sigs[n].cpp for n in self.fdefs if n in self.sigs and "." not in n}
        if not names:
            return code
        pat = re.compile(r"(?<![\w:.])(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                         + r")\b")
        parts = re.split(r'("(?:[^"\\]|\\.)*")', code)
        prefix = prefix or self.ns
        return "".join(part if i % 2 else pat.sub(lambda m: f"{prefix}::{m.group(1)}", part)
                       for i, part in enumerate(parts))

    def _rewrite_module_attributes(self):
        """`alias.f(...)` for an imported project module: rewritten to a name imported from it,
        `alias.f` (a name no Python code can clash with), or the class name itself."""
        mod = self

        class Rewrite(ast.NodeTransformer):
            def visit_Attribute(self, node):
                self.generic_visit(node)
                v = node.value
                if isinstance(v, ast.Name) and v.id in mod.module_aliases:
                    target = mod.module_aliases[v.id]
                    entry = mod.deps[target].namespace.get(node.attr)
                    if entry is None:
                        return node
                    local = node.attr if entry[0] == "class" else f"{v.id}.{node.attr}"
                    if local not in mod.imported:
                        mod._dep(target)
                        mod._import_name(local, target, node.attr, node)
                    return ast.copy_location(ast.Name(id=local, ctx=node.ctx), node)
                return node

        Rewrite().visit(self.tree)

    def _build_namespace(self):
        """What other modules can import from this one: (kind, defining module, its name)."""
        ns = {local: (kind, owner, oname) for local, (owner, oname, kind) in self.imported.items()}
        for name in self.fdefs:
            if "." not in name:
                ns[name] = ("function", self.module_name, name)
        for name, info in self.classes.items():
            if not info.external:
                ns[name] = ("class", self.module_name, name)
        for name, c in self.constants.items():
            if not c.external:
                ns[name] = ("constant", self.module_name, name)
        for name in self.skipped_globals:
            if name not in ns:
                ns[name] = ("skipped", self.module_name, name)
        self.namespace = ns
        self.all_names = None
        for node in self.tree.body:  # __all__ = [...] decides what `import *` takes
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets) \
                    and isinstance(node.value, (ast.List, ast.Tuple)):
                self.all_names = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
        # functions that may raise a math domain error (for `try: ... except ValueError` callers)
        tr = _ConstTranslator(self)
        self.math_domain = set(self.ext_math_domain)
        changed = True
        while changed:
            changed = False
            for name, f in self.fdefs.items():
                if name in self.math_domain:
                    continue
                for n in ast.walk(f):
                    if isinstance(n, ast.Call) and (tr.qualify(n.func) in lib.MATH_DOMAIN or (
                            isinstance(n.func, ast.Name) and n.func.id in self.math_domain)):
                        self.math_domain.add(name)
                        changed = True
                        break

    def public_names(self):
        if self.all_names is not None:
            return [n for n in self.namespace if n in self.all_names]
        return [n for n in self.namespace if not n.startswith("_")]

    # --- names visible inside functions -----------------------------------------------
    def global_name(self, node, tr):
        n = node.id
        if n in self.constants:
            c = self.constants[n]
            return Expr(c.cpp, P_ATOM, c.type, simple=True)
        if n in self.sigs:
            raise Unsupported(node, f"using function '{n}' as a value is not supported", "call it instead")
        if n in self.imports:
            q = self.imports[n]
            if q in lib.CONSTANTS:
                return Expr(lib.CONSTANTS[q], P_ATOM, FLOAT, simple=True)
            raise Unsupported(node, f"'{n}' ({q}) can't be used as a value")
        if n in self.skipped_globals:
            raise Unsupported(node, f"module variable '{n}' was not translated: {self.skipped_globals[n]}")
        raise Unsupported(node, f"unknown name '{n}'",
                          "define it in this file, or import it from another module of the project (from m import f)")

    # --- collection ----------------------------------------------------------------------
    def collect(self):
        handled = self._import_project()
        body = self.tree.body
        assigned_count = {}
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            assigned_count[n.id] = assigned_count.get(n.id, 0) + 1

        for i, node in enumerate(body):
            if i == 0 and is_docstring(node):
                self.doc = inspect.cleandoc(node.value.value)
            elif id(node) in handled:
                continue  # an import from another module of the project (see _import_project)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in self.deps:
                        continue
                    if a.asname:
                        self.imports[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.imports[top] = top
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                if node.level:
                    self.report.module_notes.append(f"line {node.lineno}: relative import ignored")
                    continue
                for a in node.names:
                    if a.name == "*":
                        self.report.module_notes.append(f"line {node.lineno}: `from {node.module} import *` ignored")
                        continue
                    self.imports[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.FunctionDef):
                if node.name in self.fdefs:
                    self.report.module_notes.append(f"line {node.lineno}: `{node.name}` is defined twice; "
                                                    "the later definition is used, as in Python")
                self.fdefs[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.collect_constant(node, assigned_count)
            elif isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
                self.report.module_notes.append(f"line {node.lineno}: `if __name__ == ...` block skipped")
            elif isinstance(node, ast.ClassDef):
                self._register_class(node)
            elif isinstance(node, ast.AsyncFunctionDef):
                self.report.module_notes.append(f"line {node.lineno}: async function `{node.name}` skipped")
            else:
                self.report.module_notes.append(f"line {node.lineno}: module-level "
                                                f"{type(node).__name__} statement skipped")

        for name in [n for n in self.imported if n in self.fdefs or
                     (n in self.classes and not self.classes[n].external) or
                     (n in self.constants and not self.constants[n].external)]:
            # defined here after being imported: the module's own definition wins, as in Python
            self.imported.pop(name)
            self.sigs.pop(name, None)
            self.rejected.pop(name, None)
            self.skipped_globals.pop(name, None)
        self._collect_classes()
        for f in self.fdefs.values():
            self._strip_errstate(f)
        for name, f in self.fdefs.items():
            if name in self.rejected or name in self.sigs:  # methods already have theirs
                continue
            try:
                self.sigs[name] = self.signature(f)
            except Unsupported as e:
                self.rejected[name] = e
        self.analyze_array_params()

    def _qualify_path(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or node.id not in self.imports:
            return None
        return ".".join([self.imports[node.id]] + parts[::-1])

    def is_asarray_of(self, value, name) -> bool:
        """`np.asarray(name)` / `np.asarray(name, dtype=float)`: for a float64 array NumPy returns
        the array itself, so `name = np.asarray(name, ...)` changes nothing."""
        if not (isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Name)
                and value.args[0].id == name and len(value.args) == 1):
            return False
        f = value.func
        q = self._qualify_path(f) if isinstance(f, ast.Attribute) else self.imports.get(getattr(f, "id", None))
        if q != "numpy.asarray":
            return False
        return all(k.arg == "dtype" and ast.unparse(k.value) in ("float", "np.float64", "numpy.float64")
                   for k in value.keywords)

    def _strip_errstate(self, f):
        """`with np.errstate(...):` only decides whether NumPy warns about inf/NaN; the values
        are the same either way. Replace each such block by its body (Python's `with` opens no
        new scope, so variables assigned inside stay visible after it)."""
        mod = self

        class Strip(ast.NodeTransformer):
            def visit_With(self, node):
                self.generic_visit(node)
                if node.items and all(
                        it.optional_vars is None and isinstance(it.context_expr, ast.Call)
                        and mod._qualify_path(it.context_expr.func) == "numpy.errstate" for it in node.items):
                    mod.errstate_funcs.add(f.name)
                    return node.body
                return node

        Strip().visit(f)

    @staticmethod
    def _starts_fresh(f, name, sigs) -> bool:
        """Is the first statement that mentions array parameter `name` a plain rebinding to a
        new array (`S = S.copy()`, `X = X - X.mean()`)? Then every later in-place change hits the
        new array, never the caller's, exactly as in Python."""
        body = f.body[1:] if f.body and is_docstring(f.body[0]) else f.body
        for stmt in body:
            if not any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(stmt)):
                continue
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
                return False
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name) or targets[0].id != name:
                return False
            value = stmt.value
            types = {p.name: p.type for p in sigs[f.name].params}
            branches = [value.body, value.orelse] if isinstance(value, ast.IfExp) else [value]
            if any(view_root(b, types) == name for b in branches):
                return False
            # passing it to one of our functions could modify it (or hand it back)
            for n in ast.walk(value):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in sigs and any(
                        isinstance(a, ast.Name) and a.id == name for a in list(n.args) + [k.value for k in n.keywords]):
                    return False
            return True
        return False

    def _shareable(self, t) -> bool:
        """Types Python shares rather than copies on assignment and calls."""
        return is_array(t) or t in (RNG, STR) or _optional_array(t) or isinstance(t, StructType)

    def immutable(self, t) -> bool:
        """A frozen object with no array anywhere inside: nothing about it can change. (Frozen
        only stops fields being reassigned; an array field can still be changed in place.)"""
        if not isinstance(t, StructType):
            return False
        info = self.classes[t.name]
        return info.frozen and not info.has_array_field(self.classes) and all(f.type != RNG for f in info.fields) \
            and all(self.immutable(f.type) for f in info.fields if isinstance(f.type, StructType))

    def chain_type(self, node, ptypes):
        """The static type of `p`, `p.field`, `self.inner.w` for a parameter p; else None."""
        if isinstance(node, ast.Name):
            return ptypes.get(node.id)
        if isinstance(node, ast.Attribute):
            t = self.chain_type(node.value, ptypes)
            if isinstance(t, StructType) and t.name in self.classes:
                f = self.classes[t.name].field(node.attr)
                return f.type if f is not None else None
        return None

    def _param_types(self, name):
        sig = self.sigs[name]
        pt = {p.name: p.type for p in sig.params}
        if sig.owner and sig.kind in ("instance", "property", "post_init", "init"):
            pt["self"] = self.classes[sig.owner].type
        return pt

    def analyze_array_params(self):
        """Decide how each array or object parameter is passed (and which methods are const) so
        the C++ behaves like Python, where arrays and objects are shared, not copied, when
        passed to a function. A parameter (or self) is modified if something stores into it
        (an element, a field), calls a method that modifies it, or passes it, or one of its
        fields, to a function that modifies that argument; this is iterated to a fixpoint."""
        inplace, rebound, calls = {}, {}, {}
        for name, f in self.fdefs.items():
            if name not in self.sigs:
                continue
            sig = self.sigs[name]
            ptypes = self._param_types(name)
            arr = {n for n, t in ptypes.items() if self._shareable(t)}
            ip, rb, cl = set(), set(), []
            for n in ast.walk(f):
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        for leaf in target_leaves(t):
                            if isinstance(leaf, (ast.Subscript, ast.Attribute)):
                                root = subscript_root(leaf)
                                if root in arr:
                                    ip.add(root)
                            elif isinstance(leaf, ast.Name) and leaf.id in arr:
                                if isinstance(n, ast.Assign) and self.is_asarray_of(n.value, leaf.id):
                                    continue  # x = np.asarray(x): the same array, not a rebinding
                                (ip if isinstance(n, ast.AugAssign) else rb).add(leaf.id)
                elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id in arr:
                    rb.add(n.target.id)
                elif isinstance(n, ast.Call):
                    if ast.unparse(n.func) == "object.__setattr__" and n.args and \
                            isinstance(n.args[0], ast.Name) and n.args[0].id in arr:
                        ip.add(n.args[0].id)
                    callee = None
                    if isinstance(n.func, ast.Attribute) and self.chain_type(n.func.value, ptypes) == RNG:
                        root = subscript_root(n.func.value)  # self.rng.normal(): drawing changes self
                        if root in arr:
                            ip.add(root)
                    if isinstance(n.func, ast.Name) and n.func.id in self.sigs:
                        callee = n.func.id
                    elif isinstance(n.func, ast.Attribute):
                        bt = self.chain_type(n.func.value, ptypes)
                        if isinstance(bt, StructType) and bt.name in self.classes:
                            m = self.classes[bt.name].methods.get(n.func.attr)
                            if m is not None and m.key in self.sigs:
                                callee = m.key
                                root = subscript_root(n.func.value) if not isinstance(n.func.value, ast.Name) \
                                    else n.func.value.id
                                if root in arr and m.kind in ("instance",):
                                    cl.append((callee, "self", root))
                    if callee is not None:
                        cs = self.sigs[callee]
                        pos = {p.name: i for i, p in enumerate(cs.params)}
                        pairs = list(enumerate(n.args)) + [(pos.get(k.arg, -1), k.value) for k in n.keywords]
                        for i, a in pairs:
                            root = a.id if isinstance(a, ast.Name) else (
                                subscript_root(a) if isinstance(a, (ast.Attribute, ast.Subscript)) else None)
                            if root in arr and i >= 0:
                                cl.append((callee, i, root))
            # `S = S.copy()` first, then `S += ...`: the changes never reach the caller
            fresh = {p for p in ip & rb if self._starts_fresh(f, p, self.sigs)}
            ip -= fresh
            cl = [c for c in cl if c[2] not in fresh]
            inplace[name], rebound[name], calls[name] = ip, rb, cl
            # which parameters a function may hand straight back (return a / return a[1:])
            for n in ast.walk(f):
                if isinstance(n, ast.Return) and n.value is not None:
                    root = view_root(n.value, {p.name: p.type for p in sig.params})
                    if root in arr and root != "self":
                        idx = next(i for i, p in enumerate(sig.params) if p.name == root)
                        self.returns_alias.setdefault(name, set()).add(idx)
        changed = True
        while changed:  # passing an array or object to a function that modifies it modifies it too
            changed = False
            for name, lst in calls.items():
                for callee, i, arg in lst:
                    cs = self.sigs[callee]
                    target = "self" if i == "self" else (cs.params[i].name if i < len(cs.params) else None)
                    changes = inplace[callee] if callee in inplace else self.ext_modified.get(callee, ())
                    if target is not None and target in changes and arg not in inplace[name]:
                        inplace[name].add(arg)
                        changed = True
        self.modified = inplace
        for name, ip in inplace.items():
            sig = self.sigs[name]
            if sig.owner:
                info = self.classes[sig.owner]
                if sig.kind in ("instance", "property"):
                    # (assigning a field of a frozen object is refused where it happens; changing an
                    # array inside one is allowed, in Python as here)
                    sig.const = "self" not in ip
                    if not sig.const and sig.kind == "property":
                        self.rejected[name] = Unsupported(self.fdefs[name], f"property {sig.name} modifies self",
                                                          "make it a method")
            for p in sig.params:
                if p.type == RNG:
                    # drawing advances the caller's generator, as in Python: pass by reference
                    if p.name in rebound[name]:
                        self.rejected[name] = Unsupported(
                            self.fdefs[name], f"generator parameter '{p.name}' is reassigned",
                            "use a new name for the new generator")
                    p.mode = "ref"
                    continue
                if p.type == STR:  # strings are immutable: read by reference unless rebound
                    p.mode = "value" if p.name in rebound[name] else "cref"
                    continue
                if _optional_array(p.type):
                    # an array that may be None: read it, or rebind the name, but don't modify it
                    # in place (std::optional<qt::Mat> holds its own copy, Python shares the array)
                    if p.name in ip:
                        self.rejected[name] = Unsupported(
                            self.fdefs[name], f"'{p.name}' may be None and is modified in place; that is "
                            "not supported", f"check for None, then work on a copy: {p.name}_ = {p.name}.copy()")
                    p.mode = "value" if p.name in rebound[name] else "cref"
                    continue
                if isinstance(p.type, StructType) and self.immutable(p.type):
                    p.mode = "value" if p.name in rebound[name] else "cref"  # can't change: never copied
                    continue
                if not (is_array(p.type) or isinstance(p.type, StructType)):
                    continue
                what = "array" if is_array(p.type) else "object"
                if p.name in ip and p.name in rebound[name]:
                    self.rejected[name] = Unsupported(
                        self.fdefs[name], f"{what} parameter '{p.name}' is both modified in place and "
                        "reassigned: in Python the caller sees changes made before the reassignment "
                        "but not after it", f"work on a copy: {p.name}_ = {p.name}.copy()" if what == "array"
                        else f"work on a copy: {p.name}_ = copy.deepcopy({p.name})")
                p.mode = "ref" if p.name in ip else "value" if p.name in rebound[name] else "cref"
        self._check_post_init_sharing()

    def _check_post_init_sharing(self):
        """__post_init__ changing an array or object field in place changes the caller's own
        array in Python (the field *is* the argument); the C++ object holds a copy."""
        for info in self.classes.values():
            m = info.methods.get("__init__" if info.plain else "__post_init__")
            if info.external or info.rejected is not None or m is None or m.key not in self.sigs:
                continue
            shared = {f.name for f in info.init_fields if is_array(f.type) or isinstance(f.type, StructType)}
            if info.plain:  # self.w = w (or a view of w): the field *is* the caller's array
                ptypes = {p.name: p.type for p in self.sigs[m.key].params}
                for n in ast.walk(m.fdef):
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute) \
                            and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self" \
                            and view_root(n.value, ptypes) is not None:
                        shared.add(n.targets[0].attr)
            for n in ast.walk(m.fdef):
                hit = None
                if isinstance(n, (ast.Assign, ast.AugAssign)):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        for leaf in target_leaves(t):
                            node = leaf
                            # self.w[i] = ..., self.w += ..., self.inner.x = ...: dig to self.<field>
                            while isinstance(node, (ast.Subscript, ast.Attribute)) and not (
                                    isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                                    and node.value.id == "self"):
                                node = node.value
                            if isinstance(node, ast.Attribute) and node.attr in shared and (
                                    node is not leaf or isinstance(n, ast.AugAssign)):
                                hit = node.attr
                if hit:
                    info.rejected = Unsupported(
                        n, f"__post_init__ changes field '{hit}' in place: in Python that changes the array or "
                           "object the caller passed in, which the C++ object holds a copy of",
                        f"assign a new value instead, e.g. self.{hit} = self.{hit} / self.{hit}.sum()")
                    break

    def collect_constant(self, node, assigned_count):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            self.report.module_notes.append(f"line {node.lineno}: module-level assignment skipped")
            return
        name = targets[0].id
        if name in ("Vec", "Mat"):
            self.type_aliases.add(name)  # e.g. Vec = npt.NDArray[np.float64]
            return
        if name.startswith("QTRANS_"):
            return  # qtrans' own settings (e.g. QTRANS_CASES for the verifier)
        if node.value is None:
            return
        if assigned_count.get(name, 0) > 1:
            self.skipped_globals[name] = "it is assigned more than once, so it isn't a constant"
            self.report.module_notes.append(f"line {node.lineno}: `{name}` assigned more than once; not a constant")
            return
        tr = _ConstTranslator(self)
        try:
            e = tr.expr(node.value)
            if not isinstance(e.type, Scalar) and e.type != STR:
                raise Unsupported(node, f"{e.type} constants are not supported")
            t = e.type
            if isinstance(node, ast.AnnAssign):
                t = self.parse_annotation(node.annotation)
                if not assignable(t, e.type):
                    raise Unsupported(node, f"`{name}` is annotated {t} but assigned a {e.type}")
            code = tr.as_double(e).code if t == FLOAT else e.code
        except Unsupported as err:
            self.skipped_globals[name] = err.message
            self.report.module_notes.append(f"line {node.lineno}: `{name}` not translated: {err.message}")
            return
        self.constants[name] = Constant(cpp_ident(name), t, code, lineno=node.lineno)
        self.report.constants.append(f"{name}: {t}")

    def signature(self, f):
        a = f.args
        if a.vararg or a.kwarg or a.kwonlyargs:
            raise Unsupported(f, "*args, **kwargs and keyword-only parameters are not supported")
        tr = _ConstTranslator(self)
        notes = []
        for dec in f.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            q = tr.qualify(target)
            if q in lib.DROPPABLE_DECORATORS:
                notes.append(lib.DROPPABLE_DECORATORS[q])
            else:
                raise Unsupported(dec, f"decorator @{ast.unparse(dec)} is not supported")
        nodes = list(a.posonlyargs) + list(a.args)
        defaults = [None] * (len(nodes) - len(a.defaults)) + list(a.defaults)
        params = []
        for arg, d in zip(nodes, defaults):
            if arg.annotation is None:
                raise Unsupported(arg, f"parameter '{arg.arg}' has no type annotation",
                                  f"write `{arg.arg}: float` (or int / bool) for a number, or "
                                  f"`{arg.arg}: Vec` / `{arg.arg}: Mat` for a 1-D / 2-D float array "
                                  f"(from qtrans.typing import Vec, Mat)")
            t = self.parse_annotation(arg.annotation)
            if not isinstance(t, (Scalar, ArrayType, RngType, OptionalType, StructType)) and t != STR:
                raise Unsupported(arg, f"parameter '{arg.arg}': {t} parameters are not supported; "
                                       "use float, int, bool, Vec, Mat or np.random.Generator")
            if d is not None and isinstance(t, RngType):
                raise Unsupported(d, f"generator parameter '{arg.arg}' can't have a default value")
            code = None
            if d is not None and isinstance(t, (ArrayType, StructType)):
                raise Unsupported(d, f"parameter '{arg.arg}' can't have a default {t}: Python creates it once "
                                     "and shares it between calls")
            if d is not None:
                e = tr.expr_raw(d) if isinstance(t, OptionalType) else tr.expr(d)
                if not assignable(t, e.type):
                    raise Unsupported(d, f"default for '{arg.arg}' is a {e.type} but the parameter is {t}")
                code = tr.as_double(e).code if FLOAT in (t, getattr(t, "inner", None)) and e.type != NONE \
                    else e.code
            params.append(Param(arg.arg, cpp_ident(arg.arg), t, code))
        if f.returns is None:
            raise Unsupported(f, f"function '{f.name}' has no return annotation",
                              "add -> float (or int, bool, tuple[float, float], None)")
        ret = self.parse_annotation(f.returns, allow_void=True)
        if ret == RNG:
            raise Unsupported(f, "returning a generator is not supported", "create it where it is used")
        doc = inspect.cleandoc(f.body[0].value.value) if f.body and is_docstring(f.body[0]) else None
        sig = Signature(f.name, cpp_ident(f.name), params, ret, f.lineno, doc)
        sig.decorator_notes = notes
        return sig

    # --- translation ---------------------------------------------------------------------------
    def translate(self) -> TranslationResult:
        self.collect()
        bodies, notes, general = {}, {}, {}
        while True:  # a function calling a rejected function is rejected too: iterate
            progress = False
            for name, f in self.fdefs.items():
                if name in self.rejected or name in bodies:
                    continue
                bad = self._rejected_class_in(self.sigs[name])
                if bad is not None:
                    self.rejected[name] = bad
                    progress = True
                    continue
                tr = FunctionTranslator(self, f, self.sigs[name])
                try:
                    bodies[name] = tr.translate()
                    self.array_users[name] = tr.used_arrays
                    self.line_notes[name] = [(src, key, re.sub(r"^line \d+: ", "", text))
                                             for src, key, text in getattr(tr, "line_notes", [])]
                    notes[name] = list(self.sigs[name].decorator_notes)
                    for key, text in tr.notes:
                        if key in GENERAL_NOTES:
                            general.setdefault(text, []).append(name)
                        else:
                            notes[name].append(text)
                except Unsupported as e:
                    self.rejected[name] = e
                    progress = True
            for info in self.classes.values():  # one untranslatable method sinks its class
                if info.rejected is None and not info.external:
                    bad = next((m for m in info.methods.values() if m.key in self.rejected), None)
                    if bad is not None:
                        e = self.rejected[bad.key]
                        info.rejected = Unsupported(bad.fdef, f"method {bad.name} could not be translated: "
                                                              f"{e.message}", e.hint)
                        progress = True
            if not progress:
                break
            bodies, notes, general = {}, {}, {}  # retranslate: a call may now point at a rejected function

        for info in self.classes.values():
            if info.external:
                continue
            if info.rejected is not None:
                e = info.rejected
                self.report.functions.append(FunctionReport(f"class {info.name}", info.node.lineno, False,
                                                            e.message, e.lineno, e.hint))
            else:
                self.report.functions.append(FunctionReport(f"class {info.name}", info.node.lineno, True,
                                                            notes=list(info.notes)))
        for name, f in self.fdefs.items():
            owner = self.sigs[name].owner if name in self.sigs else None
            if owner and self.classes[owner].rejected is not None:
                continue  # reported with its class
            if name in bodies:
                self.report.functions.append(FunctionReport(name, f.lineno, True, notes=notes[name]))
            else:
                e = self.rejected[name]
                self.report.functions.append(FunctionReport(name, f.lineno, False, e.message, e.lineno, e.hint))

        for text, names in general.items():
            self.report.general_notes.append(f"{text} (applies to: {', '.join(names)})")
        header = self.render(bodies)
        sigs = {n: self.sigs[n] for n in bodies}
        self.translated = set(bodies)
        self._build_namespace()
        uses_random = "qtrans_random.hpp" in header
        live = {n: self.classes[n] for n in getattr(self, "class_order", []) if self.classes[n].rejected is None}
        return TranslationResult(self.module_name, header, self.report, sigs, "qt::" in header,
                                 "qtrans_arrays.hpp" in header or uses_random, uses_random, live, self.ns,
                                 list(self.project_deps), self.line_map[:-1] if header.endswith("\n") else self.line_map,
                                 {n: v for n, v in self.line_notes.items() if n in bodies})

    # --- rendering -----------------------------------------------------------------------------
    def attached_comments(self, first_line):
        """The block of full-line comments directly above a definition."""
        out, ln = [], first_line - 1
        while ln in self.full_comments:
            out.append(self.full_comments[ln])
            ln -= 1
        return list(reversed(out))

    def cpp_signature(self, sig, defaults, used_params=None, name=None):
        parts = []
        for p in sig.params:
            attr = "[[maybe_unused]] " if used_params is not None and p.name not in used_params else ""
            if is_array(p.type) or p.type in (RNG, STR) or _optional_array(p.type) or isinstance(p.type, StructType):
                ty = {"cref": f"const {p.type.cpp()}&", "ref": f"{p.type.cpp()}&"}.get(p.mode, p.type.cpp())
            else:
                ty = p.type.cpp()
            s = f"{attr}{ty} {p.cpp}"
            if defaults and p.default_code is not None:
                s += f" = {p.default_code}"
            parts.append(s)
        return f"{sig.ret.cpp()} {name or sig.cpp}({', '.join(parts)})"

    # --- structs ------------------------------------------------------------------------------
    def _ctor_params(self, info, defaults):
        if info.plain:  # the constructor is __init__, with its own parameters
            sig = self.sigs[info.methods["__init__"].key]
            parts = self.cpp_signature(sig, defaults=defaults)
            inner = parts[parts.index("(") + 1:parts.rindex(")")]
            return [(None, None, inner)] if inner else []
        names = {f.cpp for f in info.fields}
        out = []
        for f in info.init_fields:
            pname = f.cpp + "_"
            while pname in names:
                pname += "_"
            ty = f.type.cpp()
            if is_array(f.type) or isinstance(f.type, StructType) or f.type == STR or _optional_array(f.type):
                ty = f"{ty} "  # by value: the object keeps its own copy (see the sharing analysis)
            d = f" = {f.default_code}" if defaults and f.default_code is not None else ""
            out.append((f, pname, f"{ty.strip()} {pname}{d}"))
        return out

    def render_struct(self, info, bodies) -> list:
        """The struct: fields, constructors and member declarations. Member functions are
        defined after the module's function declarations, so they can call anything."""
        out = []
        if info.doc:
            d = info.node.body[0]
            out += [("/// " + line).rstrip() + source_mark(d.lineno, d.end_lineno)
                    for line in inspect.cleandoc(info.doc).splitlines()]
        kind = "NamedTuple" if info.namedtuple else "class" if info.plain else \
            ("frozen dataclass" if info.frozen else "dataclass")
        cls = info.node.lineno
        top = min([cls] + [d.lineno for d in info.node.decorator_list])
        start = len(out)
        out.append(f"struct {info.cpp} {{  // {kind}" + source_mark(top, cls))
        for f in info.fields:
            note = "" if f.init else "  // init=False: set by __post_init__ or its default"
            ln = getattr(f.annotation, "lineno", None)
            out.append(f"    {f.type.cpp()} {f.cpp}{{}};{note}" + (source_mark(ln, getattr(f.annotation, "end_lineno", ln)) if ln else ""))
        out.append("")
        params = self._ctor_params(info, defaults=True)
        if params or not info.plain:
            out.append(f"    {info.cpp}() = default;  // only so a variable can be declared before it is assigned")
        if info.plain:
            sig = self.sigs[info.methods["__init__"].key]
            required = sum(1 for p in sig.params if p.default_code is None)
            if not sig.params:
                params = []
        else:
            required = sum(1 for f, _, _ in params if f.default_code is None)
        explicit = "explicit " if params and required <= 1 else ""  # no silent conversions
        out.append(f"    {explicit}{info.cpp}({', '.join(p for _, _, p in params)});")
        for m in info.methods.values():
            if m.key not in bodies or m.kind in ("post_init", "init"):
                continue
            sig = self.sigs[m.key]
            decl = self.cpp_signature(sig, defaults=True)
            if m.kind in ("static", "class"):
                decl = "static " + decl
            elif sig.const:
                decl += " const"
            out.append(f"    {decl};" + ("  // @property" if m.kind == "property" else ""))
        if info.eq and not info.has_array_field(self.classes) and any(f.compare for f in info.fields):
            out.append(f"    bool operator==(const {info.cpp}& other) const;")
            out.append(f"    bool operator!=(const {info.cpp}& other) const {{ return !(*this == other); }}")
        out.append("};")
        # the struct's other lines (constructors, member declarations, the closing brace) belong
        # with the class line in the source map; fields carry their own lines
        mark = source_mark(cls, cls)
        out[start:] = [l if not l.strip() or "\x1f" in l else l + mark for l in out[start:]]
        out.append("")
        return out

    def render_members(self, info, bodies) -> list:
        out = []
        params = self._ctor_params(info, defaults=False)
        if info.plain:
            inits = []
        else:
            inits = [f"{f.cpp}(std::move({pname}))" if not isinstance(f.type, Scalar) else f"{f.cpp}({pname})"
                     for f, pname, _ in params]
        for f in info.fields:
            if not f.init and f.default_code is not None:
                inits.append(f"{f.cpp}({f.default_code})")
        head = f"inline {info.cpp}::{info.cpp}({', '.join(p for _, _, p in params)})"
        post = info.methods.get("__init__" if info.plain else "__post_init__")
        body = bodies.get(post.key) if post is not None else None
        if inits:
            head += "\n    : " + ", ".join(inits)
        cls = info.node.lineno
        at = (post.fdef.lineno, post.fdef.lineno) if post is not None and info.plain else (cls, cls)
        out.append((head + (" {" if body else " {}")).replace("\n", source_mark(*at) + "\n") + source_mark(*at))
        if body:
            out.append(("    // __init__" if info.plain else "    // __post_init__") + source_mark(post.fdef.lineno, post.fdef.lineno))
            out += body
            out.append("}" + (source_mark(post.fdef.end_lineno, post.fdef.end_lineno) if post is not None else ""))
        out.append("")
        for m in info.methods.values():
            if m.key not in bodies or m.kind in ("post_init", "init"):
                continue
            sig = self.sigs[m.key]
            if sig.doc:
                out += [("/// " + line).rstrip() for line in sig.doc.splitlines()]
            used = _used_params(m.fdef, bodies[m.key], sig)
            line = "inline " + self.cpp_signature(sig, defaults=False, used_params=used,
                                                   name=f"{info.cpp}::{sig.cpp}")
            if m.kind not in ("static", "class") and sig.const:
                line += " const"
            fd = m.fdef
            first = min([fd.lineno] + [d.lineno for d in fd.decorator_list])
            out.append(line + " {" + source_mark(first, fd.body[0].lineno - 1 if fd.body[0].lineno > fd.lineno else fd.lineno))
            out += bodies[m.key]
            out.append("}" + source_mark(fd.end_lineno, fd.end_lineno))
            out.append("")
        if info.eq and not info.has_array_field(self.classes) and any(f.compare for f in info.fields):
            terms = [f"{f.cpp} == other.{f.cpp}" for f in info.fields if f.compare]
            eq_mark = source_mark(cls, cls)  # dataclass equality: from the class itself
            out.append(f"inline bool {info.cpp}::operator==(const {info.cpp}& other) const {{" + eq_mark)
            out.append(f"    return {' && '.join(terms)};" + eq_mark)
            out.append("}" + eq_mark)
            out.append("")
        return out

    def render(self, bodies) -> str:
        ns = self.ns
        out = [f"// {self.module_name}.hpp: generated by qtrans {VERSION} from {Path(self.filename).name}.",
               "// Edit the Python source and re-run qtrans rather than editing this file."]
        if self.doc:
            out.append("//")
            out += [("// " + line).rstrip() for line in self.doc.splitlines()]
        out += ["#pragma once", "", "@@INCLUDES@@", "", f"namespace {ns} {{", ""]

        own_constants = [c for c in self.constants.values() if not c.external]
        if own_constants:
            for c in own_constants:
                for line in self.attached_comments(c.lineno) if c.lineno else []:
                    out.append("// " + line)
                tail = f"  // {self.inline_comments[c.lineno]}" if c.lineno in self.inline_comments else ""
                out.append(f"inline const {c.type.cpp()} {c.cpp} = {c.code};{tail}" + (source_mark(c.lineno, c.lineno) if c.lineno else ""))
            out.append("")

        live = [self.classes[n] for n in getattr(self, "class_order", []) if self.classes[n].rejected is None]
        for info in live:
            out += self.render_struct(info, bodies)
        names = [n for n in self.fdefs if n in bodies and not self.sigs[n].owner]
        forward = len(names) > 1 or (live and names)
        if forward:
            out.append("// Declarations (so functions can call each other in any order).")
            for n in names:
                out.append(self.cpp_signature(self.sigs[n], defaults=True) + ";")
            out.append("")

        for info in live:
            out += self.render_members(info, bodies)

        for n in names:
            f, sig = self.fdefs[n], self.sigs[n]
            first = min([f.lineno] + [d.lineno for d in f.decorator_list])
            for k, c in enumerate(self.attached_comments(first)):
                ln = first - len(self.attached_comments(first)) + k
                out.append("// " + c + source_mark(ln, ln))
            head_end = f.body[0].lineno - 1 if f.body[0].lineno > f.lineno else f.lineno
            if sig.doc:
                d = f.body[0]
                out += [("/// " + line).rstrip() + source_mark(d.lineno, d.end_lineno) for line in sig.doc.splitlines()]
            used = _used_params(f, bodies[n], sig)
            out.append("inline " + self.cpp_signature(sig, defaults=not forward, used_params=used) + " {"
                       + source_mark(first, head_end))
            out += bodies[n]
            out.append("}" + source_mark(f.end_lineno, f.end_lineno))
            out.append("")

        out.append(f"}}  // namespace {ns}")
        text = "\n".join(out) + "\n"

        includes = ["<cmath>"]
        checks = [("<algorithm>", r"std::(max|min)\("), ("<cstdlib>", r"std::abs\("),
                  ("<stdexcept>", r"throw std::"), ("<tuple>", r"std::(tuple|tie|ignore|make_tuple)"),
                  ("<initializer_list>", r"std::(max|min)\(\{"), ("<string>", r"std::string"),
                  ("<optional>", r"std::optional"), ("<utility>", r"std::move\(")]
        for inc, pat in checks:
            if re.search(pat, text):
                includes.append(inc)
        inc_lines = [f"#include {i}" for i in includes]
        if "qt::Rng" in text:
            inc_lines.append('#include "qtrans_random.hpp"  // NumPy\'s generator, plus Eigen arrays')
        elif re.search(r"qt::(Vec|Mat|Mask2?)\b", text) or "Eigen::" in text \
                or any(self.array_users.get(n) for n in bodies):  # e.g. qt::dot(qt::vec({...}), ...) only
            inc_lines.append('#include "qtrans_arrays.hpp"  // Eigen, plus NumPy semantics')
        elif "qt::" in text:
            inc_lines.append('#include "qtrans_runtime.hpp"')
        for d in self.project_deps:  # the modules of the project it imports from
            inc_lines.append(f'#include "{d}.hpp"  // from {d} import ...')
        text, self.line_map = strip_marks(text.replace("@@INCLUDES@@", "\n".join(inc_lines)))
        return text


def translate_source(source: str, module_name: str = "module", filename: str = "<input>") -> TranslationResult:
    return ModuleTranslator(source, module_name, filename).translate()


def translate_file(path, out_dir=None) -> TranslationResult:
    """Translate a .py file. If out_dir is given, write <module>.hpp, qtrans_runtime.hpp
    (when needed) and <module>_report.md there. Modules of the project it imports (files next
    to it) are translated too, and their headers written alongside (see project.py)."""
    path = Path(path)
    from .project import discover, translate_project
    if len(discover([path])[0]) > 1:
        return translate_project([path], out_dir).modules[path.stem]
    result = translate_source(path.read_text(encoding="utf-8"), path.stem, str(path))
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{path.stem}.hpp").write_text(result.header, encoding="utf-8")
        (out / f"{path.stem}_report.md").write_text(result.report.to_markdown(), encoding="utf-8")
        if result.uses_runtime:
            (out / "qtrans_runtime.hpp").write_text(RUNTIME_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
        if result.uses_arrays:
            (out / "qtrans_arrays.hpp").write_text(ARRAYS_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
            (out / "qtrans_runtime.hpp").write_text(RUNTIME_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
        if result.uses_random:
            (out / "qtrans_random.hpp").write_text(RANDOM_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
    return result
