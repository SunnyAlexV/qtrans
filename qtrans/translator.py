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
from . import loops
from . import overrides as ovr
from . import hints as hints_
from .functions import FunctionTranslator, is_docstring, python_reads, source_mark, strip_marks, subscript_root, target_leaves, view_root
from .origins import Origins
from .report import FunctionReport, ModuleReport
from .types import (BOOL, FLOAT, INT, MAT, NONE, RNG, STR, VEC, VOID, ArrayType, OptionalType, RngType, Scalar,
                    StructType, TupleType, ListType, DictType, assignable, is_array)

VERSION = "0.9.0"
PROGRAM = "(program)"  # the report's name for a program's top-level code (int main())

# Notes about behaviour that differs only on overflow or error paths, shared by most numeric
# code. Reported once per module instead of under every function.
GENERAL_NOTES = {"int64", "div0", "powoverflow"}
RUNTIME_HEADER = Path(__file__).parent / "runtime" / "qtrans_runtime.hpp"
ARRAYS_HEADER = Path(__file__).parent / "runtime" / "qtrans_arrays.hpp"
RANDOM_HEADER = Path(__file__).parent / "runtime" / "qtrans_random.hpp"
PRINT_HEADER = Path(__file__).parent / "runtime" / "qtrans_print.hpp"
LIST_HEADER = Path(__file__).parent / "runtime" / "qtrans_list.hpp"


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
    program: str | None = None      # a program (top-level code): the whole <module>.cpp, with int main()
    program_map: list = None        # per program line: the Python lines it came from, as line_map
    main_lines: tuple = ()          # the Python lines (first, last) of each of main()'s statements
    library: bool = True            # anything for other code to use (functions, classes, constants)
    prints: frozenset = frozenset()  # translated functions that print (verify compares what they print)


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


def _own_body_nodes(fdef):
    """The nodes of a function's body, not entering functions or classes defined inside it."""
    stack = list(fdef.body)
    while stack:
        n = stack.pop()
        yield n
        stack += [c for c in ast.iter_child_nodes(n)
                  if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))]


def cpp_string_literal(text):
    from .core import cpp_string
    return cpp_string(text)


def _is_main_guard(node) -> bool:
    """`if __name__ == "__main__":` (either way round)."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 \
            or not isinstance(node.test.ops[0], ast.Eq):
        return False
    sides = {ast.unparse(node.test.left), ast.unparse(node.test.comparators[0])}
    return sides in ({"__name__", "'__main__'"}, {"__name__", '"__main__"'})


def _runs(node, index) -> bool:
    """A module-level statement that does something when the file runs (not a definition)."""
    if isinstance(node, ast.Expr):
        return not isinstance(node.value, ast.Constant)  # a docstring or bare literal does nothing
    if isinstance(node, ast.If):
        return "__name__" not in ast.unparse(node.test)
    return isinstance(node, (ast.For, ast.While, ast.With, ast.Try, ast.AugAssign, ast.Assert, ast.Raise,
                             ast.Delete, ast.Match))


def _stays_module_level(node, fn_reads) -> bool:
    """In a program, an assignment stays a module constant only when a function reads it (or it
    is qtrans' own setting, or a type alias); the program's other variables are main()'s."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return False
    name = targets[0].id
    return name in fn_reads or name.startswith("QTRANS_") or name in ("Vec", "Mat", "__all__")


def library_tree(tree):
    """The module without its program part (the top-level code that becomes int main()): what
    the verifier imports to call the functions, so that importing doesn't run the program."""
    body = tree.body
    if not any(_is_main_guard(n) or _runs(n, i) for i, n in enumerate(body)):
        return tree
    fn_reads = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            fn_reads |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    keep = [n for i, n in enumerate(body) if not (_is_main_guard(n) or _runs(n, i) or (
        isinstance(n, (ast.Assign, ast.AnnAssign)) and not _stays_module_level(n, fn_reads)))]
    return ast.Module(body=keep, type_ignores=[])


class _MainTranslator(FunctionTranslator):
    """The program's top-level code, as the body of int main(). Comments are carried over from
    its own lines only (not from the functions and classes between its statements)."""

    def __init__(self, mod, fdef, sig, skip):
        super().__init__(mod, fdef, sig)
        self.skip = skip  # Python lines that belong to definitions
        self.comment_floor = fdef.comment_floor

    def flush_comments(self, upto_line):
        full = self.mod.full_comments
        for ln in range(self.comment_floor + 1, upto_line):
            if ln in full and ln not in self.skip:
                self.lines.append("    " * self.depth + "// " + full[ln] + source_mark(ln, ln))
        self.comment_floor = max(self.comment_floor, upto_line - 1)


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


# The C++ computes Python's numbers only if the compiler keeps every floating-point operation as
# written: fast-math reorders and approximates them. Built that way, stop with the reason.
FAST_MATH_GUARD = """#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "built with fast-math (-ffast-math, /fp:fast), which changes floating-point results: this C++ matches Python only without it"
#endif"""

USE_OVERRIDES = [True]  # off while verify builds qtrans's own translation to compare against


class overrides_off:
    """with overrides_off(): translate as if there were no <module>_overrides.hpp."""

    def __enter__(self):
        self.saved = USE_OVERRIDES[0]
        USE_OVERRIDES[0] = False

    def __exit__(self, *exc):
        USE_OVERRIDES[0] = self.saved


class ModuleTranslator:
    def __init__(self, source: str, module_name: str, filename: str = "<input>", deps: dict | None = None,
                 overrides_text: str | None = None):
        """deps: the already-translated modules of the project (name -> ModuleTranslator) that
        this one may import from; see project.py. overrides_text: the overrides file's text
        (default: <module>_overrides.hpp next to the file, if there is one; see overrides.py)."""
        self.source = source
        self.overrides = None
        self.overrides_error = None
        if USE_OVERRIDES[0]:
            try:
                self.overrides = ovr.parse(overrides_text, ovr.path_for(filename if not filename.startswith("<")
                                                                        else f"{module_name}.py")) \
                    if overrides_text is not None else ovr.load(filename)
            except ovr.OverrideError as e:
                self.overrides_error = str(e)
        self.module_name = module_name
        self.filename = filename
        self.tree = loops.rewrite(ast.parse(source, filename))  # enumerate / zip loops -> index loops
        self.full_comments, self.inline_comments = _collect_comments(source)
        self.hints, self.hint_problems = hints_.collect(self.tree, self.full_comments, self.inline_comments)
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
        self.origins = Origins(self)       # which numbers are Python's and which NumPy's (for repr)
        self.printing: set = set()         # functions that print, also through a function they call
        self.variants: dict = {}           # function -> its versions' keys, one per kind of call (overloads)

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
            if bname in ("list", "List"):
                elem = self.parse_annotation(node.slice)
                if elem in (FLOAT, INT, STR):
                    return ListType(elem)
                raise Unsupported(node, f"'{ast.unparse(node)}': lists of float, int or str are supported")
            if bname in ("dict", "Dict"):
                kv = node.slice.elts if isinstance(node.slice, ast.Tuple) else []
                if len(kv) == 2:
                    k, v = self.parse_annotation(kv[0]), self.parse_annotation(kv[1])
                    if k in (STR, INT) and v in (FLOAT, INT, STR):
                        return DictType(k, v)
                raise Unsupported(node, f"'{ast.unparse(node)}': dicts with str or int keys and float, int or str "
                                        "values are supported")
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
        if m.name in cls_.TEXT_METHODS and f.returns is None:
            stub.returns = ast.Name(id="str", ctx=ast.Load())
        sig = self.signature(stub)
        if m.name in cls_.TEXT_METHODS:
            sig.cpp = cls_.TEXT_METHODS[m.name]  # __repr__ -> repr() (a C++ name can't start with __)
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

        # a program (code that runs, not only definitions) becomes int main(): its statements in
        # order, the `if __name__ == "__main__":` block's in its place
        self.is_program = any(_is_main_guard(n) or _runs(n, i) for i, n in enumerate(body))
        fn_reads = set()
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                fn_reads |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        self.main_body, self.main_error = [], None
        for i, node in enumerate(body):
            if i == 0 and is_docstring(node):
                self.doc = inspect.cleandoc(node.value.value)
            elif id(node) in handled:
                continue  # an import from another module of the project (see _import_project)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._collect_import(node)
            elif self.is_program and _is_main_guard(node):
                for st in node.body:
                    if isinstance(st, (ast.Import, ast.ImportFrom)):
                        self._collect_import(st)
                    elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.main_error = self.main_error or Unsupported(
                            st, "a function or class defined inside `if __name__ == \"__main__\":` is not supported",
                            "define it at module level")
                    else:
                        self.main_body.append(st)
                if node.orelse:
                    self.report.module_notes.append(f"line {node.orelse[0].lineno}: the `else:` of "
                                                    "`if __name__ == \"__main__\":` runs only on import; skipped")
            elif self.is_program and isinstance(node, (ast.Assign, ast.AnnAssign)) \
                    and not _stays_module_level(node, fn_reads):
                self.main_body.append(node)  # a variable of the program, not read by any function
            elif self.is_program and _runs(node, i):
                self.main_body.append(node)
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
                self.skipped_globals[node.name] = "it is an async function (coroutines are not supported)"
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
        self._find_printing()
        for f in self.fdefs.values():
            self._strip_errstate(f)
        pending = {}
        for name, f in self.fdefs.items():
            if name in self.rejected or name in self.sigs:  # methods already have theirs
                continue
            try:
                self.sigs[name] = self.signature(f)
            except Unsupported as e:
                if getattr(e, "missing_annotation", False) and "." not in name:
                    pending[name] = e  # types left out: perhaps the file's calls show them
                else:
                    self.rejected[name] = e
        if pending:
            self._infer_from_calls(pending)
        self.analyze_array_params()

    # --- types of unannotated functions, from how the file calls them ----------------------------
    def _main_fdef(self):
        """The program's top-level code as a function (what becomes int main())."""
        first, last = self.main_body[0], self.main_body[-1]
        return ast.FunctionDef(name=PROGRAM, args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                                                                kw_defaults=[], kwarg=None, defaults=[]),
                               body=list(self.main_body), decorator_list=[], returns=ast.Constant(None),
                               type_params=[], lineno=first.lineno, col_offset=0,
                               end_lineno=last.end_lineno, end_col_offset=0)

    def struct_repr_problem(self, info):
        """Why the dataclass repr of `info` can't be written exactly, or None. Python shows each
        field's repr, and a field that holds a NumPy float64 shows as np.float64(...): so every
        value the file puts in a numeric field (constructor arguments, attribute stores, defaults)
        must be a Python number, as qtrans can see from how each was made (origins.py)."""
        cache = self.__dict__.setdefault("_repr_problems", {})
        if info.name in cache:
            return cache[info.name]
        cache[info.name] = (info.node, f"the fields of {info.name} depend on each other")  # while working it out
        from .origins import PY
        numeric = {f.name for f in info.fields if isinstance(f.type, Scalar) or
                   (isinstance(f.type, OptionalType) and isinstance(f.type.inner, Scalar))}
        problem = None
        for f in info.fields:
            if f.repr and f.type == RNG:
                problem = (info.node, f"field '{f.name}' holds a random generator, which Python shows by its address")
            if f.name in numeric and f.default is not None and not isinstance(f.default, ast.Constant):
                problem = problem or (f.default, f"the default of field '{f.name}' isn't a plain number")
        if self.is_program and self.main_body and getattr(self, "_main_fdef_cache", None) is None:
            self._main_fdef_cache = self._main_fdef()
        defs = [(f, self.sigs.get(k)) for k, f in self.fdefs.items()]
        if getattr(self, "_main_fdef_cache", None) is not None:
            defs.append((self._main_fdef_cache, None))
        init = [f.name for f in info.init_fields] if hasattr(info, "init_fields") else [f.name for f in info.fields]
        for fdef, sig in defs:
            if problem:
                break
            fo = self.origins.function(fdef, sig)
            for n in _own_body_nodes(fdef):
                pairs = []
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == info.name:
                    pairs = [(init[i], a) for i, a in enumerate(n.args) if i < len(init)]
                    pairs += [(k.arg, k.value) for k in n.keywords]
                elif isinstance(n, ast.Call) and _ConstTranslator(self).qualify(n.func) == "dataclasses.replace":
                    pairs = [(k.arg, k.value) for k in n.keywords]
                elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and getattr(n, "value", None) is not None:
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    pairs = [(t.attr, n.value) for t in targets if isinstance(t, ast.Attribute)]
                for name, value in pairs:
                    if name in numeric and fo.of(value) != PY:
                        problem = (value, f"field '{name}' of {info.name} gets `{ast.unparse(value)}` on line {value.lineno}, which may be a "
                                          "NumPy number: Python's repr of the object would show it as np.float64(...)")
                        break
                if problem:
                    break
        cache[info.name] = problem
        return problem

    def _provisional(self, name, f, known):
        """A signature with what is known so far (None for a type not known yet)."""
        a = f.args
        nodes = list(a.posonlyargs) + list(a.args)
        defaults = [None] * (len(nodes) - len(a.defaults)) + list(a.defaults)
        params = []
        for arg, d in zip(nodes, defaults):
            t = known.get(arg.arg) if arg.annotation is None else None
            if arg.annotation is not None:
                try:
                    t = self.parse_annotation(arg.annotation)
                except Unsupported:
                    t = None
            params.append(Param(arg.arg, cpp_ident(arg.arg), t, None if d is None else "/* default */"))
        returns_value = any(isinstance(n, ast.Return) and n.value is not None and not (
            isinstance(n.value, ast.Constant) and n.value.value is None) for n in _own_body_nodes(f))
        if f.returns is not None:
            try:
                ret = self.parse_annotation(f.returns, allow_void=True)
            except Unsupported:
                ret = None
        else:
            ret = known.get("return") if returns_value else VOID
        sig = Signature(name, cpp_ident(name), params, ret, f.lineno, None)
        sig.decorator_notes = []
        return sig

    def _infer_from_calls(self, pending):
        """Parameters (and returns) written without a type, typed from the file itself: each
        parameter takes the type every call passes it (and its default's); a return, the type of
        what the body returns. A parameter called with an int in one place and a float in another
        is refused: Python would compute with each as it is (7 // 2 against 7.0 // 2), and a C++
        parameter has one type."""
        from .functions import FunctionTranslator, _Unknown
        known = {name: {} for name in pending}
        conflict = {}
        self.pending_sigs = {}
        def calls_pending(f):
            return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in pending
                       for n in _own_body_nodes(f))
        # only what can tell something: a function calling one of them, or one of them (its returns)
        callers = [(name, f) for name, f in self.fdefs.items() if "." not in name and name not in self.rejected
                   and (name in pending or calls_pending(f))]
        main = self._main_fdef() if self.is_program and self.main_body and self.main_error is None else None
        if main is not None and not calls_pending(main):
            main = None
        param_names = {name: [x.arg for x in list(self.fdefs[name].args.posonlyargs) + list(self.fdefs[name].args.args)]
                       for name in pending}
        for _ in range(2 * len(pending) + 3):
            progress = False
            self.pending_sigs = {n: self._provisional(n, self.fdefs[n], known[n]) for n in pending}
            seen = {n: {} for n in pending}  # parameter -> {type: [lines]}
            calls_seen = {n: [] for n in pending}  # each call: ({parameter: type}, line)
            todo = [(f, self.sigs.get(n) or self.pending_sigs.get(n)) for n, f in callers]
            if main is not None:
                todo.append((main, Signature(PROGRAM, "main", [], VOID, main.lineno, None)))
            for f, sig in todo:
                if sig is None:
                    continue
                tr = FunctionTranslator(self, f, sig)
                try:
                    tr.scan(tr.body)
                    tr.check_loop_vars()
                    tr.finish_scan()
                    tr.infer_types()
                except Exception:  # noqa: BLE001 - what is typed so far is enough to look at the calls
                    pass
                tr.inferring = True
                for call in (n for n in _own_body_nodes(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id in pending and not tr.is_variable(n.func.id)):
                    names = param_names[call.func.id]
                    pairs = [(names[i], a) for i, a in enumerate(call.args) if i < len(names) and not isinstance(a, ast.Starred)]
                    pairs += [(k.arg, k.value) for k in call.keywords if k.arg in names]
                    this = {}
                    for pname, a in pairs:
                        tr.current = call
                        try:
                            t = tr.expr_raw(a).type
                        except Exception:  # noqa: BLE001 - not known yet
                            continue
                        if t is not None and t not in (VOID, NONE):
                            seen[call.func.id].setdefault(pname, {}).setdefault(t, []).append(a.lineno)
                            this[pname] = t
                    calls_seen[call.func.id].append((this, call.lineno))
                # what a pending function returns, once its parameters are known
                name = getattr(f, "name", None)
                if name in pending and "return" not in known[name] and sig.ret is None \
                        and all(p.type is not None for p in sig.params):
                    types = set()
                    for r in (n for n in _own_body_nodes(f) if isinstance(n, ast.Return) and n.value is not None):
                        tr.current = r
                        try:
                            types.add(tr.expr_raw(r.value).type)
                        except Exception:  # noqa: BLE001 - e.g. a recursive call, not typed yet
                            continue
                    # one type for every return (an int on one path and a float on another would
                    # be a Python int or float depending on the path: not one C++ type)
                    if len(types) == 1:
                        t = next(iter(types))
                        if t is not None and t not in (VOID, NONE) and not isinstance(t, OptionalType):
                            known[name]["return"] = t
                            progress = True
            tr0 = _ConstTranslator(self)
            for name in pending:
                f = self.fdefs[name]
                a = f.args
                nodes = list(a.posonlyargs) + list(a.args)
                for arg, d in zip(nodes[len(nodes) - len(a.defaults):], a.defaults):
                    if arg.annotation is None:
                        try:
                            seen[name].setdefault(arg.arg, {}).setdefault(tr0.expr(d).type, []).append(d.lineno)
                        except Unsupported:
                            pass
                for pname, types in seen[name].items():
                    if pname in known[name]:
                        continue
                    if len(types) == 1:
                        (t, _), = types.items()
                        if isinstance(t, (Scalar, ArrayType, StructType)) or t in (STR, RNG):
                            known[name][pname] = t
                            progress = True
                    elif len(types) > 1:
                        conflict[(name, pname)] = types
            if not progress:
                break
        self.pending_sigs = {}
        for name, e in pending.items():
            f = self.fdefs[name]
            missing = [x.arg for x in list(f.args.posonlyargs) + list(f.args.args) if x.annotation is None]
            bad = [(p, conflict[(name, p)]) for p in missing if (name, p) in conflict and p not in known[name]]
            if bad and self._overloads(name, f, missing, known[name], calls_seen[name]):
                continue  # one C++ version per kind of call
            if bad:
                p, types = bad[0]
                where = "; ".join(f"{t} (line {', '.join(map(str, sorted(set(ls))))})" for t, ls in types.items())
                err = Unsupported(f, f"parameter '{p}' of {name}() is given different types by its calls: {where}; "
                                     "a C++ parameter has one type, and Python computes with each as it is",
                                  f"annotate it (`{p}: float`), or pass it the same type everywhere")
                self.rejected[name] = err
                continue
            try:
                sig = self.signature(f, given=known[name])
            except Unsupported as err:
                if getattr(err, "missing_annotation", False):
                    err = Unsupported(f if "return annotation" in err.message else
                                      next((x for x in f.args.args if x.annotation is None), f),
                                      err.message + (", and no call in this file shows its type"
                                                     if "parameter" in err.message else
                                                     ", and its return type can't be worked out from the body"), err.hint)
                self.rejected[name] = err
                continue
            shown = [f"{p}: {known[name][p]}" for p in missing if p in known[name]]
            if f.returns is None and "return" in known[name]:
                shown.append(f"-> {known[name]['return']}")
            sig.decorator_notes.append("types from how the file calls it (no annotations): " + ", ".join(shown))
            self.sigs[name] = sig

    def _infer_return(self, name, f, given):
        """The one type every `return` of f gives, with these parameter types; None if not one."""
        from .functions import FunctionTranslator
        sig = self._provisional(name, f, given)
        sig.ret = None
        tr = FunctionTranslator(self, f, sig)
        try:
            tr.scan(tr.body)
            tr.check_loop_vars()
            tr.finish_scan()
            tr.infer_types()
        except Exception:  # noqa: BLE001 - typed as far as it goes
            pass
        tr.inferring = True
        types = set()
        for r in (n for n in _own_body_nodes(f) if isinstance(n, ast.Return) and n.value is not None):
            tr.current = r
            try:
                types.add(tr.expr_raw(r.value).type)
            except Exception:  # noqa: BLE001 - a recursive call, typed by the others
                continue
        if len(types) != 1:
            return None
        t = next(iter(types))
        return None if t in (None, VOID, NONE) or isinstance(t, OptionalType) else t

    def _overloads(self, name, f, missing, known, calls) -> bool:
        """A function the file calls with different types (half(7) and half(7.0)): one C++
        version for each kind of call, overloads of one name, each computing as Python does
        with those types. True if made; False (and nothing changed) if some call can't be typed."""
        a = f.args
        nodes = list(a.posonlyargs) + list(a.args)
        defaults = dict(zip([x.arg for x in nodes[len(nodes) - len(a.defaults):]], a.defaults))
        tr0 = _ConstTranslator(self)
        kinds, lines = [], {}
        for types, line in calls:
            row = []
            for p in missing:
                t = types.get(p)
                if t is None and p in defaults:
                    try:
                        t = tr0.expr(defaults[p]).type
                    except Unsupported:
                        t = None
                if t is None or not (isinstance(t, (Scalar, ArrayType, StructType)) or t in (STR, RNG)):
                    return False  # a call whose types aren't known: no version can be chosen for it
                row.append(t)
            row = tuple(row)
            if row not in lines:
                kinds.append(row)
            lines.setdefault(row, []).append(line)
        if len(kinds) < 2 or len(kinds) > 8:
            return False
        made = []
        for row in kinds:
            given = dict(zip(missing, row))
            returns_value = any(isinstance(n, ast.Return) and n.value is not None and not (
                isinstance(n.value, ast.Constant) and n.value.value is None) for n in _own_body_nodes(f))
            if f.returns is None and returns_value:
                ret = self._infer_return(name, f, given)
                if ret is None:
                    return False
                given["return"] = ret
            try:
                sig = self.signature(f, given=given)
            except Unsupported:
                return False
            key = f"{name}({','.join(str(t) for t in row)})"  # no spaces: also the verifier's name for it
            where = ", ".join(map(str, sorted(set(lines[row]))))
            sig.overloaded = True
            sig.decorator_notes.append(f"one C++ version for each kind of call (overloads of {name}): this one for "
                                       + ", ".join(f"{p}: {t}" for p, t in zip(missing, row))
                                       + (f" -> {given['return']}" if "return" in given else "")
                                       + f", called so on line {where}")
            made.append((key, sig))
        # in the file's order: each version where the function was
        order = list(self.fdefs.items())
        self.fdefs.clear()
        for k, node in order:
            if k == name:
                for key, _ in made:
                    self.fdefs[key] = node
            else:
                self.fdefs[k] = node
        for key, sig in made:
            self.sigs[key] = sig
        self.variants[name] = [key for key, _ in made]
        return True

    def _collect_import(self, node):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in self.deps:
                    continue
                if a.asname:
                    self.imports[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    self.imports[top] = top
            return
        if node.module == "__future__":
            return
        if node.level:
            self.report.module_notes.append(f"line {node.lineno}: relative import ignored")
            return
        for a in node.names:
            if a.name == "*":
                self.report.module_notes.append(f"line {node.lineno}: `from {node.module} import *` ignored")
                continue
            self.imports[a.asname or a.name] = f"{node.module}.{a.name}"

    def _find_printing(self):
        """The functions that print, directly or through a function they call: calls to them
        have an effect (output), so their order matters as a random draw's does."""
        calls = {}
        for name, f in self.fdefs.items():
            calls[name] = {n.func.id for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            if "print" in calls[name]:
                self.printing.add(name)
        changed = True
        while changed:
            changed = False
            for name, called in calls.items():
                if name not in self.printing and called & self.printing:
                    self.printing.add(name)
                    changed = True

    def fdef_named(self, name):
        """The def of a module function by its Python name (its versions share one)."""
        if name in self.fdefs:
            return self.fdefs[name]
        keys = self.variants.get(name)
        return self.fdefs[keys[0]] if keys else None

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
            if (isinstance(e.type, ListType) and e.type.elem is not None) or \
                    (isinstance(e.type, DictType) and e.type.key is not None):
                e = Expr(e.code[len(e.type.cpp()):] if e.code.startswith(e.type.cpp() + "{") else e.code,
                         P_ATOM, e.type)  # inline const std::vector<double> RATES = {0.01, 0.02};
            elif not isinstance(e.type, Scalar) and e.type != STR:
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

    def signature(self, f, given=None):
        """given: types for parameters (and "return") written without an annotation, worked out
        from how the file calls the function (see _infer_from_calls)."""
        given = given or {}
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
            if arg.annotation is None and arg.arg not in given:
                e = Unsupported(arg, f"parameter '{arg.arg}' has no type annotation",
                                f"write `{arg.arg}: float` (or int / bool) for a number, or "
                                f"`{arg.arg}: Vec` / `{arg.arg}: Mat` for a 1-D / 2-D float array "
                                f"(from qtrans.typing import Vec, Mat)")
                e.missing_annotation = True
                raise e
            t = given[arg.arg] if arg.annotation is None else self.parse_annotation(arg.annotation)
            if not isinstance(t, (Scalar, ArrayType, RngType, OptionalType, StructType, ListType, DictType)) and t != STR:
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
            returns_value = any(isinstance(n, ast.Return) and n.value is not None and not (
                isinstance(n.value, ast.Constant) and n.value.value is None) for n in _own_body_nodes(f))
            if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _own_body_nodes(f)):
                raise Unsupported(f, "generators (yield) are not supported")
            if returns_value and "return" not in given:
                e = Unsupported(f, f"function '{f.name}' has no return annotation",
                                "add -> float (or int, bool, tuple[float, float], None)")
                e.missing_annotation = True
                raise e
            ret = given["return"] if returns_value else VOID  # no value returned: it returns None
        else:
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
        self.maybe_int = set()  # (function, element or None): a float return that is a Python int on some path
        for _ in range(len(self.fdefs) + 1):
            known = set(self.maybe_int)
            bodies, notes, general = self._translate_functions()
            if self.maybe_int == known:
                break  # what each function printed was judged knowing every int-returning one
        self.main_lines = None
        if self.is_program:
            self._translate_main()
        return self._finish(bodies, notes, general)

    def _translate_functions(self):
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
                ov = self._override(name)
                if ov is not None:
                    why = ovr.head_mismatch(ov.head, "inline " + self.cpp_signature(self.sigs[name], defaults=False))
                    if why is not None:
                        self.rejected[name] = Unsupported(
                            f, f"your C++ for {name}() ({self.overrides.path.name}, line {ov.line}) doesn't fit the "
                               f"Python's signature: {why}",
                            "its head must be: inline " + self.cpp_signature(self.sigs[name], defaults=False))
                        progress = True
                        continue
                    try:
                        tr.translate()  # what the Python does (it changes an argument, prints, ...) still counts
                        self.array_users[name] = tr.used_arrays
                        qtrans_could = None
                    except Unsupported as e:
                        qtrans_could = e
                        self.array_users[name] = False
                    bodies[name] = list(ov.body)
                    self.line_notes[name] = []
                    notes[name] = [self._override_note(ov, qtrans_could)]
                    continue
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
                    bad = next((m for m in info.methods.values() if m.key in self.rejected
                                and m.name not in cls_.TEXT_METHODS), None)  # a refused __repr__: refused where printed
                    if bad is not None:
                        e = self.rejected[bad.key]
                        info.rejected = Unsupported(bad.fdef, f"method {bad.name} could not be translated: "
                                                              f"{e.message}", e.hint)
                        progress = True
            if not progress:
                break
            bodies, notes, general = {}, {}, {}  # retranslate: a call may now point at a rejected function
        return bodies, notes, general

    def _override(self, name):
        """The user's own C++ for this function, if the overrides file has a block for it."""
        if self.overrides is None or name not in self.overrides.blocks:
            return None
        return self.overrides.blocks[name]

    def fingerprint(self, name) -> str:
        dep_sources = {m: t.source for m, t in self.deps.items()}
        imported = {local: (owner, there) for local, (owner, there, _kind) in self.imported.items()}
        return ovr.fingerprint(self.source, name, imported, dep_sources)

    def _override_note(self, ov, qtrans_could) -> str:
        ov.current = self.fingerprint(ov.name)
        ov.status = "up to date" if ov.fingerprint == ov.current else \
            "not yet accepted" if ov.fingerprint is None else "out of date"
        ov.qtrans_could = None if qtrans_could is None else qtrans_could.message
        where = f"{self.overrides.path.name}, line {ov.line}"
        if ov.status == "up to date":
            text = f"your own C++ ({where}), checked against this Python"
        elif ov.fingerprint is None:
            text = (f"your own C++ ({where}) has no python= fingerprint: run `qtrans verify`, then "
                    f"`qtrans override --accept {Path(self.filename).name} {ov.name}`")
        else:
            text = (f"OUT OF DATE: your own C++ ({where}) was written for an earlier version of this Python "
                    f"(or of what it calls); it is still used, and `qtrans verify` checks it: once it passes, "
                    f"`qtrans override --accept {Path(self.filename).name} {ov.name}`")
        if qtrans_could is not None:
            text += f" -- qtrans can't translate this function itself ({qtrans_could.message})"
        return text

    def _check_overrides(self):
        """Blocks that name nothing overridable are reported, not silently ignored."""
        if self.overrides_error:
            self.report.module_notes.append(f"the overrides file can't be used: {self.overrides_error}")
        if self.overrides is None:
            return
        for name, ov in self.overrides.blocks.items():
            variants = getattr(self, "variants", {})
            if name in variants:
                self.report.module_notes.append(
                    f"{self.overrides.path.name}, line {ov.line}: {name}() has one C++ version per kind of call "
                    "here; annotate its parameters so it has one, then the override is used")
            elif name not in self.fdefs or (name in self.sigs and self.sigs[name].owner):
                self.report.module_notes.append(
                    f"{self.overrides.path.name}, line {ov.line}: a block for {name}(), which is not a "
                    "function of this module: not used")

    def source_hash(self) -> str:
        """The first 16 hex digits of the Python source's SHA-256: which Python this C++ is from."""
        import hashlib
        return hashlib.sha256(self.source.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]

    def _hinted(self, head, f) -> str:
        return hints_.def_head(head, self.hints.get(f.lineno, {}))

    def _finish(self, bodies, notes, general):
        self._check_overrides()
        self.report.module_notes += self.hint_problems
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

        if self.is_program:
            first = self.main_body[0].lineno if self.main_body else 1
            e = self.main_error
            self.report.functions.append(FunctionReport(PROGRAM, first, e is None, e.message if e else None,
                                                        e.lineno if e else None, e.hint if e else None,
                                                        notes=getattr(self, "main_notes", [])))
        for text, names in general.items():
            self.report.general_notes.append(f"{text} (applies to: {', '.join(names)})")
        header = self.render(bodies)
        sigs = {n: self.sigs[n] for n in bodies}
        self.translated = set(bodies)
        self._build_namespace()
        uses_random = "qtrans_random.hpp" in header
        live = {n: self.classes[n] for n in getattr(self, "class_order", []) if self.classes[n].rejected is None}
        program = getattr(self, "program_text", None)
        pmap = getattr(self, "program_map", None)
        if program is not None and program.endswith("\n"):
            pmap = pmap[:-1]
        notes = {n: v for n, v in self.line_notes.items() if n in bodies or (n == PROGRAM and program)}
        uses_arrays = "qtrans_arrays.hpp" in header or uses_random  # the header's needs (the program's: its text)
        result = TranslationResult(self.module_name, header, self.report, sigs, "qt::" in header + (program or ""),
                                 uses_arrays, uses_random, live, self.ns,
                                 list(self.project_deps), self.line_map[:-1] if header.endswith("\n") else self.line_map,
                                 notes, program, pmap,
                                 self._main_spans(),
                                 bool(sigs or live or any(not c.external for c in self.constants.values())),
                                 frozenset(n for n in bodies if n.split("(")[0] in self.printing))
        # the functions that are the user's own C++: name -> overrides.Override (status, line, ...)
        result.overrides = {n: self.overrides.blocks[n] for n in bodies if self._override(n) is not None}
        result.overrides_file = self.overrides.path if self.overrides is not None else None
        result.bodies = dict(bodies)  # name -> its C++ body lines (for `qtrans override`)
        result.heads = {n: "inline " + self.cpp_signature(self.sigs[n], defaults=False)
                        for n in self.fdefs if n in self.sigs and not self.sigs[n].owner}
        result.rejections = {n: e.message for n, e in self.rejected.items()}
        result.fingerprint_of = self.fingerprint
        for fr in self.report.functions:
            if fr.name in result.overrides:
                fr.override = result.overrides[fr.name].status
        return result

    def _main_spans(self) -> tuple:
        """The Python lines of the program: the `if __name__ == "__main__":` line, then each statement."""
        if not self.is_program:
            return ()
        guard = next((n for n in self.tree.body if _is_main_guard(n)), None)
        head = ((guard.lineno, guard.test.end_lineno),) if guard is not None else ()
        lines = self.source.splitlines()
        ends = sorted({n.end_lineno for n in self.tree.body} | ({guard.test.end_lineno} if guard is not None else set()))
        spans = []
        for st in self.main_body:
            floor = max([e for e in ends if e < st.lineno] + [0])
            a = st.lineno
            while a - 1 > floor and (not lines[a - 2].strip() or lines[a - 2].lstrip().startswith("#")):
                a -= 1  # the comments (and blank lines) just above it are the program's too
            while a < st.lineno and not lines[a - 1].strip():
                a += 1
            spans.append((a, st.end_lineno))
        return head + tuple(spans)

    def _translate_main(self):
        """The program's statements as main()'s body (self.main_lines), or why not (self.main_error)."""
        if self.main_error is not None:
            return
        if not self.main_body:
            self.main_error = Unsupported(None, "the `if __name__ == \"__main__\":` block has nothing to run")
            return
        later = {}
        for name, f in self.fdefs.items():
            later[name.split("(")[0]] = f.lineno
        for name, info in self.classes.items():
            if not info.external:
                later[name] = info.node.lineno
        for c_name, c in self.constants.items():
            if not c.external and c.lineno:
                later[c_name] = c.lineno
        for st in self.main_body:
            for n in ast.walk(st):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and later.get(n.id, 0) > st.lineno \
                        and not any(isinstance(x, ast.Name) and x.id == n.id and isinstance(x.ctx, ast.Store)
                                    for b in self.main_body if b.lineno < st.lineno for x in ast.walk(b)):
                    self.main_error = Unsupported(
                        n, f"`{n.id}` is used here but defined further down, on line {later[n.id]}: when the file "
                           "runs, Python raises NameError", "move the definition above this line")
                    return
        first, last = self.main_body[0], self.main_body[-1]
        fdef = ast.FunctionDef(name=PROGRAM, args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                                                                kw_defaults=[], kwarg=None, defaults=[]),
                               body=list(self.main_body), decorator_list=[], returns=ast.Constant(None),
                               type_params=[], lineno=first.lineno, col_offset=0,
                               end_lineno=last.end_lineno, end_col_offset=0)
        # comments: from the line after whatever precedes main()'s first statement
        prev = [n.end_lineno for n in self.tree.body if n.end_lineno < first.lineno]
        guard = next((n for n in self.tree.body if _is_main_guard(n)), None)
        fdef.comment_floor = max(prev) if prev else 0
        if guard is not None and guard.lineno < first.lineno:
            fdef.comment_floor = max(fdef.comment_floor, guard.lineno)
        skip = set()
        mine = {id(st) for st in self.main_body}
        for n in self.tree.body:
            if id(n) in mine or _is_main_guard(n):
                continue
            a = n.lineno
            if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                a = min([a] + [d.lineno for d in n.decorator_list]) - len(self.attached_comments(min([a] + [d.lineno for d in n.decorator_list])))
            skip.update(range(a, n.end_lineno + 1))
        self._strip_errstate(fdef)
        sig = Signature(PROGRAM, "main", [], VOID, first.lineno, None)
        self.sigs[PROGRAM] = sig
        tr = _MainTranslator(self, fdef, sig, skip)
        try:
            self.main_lines = tr.translate()
            self.array_users[PROGRAM] = tr.used_arrays
            self.line_notes[PROGRAM] = [(src, key, re.sub(r"^line \d+: ", "", text))
                                        for src, key, text in getattr(tr, "line_notes", [])]
            self.main_notes = [text for key, text in tr.notes if key not in GENERAL_NOTES]
            self.main_may_raise = getattr(tr, "text_may_raise", False)  # a format that can raise
            self.main_guard = guard
        except Unsupported as e:
            self.main_error = e
        finally:
            del self.sigs[PROGRAM]

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
            if isinstance(p.type, (ListType, DictType)):
                ty = f"const {p.type.cpp()}&"  # a function never changes a list passed to it (check_lists)
            elif is_array(p.type) or p.type in (RNG, STR) or _optional_array(p.type) or isinstance(p.type, StructType):
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
        if info.name in getattr(self, "repr_needed", set()) and "__repr__" not in info.methods:
            out.append("    std::string repr() const;  // repr(), as Python writes it for this class")
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
        if info.name in getattr(self, "repr_needed", set()) and "__repr__" not in info.methods:
            mark = source_mark(cls, cls)
            pieces, text = [], f"{info.name}("
            for k, f in enumerate(x for x in info.fields if x.repr):
                text += ("" if k == 0 else ", ") + f"{f.name}="
                pieces += [cpp_string_literal(text),
                           f"{f.cpp}.repr()" if isinstance(f.type, StructType) else f"qt::py::repr({f.cpp})"]
                text = ""
            pieces.append(cpp_string_literal(text + ")"))
            code = " + ".join(pieces)
            out.append(f"inline std::string {info.cpp}::repr() const {{" + mark)
            out.append(f"    return {code};" + mark)
            out.append("}" + mark)
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
        out = [f"// {self.module_name}.hpp: generated by qtrans {VERSION} from {Path(self.filename).name}"
               f" (sha256 {self.source_hash()}).",
               "// Edit the Python source and re-run qtrans rather than editing this file."]
        if self.doc:
            out.append("//")
            out += [("// " + line).rstrip() for line in self.doc.splitlines()]
        out += ["#pragma once", "", FAST_MATH_GUARD, "", "@@INCLUDES@@", ""]
        overridden = [n for n in self.fdefs if n in bodies and self._override(n) is not None]
        if overridden and self.overrides.preamble:
            out += [f"// from {self.overrides.path.name}"] + list(self.overrides.preamble) + [""]
        out += [f"namespace {ns} {{", ""]

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
        forward = len(names) > 1 or (live and names) or bool(overridden)  # overrides: defaults go here
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
            ov = self._override(n)
            if ov is not None:  # the user's own C++, in place of the translation
                mark = source_mark(first, f.end_lineno)
                out.append(f"// your C++ ({self.overrides.path.name}): {ov.status or 'used'}; "
                           "`qtrans verify` checks it against the Python" + mark)
                out += [c.rstrip() + mark for c in ov.comments if c.strip()]
                head = ovr.without_defaults(ov.head)
                if not re.match(r"^\s*(\[\[.*?\]\]\s*)*(inline|constexpr|static)\b", head):
                    head = "inline " + head  # in a header, a function must be inline
                out += [l + mark for l in (head + " {").split("\n")]
                out += [l + mark for l in ov.body]
                out.append("}" + mark)
                out.append("")
                continue
            used = _used_params(f, bodies[n], sig)
            out.append(self._hinted("inline " + self.cpp_signature(sig, defaults=not forward, used_params=used), f)
                       + " {" + source_mark(first, head_end))
            out += bodies[n]
            out.append("}" + source_mark(f.end_lineno, f.end_lineno))
            out.append("")

        out.append(f"}}  // namespace {ns}")
        text = "\n".join(out) + "\n"
        if self.main_lines is not None:
            empty = not own_constants and not live and not names
            self.program_text, self.program_map = strip_marks(self._program(out, empty, bodies))
        incs = self._includes(text, bodies)
        if re.search(r"\bQT_(UNROLL|FORCE_INLINE|NOINLINE|HOT|COLD|LIKELY|UNLIKELY)\b", text):
            incs += [""] + hints_.MACROS.split("\n")
        text, self.line_map = strip_marks(text.replace("@@INCLUDES@@", "\n".join(incs)))
        return text

    def _includes(self, text, bodies) -> list:
        includes = ["<cmath>"]
        checks = [("<algorithm>", r"std::(max|min|reverse)\("), ("<vector>", r"std::vector<"), ("<cstdlib>", r"std::(abs|exit)\("),
                  ("<stdexcept>", r"throw std::"), ("<tuple>", r"std::(tuple|tie|ignore|make_tuple)"),
                  ("<initializer_list>", r"(?m)std::(max|min)\(\{|^\s*for \([^:;]+ : \{"), ("<string>", r"std::(string|to_string)\b"),
                  ("<optional>", r"std::optional"), ("<utility>", r"std::move\("),
                  ("<iostream>", r"std::(cout|cerr)\b"), ("<exception>", r"std::exception\b")]
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)  # not what string literals say
        for inc, pat in checks:
            if re.search(pat, code):
                includes.append(inc)
        if "int main()" in code and not re.search(r"std::(sqrt|exp|log|pow|fabs|floor|ceil|sin|cos|nearbyint|isnan|isinf|isfinite)|qt::", code):
            includes.remove("<cmath>")  # nothing mathematical (a program that only prints)
        inc_lines = [f"#include {i}" for i in includes]
        if "qt::Rng" in code:
            inc_lines.append('#include "qtrans_random.hpp"  // NumPy\'s generator, plus Eigen arrays')
        elif re.search(r"qt::(Vec|Mat|Mask2?)\b", code) or "Eigen::" in code \
                or any(self.array_users.get(n) for n in bodies) or (self.array_users.get(PROGRAM) and "int main()" in code):
            inc_lines.append('#include "qtrans_arrays.hpp"  // Eigen, plus NumPy semantics')
        elif "qt::" in code.replace("qt::py::", ""):
            inc_lines.append('#include "qtrans_runtime.hpp"')
        if "std::vector<" in code or re.search(r"qt::py::(item|slice|sum|sum_left|min|max|sorted|sort|pop|insert|index|"
                                                r"count|contains|concat|range_list|repeat|len|none|join|dict|KeyError)\b", code):
            inc_lines.append('#include "qtrans_list.hpp"  // lists, dicts, and print(), str(), repr(), format() as Python writes them')
        elif "qt::py::" in code:
            inc_lines.append('#include "qtrans_print.hpp"  // print(), str(), repr(), format() as Python writes them')
        for d in self.project_deps:  # the modules of the project it imports from
            inc_lines.append(f'#include "{d}.hpp"  // from {d} import ...')
        return inc_lines

    def _program(self, header_lines, empty_ns, bodies) -> str:
        """<module>.cpp: the module's C++ (no #pragma once; no namespace when there is nothing in
        it) followed by int main(), the program's top-level code."""
        out = [f"// {self.module_name}.cpp: generated by qtrans {VERSION} from {Path(self.filename).name}"
               f" (sha256 {self.source_hash()}).",
               "// Edit the Python source and re-run qtrans rather than editing this file."]
        rest = header_lines[2:]
        rest = [l for l in rest if l != "#pragma once"]
        if empty_ns:
            k = rest.index(f"namespace {self.ns} {{")
            rest = rest[:k] + [l for l in rest[k + 1:] if l != f"}}  // namespace {self.ns}" and l.strip()]
        else:
            rest.append("")
        body = list(self.main_lines)
        if not empty_ns:  # the module's own functions, classes and constants, from outside its namespace
            body = [self.qualify_code(l) for l in body]
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', "\n".join(body))
        calls = {self.sigs[n].cpp for n in self.fdefs if n in self.sigs and n in bodies}
        risky = getattr(self, "main_may_raise", False) or \
            re.search(r"\bthrow\b|qt::(?!py::(str|repr|np|format|round_numpy)\b|inf\b|nan\b|pi\b|e\b|tau\b)", code) or \
            any(re.search(rf"\b{re.escape(c)}\(", code) for c in calls)
        guard = getattr(self, "main_guard", None)
        head = "int main() {" + (source_mark(guard.lineno, guard.test.end_lineno) if guard is not None else "")
        if risky:
            # an exception Python wouldn't catch ends the program with its message and exit status 1
            out_main = [head, "    try {"] + ["    " + l if l.strip() else l for l in body] + [
                "    } catch (const std::exception& e) {",
                "        std::cerr << e.what() << '\\n';",
                "        return 1;",
                "    }",
                "}"]
        else:
            out_main = [head] + body + ["}"]
        text = "\n".join(out + rest + out_main) + "\n"
        return text.replace("@@INCLUDES@@", "\n".join(self._includes(text, bodies)))


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
        if result.library or result.program is None:
            (out / f"{path.stem}.hpp").write_text(result.header, encoding="utf-8")
        if result.program is not None:
            (out / f"{path.stem}.cpp").write_text(result.program, encoding="utf-8")
        (out / f"{path.stem}_report.md").write_text(result.report.to_markdown(), encoding="utf-8")
        if result.uses_runtime:
            (out / "qtrans_runtime.hpp").write_text(RUNTIME_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
        everything = result.header + (result.program or "")
        if result.uses_arrays or "qtrans_arrays.hpp" in everything or "qtrans_random.hpp" in everything:
            (out / "qtrans_arrays.hpp").write_text(ARRAYS_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
            (out / "qtrans_runtime.hpp").write_text(RUNTIME_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
        if result.uses_random or "qtrans_random.hpp" in everything:
            (out / "qtrans_random.hpp").write_text(RANDOM_HEADER.read_text(encoding="utf-8"), encoding="utf-8")
        if re.search(r"qtrans_(print|list)\.hpp", result.header + (result.program or "")):
            for h in (RUNTIME_HEADER, PRINT_HEADER, LIST_HEADER):
                (out / h.name).write_text(h.read_text(encoding="utf-8"), encoding="utf-8")
    return result
