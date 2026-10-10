"""Stage 5: @dataclass and NamedTuple classes become C++ structs.

Collection happens in two steps, because a field can have the type of another class:
first every class name is registered (so annotations can refer to it), then fields,
defaults and methods are read.

The C++ shape of a class:

    struct Option {
        double S{};  ...                       // the fields, in order
        Option() = default;                    // for declarations only (see below)
        Option(double S_, ..., double r_ = 0.0);   // the dataclass __init__, then __post_init__
        double price() const;                  // a method that doesn't modify self is const
        double forward() const;                // @property: C++ needs the ()
        static Option from_vol(...);           // @classmethod / @staticmethod
        bool operator==(const Option&) const;  // dataclass equality, field by field
    };

Python objects are shared, not copied, when assigned or passed; a C++ struct is copied.
The two agree as long as nothing modifies a shared object, so the same analysis that guards
arrays applies (translator.analyze_array_params, functions.check_aliasing): an object a
function modifies is passed by reference, and two names for one object, one of them later
modified, are refused. Frozen classes can't be modified at all, so they are always safe.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .core import cpp_ident
from .errors import Unsupported
from .types import ArrayType, OptionalType, Scalar, StructType, is_array

DUNDERS_SKIPPED = {"__repr__", "__str__", "__format__"}
DUNDERS_REFUSED = {"__eq__", "__ne__", "__hash__", "__lt__", "__le__", "__gt__", "__ge__", "__init__",
                   "__new__", "__setattr__", "__getattr__", "__getattribute__", "__bool__", "__len__",
                   "__iter__", "__getitem__", "__call__", "__add__", "__sub__", "__mul__", "__truediv__"}


@dataclass
class FieldInfo:
    name: str
    cpp: str
    type: object = None
    annotation: object = None
    default: object = None          # AST of the default value (field(default=...) or `= value`)
    factory: object = None          # AST of the default_factory's result expression
    init: bool = True               # a parameter of the constructor
    compare: bool = True            # part of ==
    default_code: str | None = None  # C++ for the default, once translated


@dataclass
class MethodInfo:
    name: str
    fdef: ast.FunctionDef
    kind: str                       # "instance", "property", "static", "class", "post_init"
    key: str = ""                   # "Class.name": its key in the module's signature table


@dataclass
class ClassInfo:
    name: str
    node: ast.ClassDef
    frozen: bool = False
    eq: bool = True
    namedtuple: bool = False
    plain: bool = False             # a plain class: fields come from __init__, which is the constructor
    fields: list = field(default_factory=list)
    methods: dict = field(default_factory=dict)
    doc: str | None = None
    rejected: Unsupported | None = None
    notes: list = field(default_factory=list)
    ns: str | None = None           # the C++ namespace of the module that defines it, when imported
    external: bool = False          # imported from another module of the project: not rendered here

    @property
    def cpp(self) -> str:
        return f"{self.ns}::{cpp_ident(self.name)}" if self.ns else cpp_ident(self.name)

    @property
    def type(self) -> StructType:
        return StructType(self.name, self.cpp)

    def field(self, name):
        return next((f for f in self.fields if f.name == name), None)

    @property
    def init_fields(self):
        return [] if self.plain else [f for f in self.fields if f.init]

    def has_array_field(self, classes, seen=None) -> bool:
        """Is an array somewhere inside (so Python's == would raise on it)?"""
        seen = set() if seen is None else seen
        seen.add(self.name)
        for f in self.fields:
            t = f.type.inner if isinstance(f.type, OptionalType) else f.type
            if is_array(t):
                return True
            if isinstance(t, StructType) and t.name not in seen and classes[t.name].has_array_field(classes, seen):
                return True
        return False


def member_safe_name(name: str, info: "ClassInfo", py_names) -> str:
    """A local or parameter name that hides no member of the class (C++ would shadow it)."""
    taken = {f.cpp for f in info.fields} | {cpp_ident(m) for m in info.methods}
    base = cpp_ident(name)
    if base not in taken:
        return base
    new = base + "_"
    while new in taken or new in py_names:
        new += "_"
    return new


def _const_bool(node, what):
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    raise Unsupported(node, f"{what} must be the constant True or False")


def dataclass_options(mod, node: ast.ClassDef):
    """(is_dataclass, frozen, eq, namedtuple) from the decorators and bases."""
    frozen, eq, is_dc = False, True, False
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        q = mod._qualify_path(target) if isinstance(target, ast.Attribute) else mod.imports.get(getattr(target, "id", None))
        if q != "dataclasses.dataclass":
            raise Unsupported(dec, f"decorator @{ast.unparse(dec)} on a class is not supported")
        is_dc = True
        if isinstance(dec, ast.Call):
            if dec.args:
                raise Unsupported(dec, "@dataclass takes keyword arguments only")
            for k in dec.keywords:
                if k.arg == "frozen":
                    frozen = _const_bool(k.value, "frozen=")
                elif k.arg == "eq":
                    eq = _const_bool(k.value, "eq=")
                elif k.arg in ("repr", "slots", "match_args"):
                    continue  # no effect on the numbers
                elif k.arg == "init" and _const_bool(k.value, "init=") is False:
                    raise Unsupported(dec, "@dataclass(init=False) is not supported", "let the dataclass write __init__")
                else:
                    raise Unsupported(dec, f"@dataclass({k.arg}=...) is not supported")
    namedtuple = False
    for b in node.bases:
        q = mod._qualify_path(b) if isinstance(b, ast.Attribute) else mod.imports.get(getattr(b, "id", None))
        if isinstance(b, ast.Name) and b.id == "object":
            continue
        if q == "typing.NamedTuple":
            namedtuple = True
        else:
            raise Unsupported(node, f"class {node.name} inherits from {ast.unparse(b)}: inheritance is not supported",
                              "use composition: a field holding the other object")
    if node.keywords:
        raise Unsupported(node, "class keywords (metaclass=...) are not supported")
    if namedtuple and is_dc:
        raise Unsupported(node, "a NamedTuple can't also be a @dataclass")
    return is_dc, frozen or namedtuple, eq, namedtuple


def collect_class_body(mod, info: ClassInfo):
    """Fields (annotated names, in order), their defaults, and the methods."""
    node = info.node
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        info.doc = body[0].value.value
        body = body[1:]
    for s in body:
        if isinstance(s, ast.Pass):
            continue
        if info.plain and isinstance(s, ast.AnnAssign) and s.value is not None:
            raise Unsupported(s, f"class attribute '{ast.unparse(s.target)}' with a value is shared by every "
                                 "object in Python", "set it in __init__ (self.x = ...), or use @dataclass")
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue  # a stray string or `...`
        if isinstance(s, ast.AnnAssign):
            if not isinstance(s.target, ast.Name):
                raise Unsupported(s, "class fields must be plain names")
            if "ClassVar" in ast.unparse(s.annotation):
                raise Unsupported(s, "ClassVar class attributes are not supported", "use a module-level constant")
            f = FieldInfo(s.target.id, cpp_ident(s.target.id), annotation=s.annotation)
            if s.value is not None:
                _field_default(mod, info, f, s.value)
            if any(g.name == f.name for g in info.fields):
                raise Unsupported(s, f"field '{f.name}' is declared twice")
            info.fields.append(f)
            continue
        if isinstance(s, ast.FunctionDef):
            info.methods[s.name] = _method(mod, info, s)
            continue
        if isinstance(s, ast.Assign):
            raise Unsupported(s, "class attributes without a type annotation are not fields of a dataclass",
                              "annotate it (x: float = 1.0) or make it a module-level constant")
        raise Unsupported(s, f"{type(s).__name__} statements in a class body are not supported")
    if info.plain:
        _fields_from_init(info)
    seen_default = False
    for f in info.init_fields:
        has = f.default is not None or f.factory is not None
        if seen_default and not has:  # Python raises TypeError at class creation too
            raise Unsupported(node, f"field '{f.name}' without a default follows a field with one")
        seen_default = seen_default or has
    names = {f.name for f in info.fields}
    for m in info.methods:
        if m in names:
            raise Unsupported(info.methods[m].fdef, f"method '{m}' has the same name as a field")


def _fields_from_init(info):
    """A plain class's fields: names annotated in the class body (`x: float`), then every
    `self.x = ...` in __init__, in order of first assignment."""
    init = info.methods.get("__init__")
    if init is None:
        raise Unsupported(info.node, f"class {info.name} has no __init__", "add one, or use @dataclass")
    for n in ast.walk(init.fdef):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            targets = [n.target]
        for t in targets:
            for leaf in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(leaf, ast.Attribute) and isinstance(leaf.value, ast.Name) and leaf.value.id == "self":
                    if info.field(leaf.attr) is None:
                        ann = n.annotation if isinstance(n, ast.AnnAssign) else None
                        info.fields.append(FieldInfo(leaf.attr, cpp_ident(leaf.attr), annotation=ann))
    for m in info.methods.values():
        if m.kind in ("instance", "property"):
            for n in ast.walk(m.fdef):
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self" \
                                and info.field(t.attr) is None:
                            raise Unsupported(n, f"{m.name} creates attribute '{t.attr}', which __init__ doesn't set",
                                              "set every attribute in __init__ (to a starting value if need be)")


def _field_default(mod, info, f: FieldInfo, value):
    if isinstance(value, ast.Call) and _is_field_call(mod, value):
        if info.namedtuple:
            raise Unsupported(value, "field(...) is for dataclasses, not NamedTuple")
        if value.args:
            raise Unsupported(value, "field(...) takes keyword arguments only")
        for k in value.keywords:
            if k.arg == "default":
                f.default = k.value
            elif k.arg == "default_factory":
                v = k.value
                if isinstance(v, ast.Lambda) and not v.args.args:
                    f.factory = v.body
                else:
                    raise Unsupported(value, "default_factory must be a lambda with no arguments",
                                      "e.g. field(default_factory=lambda: np.zeros(3))")
            elif k.arg == "init":
                f.init = _const_bool(k.value, "init=")
            elif k.arg == "compare":
                f.compare = _const_bool(k.value, "compare=")
            elif k.arg in ("repr", "hash", "metadata", "kw_only"):
                if k.arg == "kw_only":
                    raise Unsupported(value, "kw_only fields are not supported")
            else:
                raise Unsupported(value, f"field({k.arg}=...) is not supported")
        return
    f.default = value


def _is_field_call(mod, call) -> bool:
    fn = call.func
    q = mod._qualify_path(fn) if isinstance(fn, ast.Attribute) else mod.imports.get(getattr(fn, "id", None))
    return q == "dataclasses.field"


def _method(mod, info, s: ast.FunctionDef) -> MethodInfo:
    kind = "instance"
    for dec in s.decorator_list:
        d = ast.unparse(dec)
        if d == "property":
            kind = "property"
        elif d == "staticmethod":
            kind = "static"
        elif d == "classmethod":
            kind = "class"
        elif d.endswith(".setter") or d.endswith(".deleter"):
            raise Unsupported(dec, "property setters are not supported", "use a method or dataclasses.replace")
        else:
            raise Unsupported(dec, f"decorator @{d} on a method is not supported")
    if len(s.decorator_list) > 1:
        raise Unsupported(s, "a method with more than one decorator is not supported")
    if s.name == "__post_init__":
        if info.namedtuple or info.plain:
            raise Unsupported(s, "__post_init__ is only called by @dataclass")
        kind = "post_init"
    elif s.name == "__init__" and info.plain:
        if s.decorator_list:
            raise Unsupported(s, "__init__ can't have decorators")
        kind = "init"
    elif s.name in DUNDERS_REFUSED:
        raise Unsupported(s, f"defining {s.name} is not supported: it changes what Python operators mean",
                          "use a plainly named method")
    elif s.name.startswith("__") and s.name.endswith("__") and s.name not in DUNDERS_SKIPPED:
        raise Unsupported(s, f"defining {s.name} is not supported")
    args = s.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise Unsupported(s, "*args, **kwargs and keyword-only parameters are not supported")
    if kind in ("instance", "property", "post_init", "class", "init"):
        want = "cls" if kind == "class" else "self"
        if not args.args or args.args[0].arg != want:
            raise Unsupported(s, f"the first parameter of {s.name} must be '{want}'")
    if kind in ("property", "post_init") and len(args.args) != 1:
        raise Unsupported(s, f"{s.name} takes only self")
    return MethodInfo(s.name, s, kind, f"{info.name}.{s.name}")


def check_field_types(mod, info: ClassInfo):
    for f in info.fields:
        if f.annotation is None:  # a plain class's field typed by __init__ (translator.infer_init_fields)
            continue
        t = mod.parse_annotation(f.annotation)
        if not (isinstance(t, (Scalar, ArrayType, OptionalType, StructType)) or str(t) in ("str", "Generator")):
            raise Unsupported(f.annotation, f"field '{f.name}': {t} fields are not supported")
        if isinstance(t, StructType) and t.name == info.name:
            raise Unsupported(f.annotation, f"field '{f.name}' holds a {info.name} inside a {info.name}")
        f.type = t
        if f.default is not None and is_array(t):
            raise Unsupported(f.default, f"field '{f.name}': an array default would be shared by every object "
                                         "(Python refuses it too)",
                              "use field(default_factory=lambda: np.zeros(n))")


def struct_order(classes) -> list:
    """Classes ordered so each struct is defined after the structs its fields hold."""
    order, state = [], {}

    def visit(name, chain):
        if state.get(name) == "done":
            return
        if state.get(name) == "active":
            raise Unsupported(classes[name].node, "classes that hold each other in a cycle are not supported")
        state[name] = "active"
        for f in classes[name].fields:
            t = f.type.inner if isinstance(f.type, OptionalType) else f.type
            if isinstance(t, StructType) and t.name in classes:
                visit(t.name, chain + [name])
        state[name] = "done"
        order.append(name)

    for n in classes:
        visit(n, [])
    return order
