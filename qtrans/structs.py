"""Expressions on objects (stage 5): fields, properties, methods, constructors, ==.

Mixed into ExprTranslator. Inside a method, `self` is the C++ `*this`: `self.S` becomes the
member `S` (or `this->S` when a local of the same name would hide it), `self.price()`
becomes `price()`.
"""
from __future__ import annotations

import ast

from .core import P_ATOM, P_EQ, Expr, wrap
from .errors import Unsupported
from .types import BOOL, OptionalType, Scalar, StructType, assignable

THIS = "(*this)"


class StructOps:
    """Mixin for ExprTranslator: everything that involves dataclass / NamedTuple objects."""

    # --- helpers ---------------------------------------------------------------------------
    def class_info(self, name, node):
        info = self.mod.classes.get(name)
        if info is None:
            return None
        if info.rejected is not None:
            raise Unsupported(node, f"uses class {name}, which could not be translated: {info.rejected.message}")
        return info

    def _member(self, base: Expr, cpp_name: str) -> str:
        if base.code == THIS:
            # locals named like a member are renamed (see ident), so a plain name is the member;
            # `this->` only where Python itself uses the same name for a local and a method
            hidden = cpp_name in {self.ident(n) for n in getattr(self, "py_names", ())}
            return f"this->{cpp_name}" if hidden else cpp_name
        return f"{wrap(base, P_ATOM)}.{cpp_name}"

    def convert_to(self, e: Expr, t, node, what) -> Expr:
        """e as a value of the declared type t (a field or parameter)."""
        if isinstance(t, OptionalType):
            if not assignable(t, e.type):
                raise Unsupported(node, f"{what} is declared {t} but gets a {e.type}")
            return self.to_optional(e, t)
        if e.type == t:
            return e
        if isinstance(t, Scalar) and isinstance(e.type, Scalar) and assignable(t, e.type):
            return self.as_type(e, t)
        raise Unsupported(node, f"{what} is declared {t} but gets a {e.type}")

    # --- attributes ------------------------------------------------------------------------
    def struct_attribute(self, node, base: Expr) -> Expr:
        info = self.class_info(base.type.name, node)
        f = info.field(node.attr)
        if f is not None:
            return Expr(self._member(base, f.cpp), P_ATOM, f.type, simple=base.simple)
        m = info.methods.get(node.attr)
        if m is not None and m.kind == "property":
            sig = self._method_sig(m, node)
            return Expr(self._member(base, sig.cpp) + "()", P_ATOM, sig.ret)
        if m is not None:
            raise Unsupported(node, f"using method {info.name}.{node.attr} as a value is not supported", "call it")
        raise Unsupported(node, f"{info.name} has no field or property '{node.attr}'")

    def _method_sig(self, m, node):
        if m.key in self.mod.rejected:
            raise Unsupported(node, f"calls {m.key}(), which could not be translated")
        sig = self.mod.sigs.get(m.key)
        if sig is None:
            raise Unsupported(node, f"{m.key} is not translated")
        return sig

    # --- calls -------------------------------------------------------------------------------
    def struct_call(self, node):
        """Constructor, static / class method, or method call; None if `node` is none of these."""
        f = node.func
        owner = self.mod.classes.get(getattr(getattr(self, "sig", None), "owner", None) or "")
        if isinstance(f, ast.Name) and not self.is_variable(f.id) and self.qualify(f) == "dataclasses.replace":
            return self.replace_call(node)
        if isinstance(f, ast.Name):
            if f.id == "cls" and owner is not None and self.sig.kind == "class":
                return self.construct(node, owner)
            if f.id in self.mod.classes and not self.is_variable(f.id):
                return self.construct(node, self.class_info(f.id, node))
            return None
        if not isinstance(f, ast.Attribute):
            return None
        # Class.method(...) / cls.method(...)
        if isinstance(f.value, ast.Name) and not self.is_variable(f.value.id):
            name = owner.name if (f.value.id == "cls" and owner is not None) else f.value.id
            if name in self.mod.classes:
                info = self.class_info(name, node)
                m = info.methods.get(f.attr)
                if m is None or m.kind not in ("static", "class"):
                    raise Unsupported(node, f"{name}.{f.attr}(...) must be a @staticmethod or @classmethod")
                sig = self._method_sig(m, node)
                args = self.bind_args(node, sig, f"{name}.{f.attr}")
                return Expr(f"{info.cpp}::{sig.cpp}({', '.join(args)})", P_ATOM, sig.ret)
        if self.qualify(f) == "dataclasses.replace":
            return self.replace_call(node)
        if self._is_module_path(f.value):
            return None
        base = self.expr(f.value)
        if not isinstance(base.type, StructType):
            return None
        info = self.class_info(base.type.name, node)
        m = info.methods.get(f.attr)
        if m is None:
            if info.field(f.attr) is not None:
                raise Unsupported(node, f"field {info.name}.{f.attr} is not callable")
            raise Unsupported(node, f"{info.name} has no method '{f.attr}'")
        if m.kind == "property":
            raise Unsupported(node, f"{info.name}.{f.attr} is a property: use it without ()")
        sig = self._method_sig(m, node)
        args = self.bind_args(node, sig, f"{info.name}.{f.attr}")
        if m.kind in ("static", "class"):
            return Expr(f"{info.cpp}::{sig.cpp}({', '.join(args)})", P_ATOM, sig.ret)
        return Expr(f"{self._member(base, sig.cpp)}({', '.join(args)})", P_ATOM, sig.ret)

    def construct(self, node, info) -> Expr:
        """Option(S=100.0, K=95.0, ...): the arguments in field order, defaults spelled out where
        a later field is given. The C++ constructor then runs __post_init__, as Python does."""
        fields = info.init_fields
        if len(node.args) > len(fields):
            raise Unsupported(node, f"{info.name}() takes {len(fields)} arguments but {len(node.args)} were given")
        bound = dict(zip((f.name for f in fields), node.args))
        for k in node.keywords:
            if k.arg is None or k.arg not in {f.name for f in fields}:
                raise Unsupported(node, f"{info.name}() has no field '{k.arg}' to set")
            if k.arg in bound:
                raise Unsupported(node, f"{info.name}() got two values for '{k.arg}'")
            bound[k.arg] = k.value
        last = max((i for i, f in enumerate(fields) if f.name in bound), default=-1)
        codes = []
        for i, f in enumerate(fields):
            if f.name not in bound:
                if f.default_code is None:
                    raise Unsupported(node, f"{info.name}() is missing field '{f.name}' (Python raises TypeError)")
                if i < last:
                    codes.append(f.default_code)
                continue
            a = bound[f.name]
            e = self.expr_raw(a) if isinstance(f.type, OptionalType) else self.expr(a)
            codes.append(self.convert_to(e, f.type, a, f"field '{f.name}' of {info.name}").code)
        return Expr(f"{info.cpp}({', '.join(codes)})", P_ATOM, info.type)

    def replace_call(self, node) -> Expr:
        """dataclasses.replace(o, K=100.0): a new object through the constructor (so
        __post_init__ runs again, as in Python), with the other fields taken from o."""
        if len(node.args) != 1:
            raise Unsupported(node, "dataclasses.replace(obj, field=value, ...) is supported")
        o = self.expr(node.args[0])
        if not isinstance(o.type, StructType):
            raise Unsupported(node, "dataclasses.replace needs a dataclass object")
        info = self.class_info(o.type.name, node)
        if info.namedtuple:
            raise Unsupported(node, "use ._replace for a NamedTuple; it is not supported yet")
        if not o.simple:
            raise Unsupported(node.args[0], "store the object in a variable before replace()")
        given = {k.arg: k.value for k in node.keywords}
        for k in given:
            f = info.field(k)
            if f is None:
                raise Unsupported(node, f"{info.name} has no field '{k}'")
            if not f.init:
                raise Unsupported(node, f"replace() can't set '{k}': it has init=False (Python raises ValueError)")
        codes = []
        for f in info.init_fields:
            if f.name in given:
                e = self.expr_raw(given[f.name]) if isinstance(f.type, OptionalType) else self.expr(given[f.name])
                codes.append(self.convert_to(e, f.type, given[f.name], f"field '{f.name}'").code)
            else:
                codes.append(self._member(o, f.cpp))
        return Expr(f"{info.cpp}({', '.join(codes)})", P_ATOM, info.type)

    # --- comparison ------------------------------------------------------------------------
    def struct_compare(self, node, a: Expr, op, b: Expr) -> Expr:
        if not isinstance(op, (ast.Eq, ast.NotEq)):
            raise Unsupported(node, "only == and != compare dataclass objects (ordering needs order=True)")
        if a.type != b.type:
            raise Unsupported(node, f"comparing a {a.type} with a {b.type}: Python says they are never equal")
        info = self.class_info(a.type.name, node)
        if not info.eq:
            raise Unsupported(node, f"{info.name} has eq=False, so == compares identity in Python")
        if info.has_array_field(self.mod.classes):
            raise Unsupported(node, f"== on {info.name}: it holds an array, and Python's dataclass == then raises "
                                    "ValueError (the truth value of an array is ambiguous)",
                              "compare the fields you mean, e.g. np.array_equal(a.w, b.w)")
        sym = "==" if isinstance(op, ast.Eq) else "!="
        return Expr(f"{wrap(a, P_EQ)} {sym} {wrap(b, P_EQ - 1)}", P_EQ, BOOL)
