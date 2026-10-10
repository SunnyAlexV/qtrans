"""Expression translation: one Python expression -> one C++ expression plus its type.

Every method returns an `Expr`, which carries:
  code   the C++ text
  prec   C++ precedence of the outermost operator (0 = atom), used to add exactly the
         parentheses C++ needs and no more, so the output stays readable
  type   the qtrans type (see types.py)

The places where a naive token-for-token translation would compute a different number
are handled explicitly and commented below; search for "Python semantics".
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from . import library as lib
from .errors import Unsupported
from .arrays import ArrayOps
from .types import BOOL, FLOAT, INT, RNG, VOID, ArrayType, MaskType, Scalar, TupleType, assignable, is_array, join

from .core import (CPP_RESERVED, P_ADD, P_AND, P_ATOM, P_COND, P_EQ, P_MUL, P_OR, P_REL,  # noqa: F401
                   P_UNARY, Expr, cpp_ident, float_literal, wrap)

_UNSUPPORTED_EXPR = {
    "List": ("lists", "arrays arrive in stage 2"),
    "Dict": ("dicts", None),
    "Set": ("sets", None),
    "ListComp": ("list comprehensions", "arrays arrive in stage 2"),
    "SetComp": ("set comprehensions", None),
    "DictComp": ("dict comprehensions", None),
    "GeneratorExp": ("generator expressions", "use an explicit for-loop over range()"),
    "Lambda": ("lambdas", "define a named function at module level"),
    "JoinedStr": ("f-strings", "strings are only allowed as error messages"),
    "Subscript": ("indexing / subscripts", "arrays arrive in stage 2"),
    "Starred": ("starred expressions", None),
    "NamedExpr": ("walrus (:=) expressions", "assign on a separate line"),
    "Await": ("await expressions", None),
    "Yield": ("generators (yield)", None),
    "YieldFrom": ("generators (yield from)", None),
    "Slice": ("slices", "arrays arrive in stage 2"),
}

_CMP = {ast.Lt: ("<", P_REL), ast.LtE: ("<=", P_REL), ast.Gt: (">", P_REL),
        ast.GtE: (">=", P_REL), ast.Eq: ("==", P_EQ), ast.NotEq: ("!=", P_EQ)}

_BUILTINS = {"abs", "min", "max", "float", "int", "bool", "round", "pow"}
_REJECTED_BUILTINS = {
    "print": "printing / I/O isn't translated; keep it in the Python caller",
    "len": "len() needs sequences (arrays arrive in stage 2)",
    "sum": "sum() needs sequences; use a for-loop over range() (arrays arrive in stage 2)",
    "range": "range() is only supported as the iterable of a for-loop",
    "divmod": "divmod() returns a pair; use // and % separately",
    "isinstance": "run-time type checks have no meaning in typed C++",
    "complex": "complex numbers are not supported",
    "str": "strings are not supported",
}


class _Unknown(Exception):
    """Raised during type inference when a variable's type isn't known yet."""


class ExprTranslator(ArrayOps):
    def __init__(self, mod):
        self.mod = mod  # the ModuleTranslator: imports, constants, function signatures
        self.notes: list[str] = []
        self._noted: set = set()
        self.current = None  # statement being translated, for error line numbers

    # --- hooks overridden by FunctionTranslator --------------------------------------
    def is_variable(self, name: str) -> bool:
        return False

    def variable(self, node) -> Expr:
        raise NotImplementedError

    # --- helpers ---------------------------------------------------------------------
    def note(self, key, text):
        if key not in self._noted:
            self._noted.add(key)
            self.notes.append((key, text))

    def need_scalar(self, e: Expr, node, what: str):
        if not isinstance(e.type, Scalar):
            hint = None
            if is_array(e.type) or isinstance(e.type, MaskType):
                hint = "for arrays use the numpy function (np.exp, np.sum, ...), not math or builtins"
            raise Unsupported(node or self.current, f"{what} needs a number, but got a {e.type} value", hint)

    def as_double(self, e: Expr) -> Expr:
        if e.type == FLOAT:
            return e
        if isinstance(e.value, (bool, int)):
            code, prec = float_literal(float(e.value))
            return Expr(code, prec, FLOAT, value=float(e.value), simple=True)
        return Expr(f"static_cast<double>({e.code})", P_ATOM, FLOAT)

    def as_type(self, e: Expr, t) -> Expr:
        """Convert so the C++ type is exactly t. Needed where C++ won't convert for us:
        std::max(x, 0) with x a long long fails to compile, and qt::mod(x, 7) would be
        ambiguous between the long long and double overloads."""
        if t == FLOAT:
            return self.as_double(e)
        if t == INT:
            if e.type == INT and e.int_lit:
                return Expr(e.code + "LL", e.prec, INT, value=e.value, simple=e.simple)
            if e.type == BOOL:
                return Expr(f"static_cast<long long>({e.code})", P_ATOM, INT)
        return e

    def promote(self, e: Expr, t) -> Expr:
        """Make an int -> double conversion explicit when the int isn't a literal. The value is
        the same either way (Python converts too); writing it out keeps the C++ clean under
        -Wconversion and shows the reader where Python promotes an int."""
        if t == FLOAT and e.type in (INT, BOOL) and e.value is None:
            return self.as_double(e)
        return e

    def truth(self, e: Expr, node=None) -> Expr:
        """Python truthiness written out explicitly (`if x:` -> `if (x != 0.0)`).
        NaN is truthy in Python and NaN != 0.0 is true in C++, so they agree."""
        if e.type == BOOL:
            return e
        if e.type == INT:
            return Expr(f"{wrap(e, P_REL)} != 0", P_EQ, BOOL)
        if e.type == FLOAT:
            return Expr(f"{wrap(e, P_REL)} != 0.0", P_EQ, BOOL)
        if is_array(e.type) or isinstance(e.type, MaskType):
            raise Unsupported(node or self.current, "the truth value of an array is ambiguous "
                              "(NumPy raises ValueError)", "use .any() or .all()")
        raise Unsupported(node or self.current, f"a {e.type} value can't be used as a condition")

    # --- dispatch ----------------------------------------------------------------------
    def expr(self, node) -> Expr:
        method = getattr(self, "e_" + type(node).__name__, None)
        if method is None:
            what, hint = _UNSUPPORTED_EXPR.get(type(node).__name__, (type(node).__name__, None))
            raise Unsupported(node, f"{what} are not supported in stage 1", hint)
        return method(node)

    def e_Constant(self, node):
        v = node.value
        if isinstance(v, bool):
            return Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True)
        if isinstance(v, int):
            if abs(v) >= 2 ** 63:
                raise Unsupported(node, "integer literal does not fit in 64 bits")
            if v >= 2 ** 31:
                return Expr(f"{v}LL", P_ATOM, INT, value=v, simple=True)
            return Expr(str(v), P_ATOM, INT, int_lit=True, value=v, simple=True)
        if isinstance(v, float):
            code, prec = float_literal(v)
            return Expr(code, prec, FLOAT, value=v, simple=True)
        if isinstance(v, str):
            raise Unsupported(node, "string values are not supported",
                              "strings are only allowed as exception messages")
        if v is None:
            raise Unsupported(node, "None is not supported as a value",
                              "return a number, or raise an exception for the error case")
        raise Unsupported(node, f"{type(v).__name__} literals are not supported")

    def e_Name(self, node):
        if self.is_variable(node.id):
            return self.variable(node)
        return self.mod.global_name(node, self)

    def _is_module_path(self, node) -> bool:
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and self.qualify(node) is not None

    def qualify(self, node):
        """`np.exp` -> 'numpy.exp', `norm.cdf` -> 'scipy.stats.norm.cdf'; None if the
        expression isn't a path into an imported module."""
        if isinstance(node, ast.Name):
            if self.is_variable(node.id) or node.id in self.mod.constants or node.id in self.mod.sigs:
                return None
            return self.mod.imports.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualify(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def e_Subscript(self, node):
        return self.array_subscript(node)

    def e_Attribute(self, node):
        q = self.qualify(node)
        src = ast.unparse(node)
        if q is None and not self._is_module_path(node.value):
            base = self.expr(node.value)
            if is_array(base.type):
                return self.array_attribute(node, base)
        if q is None:
            raise Unsupported(node, f"attribute access '{src}' is not supported in stage 1",
                              "objects and classes arrive in stage 4")
        if q in lib.CONSTANTS:
            return Expr(lib.CONSTANTS[q], P_ATOM, FLOAT, simple=True)
        raise Unsupported(node, f"'{src}' ({q}) can't be used as a value")

    # --- operators -----------------------------------------------------------------------
    def e_BinOp(self, node):
        return self.binop(node, node.op, self.expr(node.left), self.expr(node.right))

    def binop(self, node, op, L: Expr, R: Expr) -> Expr:
        node = node or self.current
        if any(is_array(e.type) or isinstance(e.type, MaskType) for e in (L, R)):
            return self.array_binop(node, op, L, R)
        if isinstance(op, ast.MatMult):
            raise Unsupported(node, "@ needs arrays on both sides")
        self.need_scalar(L, node, "arithmetic")
        self.need_scalar(R, node, "arithmetic")
        both_int = L.type != FLOAT and R.type != FLOAT

        if isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
            sym, p = {ast.Add: ("+", P_ADD), ast.Sub: ("-", P_ADD), ast.Mult: ("*", P_MUL)}[type(op)]
            if not both_int:
                L, R = self.promote(L, FLOAT), self.promote(R, FLOAT)
            if both_int:
                # Python semantics: ints never overflow. `100000 * 100000` with two C++
                # int literals would overflow 32 bits, so promote one side to long long.
                if L.int_lit and R.int_lit:
                    L = self.as_type(L, INT)
                self.note("int64", "int arithmetic uses 64-bit long long; Python ints are "
                                   "unbounded, so results beyond ±9.2e18 would differ")
            return Expr(f"{wrap(L, p)} {sym} {wrap(R, p - 1)}", p, INT if both_int else FLOAT)

        if isinstance(op, ast.Div):
            # Python semantics: `/` is always true division. In C++ `7 / 2` is 3.
            if both_int:
                L = self.as_double(L)
            R = self.promote(R, FLOAT)
            if R.value is None or R.value == 0:
                self.note("div0", "division by zero gives ±inf or NaN in C++ where Python "
                                  "raises ZeroDivisionError")
            return Expr(f"{wrap(L, P_MUL)} / {wrap(R, P_MUL - 1)}", P_MUL, FLOAT)

        if isinstance(op, (ast.FloorDiv, ast.Mod)):
            # Python semantics: // rounds toward -inf and % takes the divisor's sign.
            # C++ / and % truncate toward zero, so -7 // 2 would give -3 instead of -4.
            fn = "floordiv" if isinstance(op, ast.FloorDiv) else "mod"
            if both_int:
                a, b = self.as_type(L, INT), self.as_type(R, INT)
                return Expr(f"qt::{fn}({a.code}, {b.code})", P_ATOM, INT)
            a, b = self.as_double(L), self.as_double(R)
            return Expr(f"qt::{fn}({a.code}, {b.code})", P_ATOM, FLOAT)

        if isinstance(op, ast.Pow):
            return self.power(node, L, R)

        raise Unsupported(node, f"the {type(op).__name__} operator is not supported in stage 1")

    def power(self, node, L: Expr, R: Expr) -> Expr:
        node = node or self.current
        self.need_scalar(L, node, "**")
        self.need_scalar(R, node, "**")
        if L.type != FLOAT and R.type != FLOAT:
            # Python semantics: int ** int is an int when the exponent is >= 0 and a float
            # when it is negative. The type therefore depends on the exponent's value, so
            # it must be a constant we can see.
            if isinstance(R.value, (bool, int)):
                n = int(R.value)
                if n < 0:
                    return Expr(f"std::pow({self.as_double(L).code}, {n}.0)", P_ATOM, FLOAT)
                a = self.as_type(L, INT)
                self.note("int64", "int arithmetic uses 64-bit long long; Python ints are "
                                   "unbounded, so results beyond ±9.2e18 would differ")
                if n == 2 and L.simple and not str(L.value or "").startswith("-"):
                    return Expr(f"{a.code} * {a.code}", P_MUL, INT)
                return Expr(f"qt::ipow({a.code}, {n})", P_ATOM, INT)
            raise Unsupported(node, "int ** int with a non-constant exponent: Python returns an "
                                    "int or a float depending on the exponent's sign at run time",
                              "convert one side with float(...)")
        if R.value is not None and float(R.value) == 2.0 and L.simple and L.prec == P_ATOM:
            # x ** 2 == x * x exactly (one correctly rounded multiplication)
            return Expr(f"{L.code} * {L.code}", P_MUL, FLOAT)
        self.note("powoverflow", "float ** raises OverflowError in Python when the result "
                                 "overflows; C++ returns inf")
        integral = R.type != FLOAT or (R.value is not None and float(R.value) == int(float(R.value)))
        if not integral:
            self.note("complexpow", "x ** y with a negative x and non-integer y gives a complex "
                                    "number in Python but NaN in C++")
        return Expr(f"std::pow({L.code}, {R.code})", P_ATOM, FLOAT)

    def e_UnaryOp(self, node):
        e = self.expr(node.operand)
        if isinstance(e.type, MaskType) and isinstance(node.op, ast.Invert):
            return Expr(f"!{wrap(e, P_ATOM)}", P_UNARY, e.type)
        if is_array(e.type):
            if isinstance(node.op, ast.USub):
                inner = wrap(e, P_UNARY)
                return Expr("-" + (f"({inner})" if inner.startswith("-") else inner), P_UNARY, e.type)
            if isinstance(node.op, ast.UAdd):
                return e
        if isinstance(node.op, ast.Not):
            return Expr(f"!{wrap(self.truth(e, node), P_ATOM)}", P_UNARY, BOOL)
        self.need_scalar(e, node, "unary +/-")
        t = INT if e.type == BOOL else e.type
        if isinstance(node.op, ast.UAdd):
            return Expr(e.code, e.prec, t, e.int_lit, e.value, e.simple)
        if isinstance(node.op, ast.USub):
            inner = wrap(e, P_UNARY)
            if inner.startswith("-"):
                inner = f"({inner})"  # `- -x` must not become the decrement operator `--x`
            value = -e.value if isinstance(e.value, (int, float)) else None
            return Expr("-" + inner, P_UNARY, t, e.int_lit, value, e.simple and value is not None)
        raise Unsupported(node, f"the {type(node.op).__name__} operator is not supported in stage 1")

    def e_BoolOp(self, node):
        parts = [self.expr(v) for v in node.values]
        for v, e in zip(node.values, parts):
            if is_array(e.type) or isinstance(e.type, MaskType):
                raise Unsupported(v, "`and`/`or` on arrays raises ValueError in NumPy",
                                  "use & and | with parentheses: (a > 0) & (b > 0)")
            if e.type != BOOL:
                raise Unsupported(v, "`and`/`or` on non-bool values returns one of the operands in "
                                     "Python, which has no direct C++ equivalent",
                                  "compare explicitly, e.g. `x > 0 and y > 0`")
        if isinstance(node.op, ast.And):
            return Expr(" && ".join(wrap(p, P_AND) for p in parts), P_AND, BOOL)
        return Expr(" || ".join(wrap(p, P_AND - 1) for p in parts), P_OR, BOOL)

    def e_Compare(self, node):
        vals = [node.left] + list(node.comparators)
        exprs = [self.expr(v) for v in vals]
        if any(is_array(e.type) for e in exprs):
            if len(node.ops) > 1:
                raise Unsupported(node, "chained comparisons of arrays raise ValueError in NumPy")
            return self.array_compare(node, exprs[0], node.ops[0], exprs[1])
        for v, e in zip(vals, exprs):
            self.need_scalar(e, v, "comparison")
        if len(node.ops) > 1:
            # Python semantics: a < b < c means (a < b) and (b < c) with b evaluated once.
            for v, e in zip(vals[1:-1], exprs[1:-1]):
                if not e.simple:
                    raise Unsupported(v, "chained comparison with a computed middle term",
                                      "store the middle value in a variable first")
        parts = []
        for op, a, b in zip(node.ops, exprs, exprs[1:]):
            if type(op) not in _CMP:
                raise Unsupported(node, f"the `{type(op).__name__}` comparison is not supported "
                                        "(no `is`, `in`)")
            sym, p = _CMP[type(op)]
            parts.append(Expr(f"{wrap(a, p)} {sym} {wrap(b, p - 1)}", p, BOOL))
        if len(parts) == 1:
            return parts[0]
        return Expr(" && ".join(wrap(x, P_AND) for x in parts), P_AND, BOOL)

    def e_IfExp(self, node):
        c = self.truth(self.expr(node.test), node)
        b, o = self.expr(node.body), self.expr(node.orelse)
        t = join(b.type, o.type)
        if t is None or t == VOID:
            raise Unsupported(node, f"the two branches of `x if c else y` have types {b.type} and {o.type}")
        if isinstance(t, Scalar):
            b, o = self.as_type(b, t), self.as_type(o, t)
        if isinstance(t, MaskType):
            raise Unsupported(node, "a conditional choosing between boolean arrays is not supported")
        if is_array(t):  # two different Eigen expression types: make both concrete arrays
            b, o = self._plain(b), self._plain(o)
        return Expr(f"{wrap(c, P_COND - 1)} ? {wrap(b, P_COND - 1)} : {wrap(o, P_COND)}", P_COND, t)

    def e_Tuple(self, node):
        parts = [self.expr(e) for e in node.elts]
        for e, p in zip(node.elts, parts):
            if not is_array(p.type):
                self.need_scalar(p, e, "a tuple element")
        parts = [self.as_type(p, p.type) for p in parts]
        return Expr(f"std::make_tuple({', '.join(p.code for p in parts)})", P_ATOM,
                    TupleType(tuple(p.type for p in parts)))

    # --- calls -----------------------------------------------------------------------------
    def e_Call(self, node):
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise Unsupported(node, "*args / **kwargs in calls are not supported")
        f = node.func
        if isinstance(f, ast.Name):
            name = f.id
            if self.is_variable(name):
                raise Unsupported(node, f"'{name}' is a variable, not a function")
            if name in self.mod.sigs:
                return self.call_user(node, name)
            if name in self.mod.imports:
                return self.call_library(node, self.mod.imports[name])
            if name in _BUILTINS:
                return self.call_builtin(node, name)
            if name == "len" and len(node.args) == 1 and not node.keywords:
                a = self.expr(node.args[0])
                if is_array(a.type):
                    return Expr(f"{wrap(a, P_ATOM)}.{'size' if a.type.rank == 1 else 'rows'}()", P_ATOM, INT)
            if name in ("sum", "max", "min") and node.args:
                a = self.expr(node.args[0])
                if is_array(a.type):
                    raise Unsupported(node, f"built-in {name}() of an array is not supported",
                                      f"for arrays use the numpy function np.{name}(...): built-in sum() "
                                      "adds left to right, np.sum pairwise, so their results differ")
            if name in _REJECTED_BUILTINS:
                raise Unsupported(node, f"{name}() is not supported", _REJECTED_BUILTINS[name])
            raise Unsupported(node, f"unknown function '{name}'",
                              "stage 1 translates one module at a time; define it in this file")
        q = self.qualify(f)
        if q is None and isinstance(f, ast.Attribute) and not self._is_module_path(f.value):
            base = self.expr(f.value)
            if is_array(base.type) or isinstance(base.type, MaskType):
                return self.array_method(node, base, f.attr)
            if base.type == RNG:
                return self.rng_method(node, base, f.attr)
        if q is None:
            raise Unsupported(node, f"method call '{ast.unparse(f)}(...)' is not supported",
                              "objects and classes arrive in stage 4")
        return self.call_library(node, q)

    def call_user(self, node, name):
        if name in self.mod.rejected:
            raise Unsupported(node, f"calls {name}(), which could not be translated")
        sig = self.mod.sigs[name]
        params = sig.params
        if len(node.args) > len(params):
            raise Unsupported(node, f"{name}() takes {len(params)} arguments but {len(node.args)} were given")
        bound = {}
        for p, a in zip(params, node.args):
            bound[p.name] = a
        for k in node.keywords:
            if k.arg not in {p.name for p in params}:
                raise Unsupported(node, f"{name}() has no parameter '{k.arg}'")
            if k.arg in bound:
                raise Unsupported(node, f"{name}() got two values for '{k.arg}'")
            bound[k.arg] = k.value
        last = max((i for i, p in enumerate(params) if p.name in bound), default=-1)
        codes = []
        passed_arrays = []
        for i, p in enumerate(params):
            if p.name in bound and p.type == RNG:
                a = bound[p.name]
                e = self.expr(a)
                if e.type != RNG or not isinstance(a, ast.Name):
                    raise Unsupported(a, f"argument '{p.name}' of {name}() must be a generator variable")
                codes.append(e.code)
                continue
            if p.name in bound and is_array(p.type):
                a = bound[p.name]
                e = self.expr(a)
                if e.type != p.type:
                    raise Unsupported(a, f"argument '{p.name}' of {name}() must be a {p.type}, got {e.type}")
                if p.mode == "ref":
                    if not isinstance(a, ast.Name):
                        raise Unsupported(a, f"{name}() modifies its argument '{p.name}' in place, so it "
                                             "must be given an array variable", "store the value in a variable first")
                if isinstance(a, ast.Name):
                    passed_arrays.append(a.id)
                codes.append(e.code)
                continue
            if p.name in bound:
                e = self.expr(bound[p.name])
                self.need_scalar(e, bound[p.name], f"argument '{p.name}' of {name}()")
                if not assignable(p.type, e.type):
                    raise Unsupported(bound[p.name], f"argument '{p.name}' of {name}() is declared "
                                                     f"{p.type} but a {e.type} is passed",
                                      f"annotate '{p.name}' as float, or convert with int(...)")
                codes.append(self.promote(e, p.type).code)
            elif p.default_code is not None:
                if i < last:  # a later argument was passed by keyword: spell the default out
                    codes.append(p.default_code)
            else:
                raise Unsupported(node, f"{name}() is missing argument '{p.name}'")
        if any(p.mode == "ref" for p in params) and len(set(passed_arrays)) < len(passed_arrays):
            raise Unsupported(node, f"the same array is passed twice to {name}(), which modifies an "
                                    "argument in place: the two parameters would share memory in Python",
                              "pass a copy for one of them")
        return Expr(f"{sig.cpp}({', '.join(codes)})", P_ATOM, sig.ret)

    def _arity(self, node, args, lo, hi=None):
        hi = lo if hi is None else hi
        if not lo <= len(args) <= hi:
            want = str(lo) if lo == hi else f"{lo} to {hi}"
            raise Unsupported(node, f"{ast.unparse(node.func)}() takes {want} arguments here, got {len(args)}")

    def absolute(self, e: Expr) -> Expr:
        if e.type == FLOAT:
            return Expr(f"std::fabs({e.code})", P_ATOM, FLOAT)
        return Expr(f"std::abs({self.as_type(e, INT).code})", P_ATOM, INT)

    def call_builtin(self, node, name):
        if node.keywords:
            raise Unsupported(node, f"keyword arguments to {name}() are not supported")
        if name == "float" and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            v = node.args[0].value.strip().lower()
            special = {"inf": "qt::inf", "infinity": "qt::inf", "+inf": "qt::inf",
                       "-inf": "-qt::inf", "-infinity": "-qt::inf", "nan": "qt::nan"}
            if v in special:
                return Expr(special[v], P_UNARY if special[v][0] == "-" else P_ATOM, FLOAT)
            try:
                return self.expr(ast.copy_location(ast.Constant(float(v)), node))
            except ValueError:
                raise Unsupported(node, f"float({node.args[0].value!r}) is not a valid number")
        args = [self.expr(a) for a in node.args]
        if name == "abs" and len(args) == 1 and is_array(args[0].type):
            return Expr(f"{wrap(args[0], P_ATOM)}.abs()", P_ATOM, args[0].type)
        for a, n in zip(args, node.args):
            self.need_scalar(a, n, f"{name}()")
        if name == "abs":
            self._arity(node, args, 1)
            return self.absolute(args[0])
        if name in ("min", "max"):
            if len(args) < 2:
                raise Unsupported(node, f"{name}() of a sequence needs arrays (stage 2)")
            t = args[0].type
            for a in args[1:]:
                t = join(t, a.type)
            conv = [self.as_type(a, t) for a in args]
            # std::max(a, b) returns (a < b) ? b : a, the same element Python's max picks,
            # including which argument wins on ties and how NaN propagates.
            if len(conv) == 2:
                return Expr(f"std::{name}({conv[0].code}, {conv[1].code})", P_ATOM, t)
            return Expr(f"std::{name}({{{', '.join(c.code for c in conv)}}})", P_ATOM, t)
        if name == "float":
            self._arity(node, args, 0, 1)
            return self.as_double(args[0]) if args else Expr("0.0", P_ATOM, FLOAT, value=0.0, simple=True)
        if name == "int":
            self._arity(node, args, 0, 1)
            if not args:
                return Expr("0", P_ATOM, INT, int_lit=True, value=0, simple=True)
            a = args[0]
            if a.type == FLOAT:
                # Python's int() truncates toward zero, as the cast does; qt::to_int adds
                # Python's errors for NaN/inf instead of C++'s undefined behaviour.
                self.note("toint", "float -> int beyond ±9.2e18 raises OverflowError in C++; "
                                   "Python returns a big integer")
                return Expr(f"qt::to_int({a.code})", P_ATOM, INT)
            return self.as_type(a, INT)
        if name == "bool":
            self._arity(node, args, 0, 1)
            return self.truth(args[0], node) if args else Expr("false", P_ATOM, BOOL, value=False, simple=True)
        if name == "round":
            if len(args) == 2:
                raise Unsupported(node, "round(x, ndigits) is not supported: Python rounds the exact "
                                        "decimal value of the double, which C++ can't reproduce simply",
                                  "multiply, round and divide explicitly if you need it")
            self._arity(node, args, 1)
            a = args[0]
            if a.type == FLOAT:
                # Python semantics: round() is half-to-even (round(2.5) == 2). std::round
                # rounds half away from zero; std::nearbyint uses the default
                # round-to-nearest-even mode, which matches Python.
                self.note("toint", "float -> int beyond ±9.2e18 raises OverflowError in C++; "
                                   "Python returns a big integer")
                return Expr(f"qt::to_int(std::nearbyint({a.code}))", P_ATOM, INT)
            return self.as_type(a, INT)
        if name == "pow":
            if len(args) == 3:
                raise Unsupported(node, "three-argument pow() is not supported")
            self._arity(node, args, 2)
            return self.power(node, args[0], args[1])
        raise Unsupported(node, f"{name}() is not supported")

    def call_library(self, node, qual):
        src = ast.unparse(node.func)
        if qual in lib.NORM:
            return self.call_norm(node, qual)
        if qual in ("numpy.array", "numpy.asarray"):
            return self.array_literal(node, src)
        if qual == "numpy.random.default_rng":
            return self.default_rng(node)
        if qual.startswith("numpy.random."):
            raise Unsupported(node, f"{src}() uses NumPy's legacy global random state",
                              "create a generator and draw from it: rng = np.random.default_rng(seed); "
                              "rng.standard_normal(n)")
        args = [self.expr(a) for a in node.args]
        if self.wants_array_call(qual, args):
            kws = {k.arg: self.expr(k.value) for k in node.keywords if k.arg != "dtype"}
            return self.array_call(node, qual, args, kws)
        if node.keywords:
            raise Unsupported(node, f"keyword arguments to {src}() are not supported")
        for a, n in zip(args, node.args):
            self.need_scalar(a, n, f"{src}()")

        if qual in lib.MATH_DOMAIN:
            self.note("domain:" + qual, f"{qual} raises ValueError outside its domain in Python; "
                                        "C++ returns NaN")
        if qual in lib.MATH_RANGE:
            self.note("range:" + qual, f"{qual} raises OverflowError on overflow in Python; "
                                       "C++ returns inf")

        if qual in lib.UNARY_FLOAT:
            self._arity(node, args, 1)
            return Expr(f"{lib.UNARY_FLOAT[qual]}({args[0].code})", P_ATOM, FLOAT)
        if qual == "math.log":
            self._arity(node, args, 1, 2)
            if len(args) == 1:
                return Expr(f"std::log({args[0].code})", P_ATOM, FLOAT)
            # math.log(x, base) is computed as log(x) / log(base) by CPython too
            return Expr(f"std::log({args[0].code}) / std::log({args[1].code})", P_MUL, FLOAT)
        if qual in lib.BINARY_FLOAT:
            self._arity(node, args, 2)
            return Expr(f"{lib.BINARY_FLOAT[qual]}({args[0].code}, {args[1].code})", P_ATOM, FLOAT)
        if qual in lib.PREDICATES:
            self._arity(node, args, 1)
            return Expr(f"{lib.PREDICATES[qual]}({args[0].code})", P_ATOM, BOOL)
        if qual in ("math.floor", "math.ceil", "math.trunc"):
            # Python semantics: math.floor returns an *int* (numpy.floor returns a float).
            self._arity(node, args, 1)
            a = args[0]
            if a.type != FLOAT:
                return self.as_type(a, INT)
            fn = qual.split(".")[1]
            self.note("toint", "float -> int beyond ±9.2e18 raises OverflowError in C++; "
                               "Python returns a big integer")
            return Expr(f"qt::to_int(std::{fn}({a.code}))", P_ATOM, INT)
        if qual == "math.pow":
            self._arity(node, args, 2)
            return Expr(f"std::pow({args[0].code}, {args[1].code})", P_ATOM, FLOAT)
        if qual == "numpy.power":
            self._arity(node, args, 2)
            return self.power(node, args[0], args[1])
        if qual in ("numpy.abs", "numpy.absolute"):
            self._arity(node, args, 1)
            return self.absolute(args[0])
        if qual in ("numpy.maximum", "numpy.minimum"):
            self._arity(node, args, 2)
            a, b = args
            if a.type != FLOAT and b.type != FLOAT:
                fn = "max" if qual.endswith("maximum") else "min"
                return Expr(f"std::{fn}({self.as_type(a, INT).code}, {self.as_type(b, INT).code})", P_ATOM, INT)
            # numpy.maximum propagates NaN from either side; std::max does not.
            fn = "np_maximum" if qual.endswith("maximum") else "np_minimum"
            return Expr(f"qt::{fn}({self.as_double(a).code}, {self.as_double(b).code})", P_ATOM, FLOAT)
        if qual == "numpy.sign":
            self._arity(node, args, 1)
            a = args[0] if args[0].type != BOOL else self.as_type(args[0], INT)
            return Expr(f"qt::np_sign({self.as_type(a, a.type).code})", P_ATOM, a.type)
        if qual == "numpy.square":
            self._arity(node, args, 1)
            a = self.as_type(args[0], INT) if args[0].type != FLOAT else args[0]
            return Expr(f"qt::square({a.code})", P_ATOM, a.type)
        raise Unsupported(node, f"{src}() ({qual}) is not in stage 1's library table",
                          "see the README for the supported functions")

    def call_norm(self, node, qual):
        """scipy.stats.norm.cdf/sf/pdf/ppf. Plain calls map to qt::norm_cdf(x) etc. With a
        scale, the 3-argument runtime overloads apply loc and scale the way scipy does
        (cdf((x - loc) / scale), ppf(q) * scale + loc) and return NaN for scale <= 0."""
        fn = qual.rsplit(".", 1)[1]
        names = ["q" if fn == "ppf" else "x", "loc", "scale"]
        if len(node.args) > 3:
            raise Unsupported(node, f"norm.{fn}() takes at most 3 arguments")
        bound = dict(zip(names, node.args))
        for k in node.keywords:
            if k.arg not in names or k.arg in bound:
                raise Unsupported(node, f"unsupported keyword '{k.arg}' for norm.{fn}()")
            bound[k.arg] = k.value
        if names[0] not in bound:
            raise Unsupported(node, f"norm.{fn}() needs its first argument")
        ex = {}
        for k, v in bound.items():
            ex[k] = self.expr(v)
            if k == names[0] and is_array(ex[k].type):
                continue
            self.need_scalar(ex[k], v, f"norm.{fn}()")
            ex[k] = self.as_double(ex[k])
        x, loc, scale = ex[names[0]], ex.get("loc"), ex.get("scale")
        if is_array(x.type):
            return self.norm_array(node, fn, x, loc, scale)
        if scale is not None:
            loc_code = loc.code if loc is not None else "0.0"
            return Expr(f"qt::norm_{fn}({x.code}, {loc_code}, {scale.code})", P_ATOM, FLOAT)
        if loc is None:
            return Expr(f"qt::norm_{fn}({x.code})", P_ATOM, FLOAT)
        if fn == "ppf":
            return self.binop(node, ast.Add(), Expr(f"qt::norm_ppf({x.code})", P_ATOM, FLOAT), loc)
        z = self.binop(node, ast.Sub(), x, loc)
        return Expr(f"qt::norm_{fn}({z.code})", P_ATOM, FLOAT)
