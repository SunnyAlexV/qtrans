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
from .arrays import STACKERS, ArrayOps
from .structs import StructOps
from .types import (BOOL, FLOAT, INT, NONE, RNG, STR, VOID, ArrayType, MaskType, OptionalType, Scalar, StructType,
                    TupleType, assignable, is_array, join)

from .core import cpp_string
from .core import SIZE_IV, dims_of, int_token, iv_clip, iv_fits, iv_of, iv_op, same_dim
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
    "JoinedStr": ("f-strings", "strings can be compared, passed, returned and joined with +; not formatted"),
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


_FOLD = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
         ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


_FLIP_CMP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def _is_lvalue(node) -> bool:
    """A variable, or a field / element of one: something C++ can bind a reference to."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return isinstance(node, ast.Name)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# calls whose result is a long long whatever their arguments (so no cast is needed around them)
_LONG_LONG_CALLS = {"static_cast<long long>", "std::min", "std::max", "qt::iadd", "qt::isub", "qt::imul",
                    "qt::ineg", "qt::iabs", "qt::ipow", "qt::floordiv", "qt::mod", "qt::to_int", "std::abs"}


def _outer_call(code: str):
    """The function name if `code` is exactly one call, `f(...)`, else None."""
    i = code.find("(")
    if i <= 0 or not code.endswith(")"):
        return None
    depth = 0
    for k in range(i, len(code)):
        depth += {"(": 1, ")": -1}.get(code[k], 0)
        if depth == 0:
            return code[:i] if k == len(code) - 1 else None
    return None


class ExprTranslator(ArrayOps, StructOps):
    def __init__(self, mod):
        self.mod = mod  # the ModuleTranslator: imports, constants, function signatures
        self.notes: list[str] = []
        self._noted: set = set()
        self.current = None  # statement being translated, for error line numbers
        self.used_arrays = False  # any array value at all, even a temporary: needs qtrans_arrays.hpp

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
        # where it applies, for `qtrans explain`: every statement it comes from, as emitted
        src = getattr(self, "cur_src", None)
        if src is not None and not getattr(self, "inferring", False):
            seen = self.__dict__.setdefault("_line_noted", set())
            if (src, key) not in seen:
                seen.add((src, key))
                self.__dict__.setdefault("line_notes", []).append((src, key, text))

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

    def as_type(self, e: Expr, t, index_cast: bool = True) -> Expr:
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
            if index_cast and e.type == INT and any(m in e.code for m in (".size()", ".rows()", ".cols()", ".count()")) \
                    and _outer_call(e.code) not in _LONG_LONG_CALLS:
                # an array's size is Eigen's Index (long on Linux and macOS, a different type
                # from long long): std::min(n, v.size()) wouldn't compile
                return Expr(f"static_cast<long long>({e.code})", P_ATOM, INT, iv=e.iv, dimtok=e.dimtok)
        return e

    def as_string(self, e: Expr) -> Expr:
        """A literal as std::string, so that member functions and + work on it."""
        if e.type == STR and isinstance(e.value, str):
            return Expr(f"std::string({e.code})", P_ATOM, STR, value=e.value)
        return e

    def promote(self, e: Expr, t) -> Expr:
        """Make an int -> double conversion explicit when the int isn't a literal. The value is
        the same either way (Python converts too); writing it out keeps the C++ clean under
        -Wconversion and shows the reader where Python promotes an int."""
        if t == FLOAT and e.type in (INT, BOOL) and e.value is None:
            return self.as_double(e)
        return e

    def cond(self, node) -> Expr:
        """`if x:` / `while x:` / `not x`: an optional is false when None."""
        return self.truth(self.expr_raw(node), node)

    def truth(self, e: Expr, node=None) -> Expr:
        """Python truthiness written out explicitly (`if x:` -> `if (x != 0.0)`).
        NaN is truthy in Python and NaN != 0.0 is true in C++, so they agree."""
        if e.type == NONE:
            return Expr("false", P_ATOM, BOOL, value=False, simple=True)
        if isinstance(e.type, OptionalType):
            if not isinstance(e.type.inner, Scalar):
                raise Unsupported(node or self.current, "the truth value of an array that may be None",
                                  "test `x is None` explicitly")
            return Expr(f"qt::truthy({e.code})", P_ATOM, BOOL)
        if e.type == STR:
            return Expr(f"!{wrap(self.as_string(e), P_ATOM)}.empty()", P_UNARY, BOOL)  # "" is falsy
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
        """The value of `node`, as a number or array. An optional (`x: float | None`) is
        unwrapped with a check, because wherever the value itself is used Python raises
        TypeError on None; the few places that accept None call expr_raw instead."""
        e = self.expr_raw(node)
        if isinstance(e.type, OptionalType):
            return self.unwrap(e)
        if e.type == NONE:
            raise Unsupported(node, "None is used as a value here (Python raises TypeError)",
                              "annotate the variable or parameter as `float | None` and test `x is None`")
        return e

    def expr_raw(self, node) -> Expr:
        method = getattr(self, "e_" + type(node).__name__, None)
        if method is None:
            what, hint = _UNSUPPORTED_EXPR.get(type(node).__name__, (type(node).__name__, None))
            raise Unsupported(node, f"{what} are not supported in stage 1", hint)
        e = method(node)
        t = e.type.inner if isinstance(e.type, OptionalType) else e.type
        if is_array(t) or isinstance(t, MaskType):
            self.used_arrays = True
        return e

    def unwrap(self, e: Expr) -> Expr:
        return Expr(f"qt::value({e.code})", P_ATOM, e.type.inner)

    def to_optional(self, e: Expr, t: OptionalType) -> Expr:
        """e (None, a value, or an optional) as the C++ type t, ready to store or pass."""
        if e.type == NONE:
            return Expr("std::nullopt", P_ATOM, t)
        if isinstance(e.type, OptionalType):
            if e.type.inner == t.inner:
                return e
            return Expr(f"({e.code} ? {t.cpp()}({self.as_type(self.unwrap(e), t.inner).code}) : {t.cpp()}())",
                        P_ATOM, t)
        v = self.as_type(e, t.inner) if isinstance(t.inner, Scalar) else e
        return Expr(v.code, v.prec, t)

    def e_Constant(self, node):
        v = node.value
        if isinstance(v, bool):
            return Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True, iv=(int(v), int(v)))
        if isinstance(v, int):
            if abs(v) >= 2 ** 63:
                raise Unsupported(node, "integer literal does not fit in 64 bits")
            if v >= 2 ** 31:
                return Expr(f"{v}LL", P_ATOM, INT, value=v, simple=True, iv=(v, v))
            return Expr(str(v), P_ATOM, INT, int_lit=True, value=v, simple=True, iv=(v, v))
        if isinstance(v, float):
            code, prec = float_literal(v)
            return Expr(code, prec, FLOAT, value=v, simple=True)
        if isinstance(v, str):
            return Expr(cpp_string(v), P_ATOM, STR, value=v, simple=True)
        if v is None:
            return Expr("std::nullopt", P_ATOM, NONE, simple=True)
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
        if isinstance(node.value, ast.Name) and node.value.id in self.mod.classes \
                and not self.is_variable(node.value.id):
            raise Unsupported(node, f"'{src}': class attributes are not supported", "use a module-level constant")
        if q is None and not self._is_module_path(node.value):
            base = self.expr(node.value)
            if is_array(base.type):
                return self.array_attribute(node, base)
            if isinstance(base.type, StructType):
                return self.struct_attribute(node, base)
        if q is None:
            raise Unsupported(node, f"attribute access '{src}' is not supported here")
        if q in lib.CONSTANTS:
            return Expr(lib.CONSTANTS[q], P_ATOM, FLOAT, simple=True)
        raise Unsupported(node, f"'{src}' ({q}) can't be used as a value")

    # --- operators -----------------------------------------------------------------------
    def e_BinOp(self, node):
        return self.binop(node, node.op, self.expr(node.left), self.expr(node.right))

    def binop(self, node, op, L: Expr, R: Expr) -> Expr:
        node = node or self.current
        if STR in (L.type, R.type):
            if isinstance(op, ast.Add) and L.type == R.type == STR:
                if isinstance(L.value, str) and isinstance(R.value, str):
                    return self.expr(ast.Constant(L.value + R.value))
                return Expr(f"{wrap(self.as_string(L), P_ADD)} + {wrap(R, P_ADD - 1)}", P_ADD, STR)
            raise Unsupported(node, "on strings only + (joining two strings) is supported",
                              "Python raises TypeError for str + number; formatting isn't supported")
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
                return self.int_arith(sym, p, L, R)
            return Expr(f"{wrap(L, p)} {sym} {wrap(R, p - 1)}", p, FLOAT)

        if isinstance(op, ast.Div):
            # Python semantics: `/` is always true division. In C++ `7 / 2` is 3.
            L = self.as_double(L) if both_int else self.promote(L, FLOAT)
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
                # |a // b| <= |a| (b an int, not 0), |a % b| < |b|; both checked at run time
                if fn == "floordiv":
                    m = max(abs(x) for x in iv_of(L))
                else:
                    m = max(abs(x) for x in iv_of(R)) - 1
                return Expr(f"qt::{fn}({a.code}, {b.code})", P_ATOM, INT, iv=iv_clip((-m, m)))
            a, b = self.as_double(L), self.as_double(R)
            return Expr(f"qt::{fn}({a.code}, {b.code})", P_ATOM, FLOAT)

        if isinstance(op, ast.Pow):
            return self.power(node, L, R)

        raise Unsupported(node, f"the {type(op).__name__} operator is not supported in stage 1")

    def int_arith(self, sym, p, L: Expr, R: Expr) -> Expr:
        """a + b, a - b, a * b on ints. Plain C++ when the result provably fits in 64 bits, from
        the intervals the operands are known to lie in (literals, array sizes, range() loop
        indices: `i + 1` for i in range(n) can't overflow, since i < n); otherwise checked
        (qt::iadd, ...): the exact result, or OverflowError where Python's unbounded int would
        leave 64 bits."""
        r = iv_op(sym, iv_of(L), iv_of(R))
        if L.int_lit and R.int_lit:
            L = Expr(self.as_type(L, INT, False).code, P_ATOM, INT, iv=L.iv)
        if iv_fits(r):
            return Expr(f"{wrap(L, p)} {sym} {wrap(R, p - 1)}", p, INT, iv=r)
        self.note("int64", "int arithmetic that could leave 64 bits is checked: where Python's unbounded "
                           "int would exceed ±9.2e18, C++ raises OverflowError")
        fn = {"+": "iadd", "-": "isub", "*": "imul"}[sym]
        # (qt::iadd takes long long: an Index converts, no cast needed)
        return Expr(f"qt::{fn}({self.as_type(L, INT, False).code}, {self.as_type(R, INT, False).code})", P_ATOM, INT,
                    iv=iv_clip(r))

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
                lo, hi = iv_of(L)
                ends = [lo ** n, hi ** n]
                r = (0 if n % 2 == 0 and lo <= 0 <= hi else min(ends), max(ends)) if n else (1, 1)
                if iv_fits(r):
                    if n == 2 and L.simple and not str(L.value or "").startswith("-"):
                        return Expr(f"{a.code} * {a.code}", P_MUL, INT, iv=r)
                    return Expr(f"qt::ipow({a.code}, {n})", P_ATOM, INT, iv=r)
                self.note("int64", "int arithmetic that could leave 64 bits is checked: where Python's unbounded "
                                   "int would exceed ±9.2e18, C++ raises OverflowError")
                return Expr(f"qt::ipow({a.code}, {n})", P_ATOM, INT)  # checked inside
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
                                    "number for a Python float but NaN in C++ (and for np.float64)")
        L, R = self.promote(L, FLOAT), self.promote(R, FLOAT)
        return Expr(f"std::pow({L.code}, {R.code})", P_ATOM, FLOAT)

    def e_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            raw = self.expr_raw(node.operand)
            if raw.type == STR:
                return Expr(f"{wrap(self.as_string(raw), P_ATOM)}.empty()", P_ATOM, BOOL)
            return Expr(f"!{wrap(self.truth(raw, node.operand), P_ATOM)}", P_UNARY, BOOL)
        e = self.expr(node.operand)
        if isinstance(e.type, MaskType) and isinstance(node.op, ast.Invert):
            return Expr(f"!{wrap(e, P_ATOM)}", P_UNARY, e.type, dims=e.dims)
        if is_array(e.type):
            if isinstance(node.op, ast.USub):
                inner = wrap(e, P_UNARY)
                return Expr("-" + (f"({inner})" if inner.startswith("-") else inner), P_UNARY, e.type, dims=e.dims)
            if isinstance(node.op, ast.UAdd):
                return e
        return self._unary(node, e)

    def _unary(self, node, e):
        self.need_scalar(e, node, "unary +/-")
        t = INT if e.type == BOOL else e.type
        if isinstance(node.op, ast.UAdd):
            return Expr(e.code, e.prec, t, e.int_lit, e.value, e.simple)
        if isinstance(node.op, ast.USub):
            inner = wrap(e, P_UNARY)
            if inner.startswith("-"):
                inner = f"({inner})"  # `- -x` must not become the decrement operator `--x`
            value = -e.value if isinstance(e.value, (int, float)) else None
            r = (-iv_of(e)[1], -iv_of(e)[0]) if t == INT else None
            if t == INT and not iv_fits(r):
                return Expr(f"qt::ineg({e.code})", P_ATOM, INT, iv=iv_clip(r))  # -(-2**63) is 2**63
            return Expr("-" + inner, P_UNARY, t, e.int_lit, value, e.simple and value is not None, iv=r)
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
        if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            return self._is_none(node)
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            return self._in_literals(node)
        vals = [node.left] + list(node.comparators)
        exprs = [self.expr(v) for v in vals]
        if any(e.type == STR for e in exprs):
            return self._compare_strings(node, vals, exprs)
        if any(isinstance(e.type, StructType) for e in exprs):
            if len(node.ops) > 1:
                raise Unsupported(node, "chained comparisons of objects are not supported")
            return self.struct_compare(node, exprs[0], node.ops[0], exprs[1])
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
            a, b = exprs
            if _is_num(a.value) and _is_num(b.value):
                # both sides known when translating (A.ndim == 1 for a Mat): fold it, so a
                # branch that can never run is recognised as such
                v = _FOLD[type(node.ops[0])](a.value, b.value)
                return Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True)
            return parts[0]
        return Expr(" && ".join(wrap(x, P_AND) for x in parts), P_AND, BOOL)

    def _compare_strings(self, node, vals, exprs):
        if not all(e.type == STR for e in exprs):
            if all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops) and len(node.ops) == 1:
                raise Unsupported(node, "comparing a string with a number: Python says they are never equal",
                                  "compare like with like")
            raise Unsupported(node, "ordering a string against a number raises TypeError in Python")
        if len(node.ops) > 1:
            for v, e in zip(vals[1:-1], exprs[1:-1]):
                if not e.simple:
                    raise Unsupported(v, "chained comparison with a computed middle term",
                                      "store the middle value in a variable first")
        parts = []
        for op, a, b in zip(node.ops, exprs, exprs[1:]):
            sym, p = _CMP[type(op)]
            if isinstance(a.value, str) and isinstance(b.value, str):  # two literals: decide now
                v = _FOLD[type(op)](a.value, b.value)
                parts.append(Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True))
                continue
            if isinstance(a.value, str):  # "x" == s would compare char pointers: put the std::string first
                a, b, sym = b, a, _FLIP_CMP[sym]
            parts.append(Expr(f"{wrap(a, p)} {sym} {wrap(b, p - 1)}", p, BOOL))
        if len(parts) == 1:
            return parts[0]
        return Expr(" && ".join(wrap(x, P_AND) for x in parts), P_AND, BOOL)

    def _in_literals(self, node):
        """`x in ("a", "b")` / `n not in (1, 2)`: a chain of == against a literal tuple or list."""
        if len(node.ops) != 1 or not isinstance(node.comparators[0], (ast.Tuple, ast.List, ast.Set)):
            raise Unsupported(node, "`in` is supported against a literal tuple or list: x in (a, b, c)")
        x = self.expr(node.left)
        if not x.simple:
            raise Unsupported(node.left, "the value tested with `in` must be a variable or constant",
                              "store it in a variable first")
        negate = isinstance(node.ops[0], ast.NotIn)
        if not node.comparators[0].elts:
            v = negate
            return Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True)
        tests = []
        for e in node.comparators[0].elts:
            cmp = ast.copy_location(ast.Compare(left=node.left, ops=[ast.Eq()], comparators=[e]), node)
            tests.append(self.expr(cmp))
        if all(isinstance(t.value, bool) for t in tests):
            v = any(t.value for t in tests) != negate
            return Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True)
        body = " || ".join(wrap(t, P_AND - 1) for t in tests)
        if negate:
            return Expr(f"!({body})", P_UNARY, BOOL)
        return Expr(body, P_OR, BOOL)

    def _is_none(self, node):
        """`x is None` / `x is not None`. Only comparisons with None are supported: `is` between
        numbers asks whether two objects are the same, which C++ values don't have."""
        if len(node.ops) != 1:
            raise Unsupported(node, "chained `is` comparisons are not supported")
        a, b = node.left, node.comparators[0]
        none = lambda n: isinstance(n, ast.Constant) and n.value is None  # noqa: E731
        if not (none(a) or none(b)):
            raise Unsupported(node, "`is` is supported only as `x is None` / `x is not None`",
                              "compare values with == or !=")
        e = self.expr_raw(b if none(a) else a)
        negate = isinstance(node.ops[0], ast.IsNot)
        if e.type == NONE:
            v = not negate
            return Expr("true" if v else "false", P_ATOM, BOOL, value=v, simple=True)
        if not isinstance(e.type, OptionalType):  # a number or array is never None
            return Expr("false" if not negate else "true", P_ATOM, BOOL, value=negate, simple=True)
        return Expr(f"{wrap(e, P_ATOM)}.has_value()" if negate else f"!{wrap(e, P_ATOM)}.has_value()",
                    P_ATOM if negate else P_UNARY, BOOL)

    def e_IfExp(self, node):
        c = self.cond(node.test)
        b, o = self.expr_raw(node.body), self.expr_raw(node.orelse)
        t = join(b.type, o.type)
        if t is None or t == VOID or t == NONE:
            raise Unsupported(node, f"the two branches of `x if c else y` have types {b.type} and {o.type}")
        if isinstance(t, OptionalType):
            b, o = self.to_optional(b, t), self.to_optional(o, t)
            if is_array(t.inner):  # two different expression types: make both the stored type
                b, o = Expr(f"{t.cpp()}({b.code})", P_ATOM, t), Expr(f"{t.cpp()}({o.code})", P_ATOM, t)
            else:
                b = Expr(f"{t.cpp()}({b.code})", P_ATOM, t) if b.code == "std::nullopt" else b
                o = Expr(f"{t.cpp()}({o.code})", P_ATOM, t) if o.code == "std::nullopt" else o
            return Expr(f"{wrap(c, P_COND - 1)} ? {wrap(b, P_COND - 1)} : {wrap(o, P_COND)}", P_COND, t)
        if isinstance(t, Scalar):
            b, o = self.as_type(b, t), self.as_type(o, t)
        if isinstance(t, MaskType):
            raise Unsupported(node, "a conditional choosing between boolean arrays is not supported")
        dims = iv = tok = None
        if is_array(t):  # two different Eigen expression types: make both concrete arrays
            b, o = self._plain(b), self._plain(o)
            dims = tuple(x if same_dim(x, y) else None for x, y in zip(dims_of(b), dims_of(o)))
        elif t == INT and same_dim(int_token(b), int_token(o)):
            tok = int_token(b)
        return Expr(f"{wrap(c, P_COND - 1)} ? {wrap(b, P_COND - 1)} : {wrap(o, P_COND)}", P_COND, t,
                    dims=dims, iv=iv, dimtok=tok)

    def e_Tuple(self, node):
        parts = [self.expr(e) for e in node.elts]
        for e, p in zip(node.elts, parts):
            if not is_array(p.type) and p.type != STR:
                self.need_scalar(p, e, "a tuple element")
        parts = [self.as_type(p, p.type) for p in parts]
        return Expr(f"std::make_tuple({', '.join(p.code for p in parts)})", P_ATOM,
                    TupleType(tuple(p.type for p in parts)))

    # --- calls -----------------------------------------------------------------------------
    def e_Call(self, node):
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise Unsupported(node, "*args / **kwargs in calls are not supported")
        f = node.func
        made = self.struct_call(node)
        if made is not None:
            return made
        if isinstance(f, ast.Name):
            name = f.id
            if self.is_variable(name):
                raise Unsupported(node, f"'{name}' is a variable, not a function")
            if name in self.mod.sigs or (name in self.mod.rejected and name in self.mod.imported):
                return self.call_user(node, name)
            if name in self.mod.imports:
                return self.call_library(node, self.mod.imports[name])
            if name in _BUILTINS:
                return self.call_builtin(node, name)
            if name == "len" and len(node.args) == 1 and not node.keywords:
                a = self.expr(node.args[0])
                if is_array(a.type):
                    return Expr(f"{wrap(a, P_ATOM)}.{'size' if a.type.rank == 1 else 'rows'}()", P_ATOM, INT,
                                iv=SIZE_IV, dimtok=dims_of(a)[0])
            if name in ("sum", "max", "min") and node.args:
                a = self.expr(node.args[0])
                if is_array(a.type):
                    raise Unsupported(node, f"built-in {name}() of an array is not supported",
                                      f"for arrays use the numpy function np.{name}(...): built-in sum() "
                                      "adds left to right, np.sum pairwise, so their results differ")
            if name in _REJECTED_BUILTINS:
                raise Unsupported(node, f"{name}() is not supported", _REJECTED_BUILTINS[name])
            if name in self.mod.skipped_globals:
                raise Unsupported(node, f"'{name}' was not translated: {self.mod.skipped_globals[name]}")
            raise Unsupported(node, f"unknown function '{name}'",
                              "define it in this file, or import it from another module of the project (from m import f)")
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
            if name in self.mod.imported:  # from another module of the project: say why
                raise Unsupported(node, f"calls {name}(): {self.mod.rejected[name].message}")
            raise Unsupported(node, f"calls {name}(), which could not be translated")
        sig = self.mod.sigs[name]
        return Expr(f"{sig.cpp}({', '.join(self.bind_args(node, sig, name))})", P_ATOM, sig.ret)

    def bind_args(self, node, sig, display):
        """C++ arguments for a call to a translated function or method, in parameter order,
        converted to the parameter types, with defaults spelled out where a later argument
        is passed by keyword."""
        params = sig.params
        if len(node.args) > len(params):
            raise Unsupported(node, f"{display}() takes {len(params)} arguments but {len(node.args)} were given")
        bound = {}
        for p, a in zip(params, node.args):
            bound[p.name] = a
        for k in node.keywords:
            if k.arg not in {p.name for p in params}:
                raise Unsupported(node, f"{display}() has no parameter '{k.arg}'")
            if k.arg in bound:
                raise Unsupported(node, f"{display}() got two values for '{k.arg}'")
            bound[k.arg] = k.value
        last = max((i for i, p in enumerate(params) if p.name in bound), default=-1)
        codes = []
        passed = []
        for i, p in enumerate(params):
            if p.name not in bound:
                if p.default_code is None:
                    raise Unsupported(node, f"{display}() is missing argument '{p.name}'")
                if i < last:  # a later argument was passed by keyword: spell the default out
                    codes.append(p.default_code)
                continue
            a = bound[p.name]
            codes.append(self.argument(a, p, display, passed))
        if any(getattr(p, "mode", "") == "ref" for p in params) and len(set(passed)) < len(passed):
            raise Unsupported(node, f"the same array or object is passed twice to {display}(), which modifies "
                                    "an argument in place: the two parameters would share memory in Python",
                              "pass a copy for one of them")
        return codes

    def argument(self, a, p, display, passed) -> str:
        """One argument `a` for parameter p, as C++."""
        if p.type == RNG:
            e = self.expr(a)
            if e.type != RNG or not isinstance(a, ast.Name):
                raise Unsupported(a, f"argument '{p.name}' of {display}() must be a generator variable")
            return e.code
        if isinstance(p.type, OptionalType):
            e = self.expr_raw(a)
            if not assignable(p.type, e.type):
                raise Unsupported(a, f"argument '{p.name}' of {display}() is declared {p.type} but a {e.type} is passed")
            return self.to_optional(e, p.type).code
        if is_array(p.type) or isinstance(p.type, StructType) or p.type == STR:
            e = self.expr(a)
            if e.type != p.type:
                raise Unsupported(a, f"argument '{p.name}' of {display}() must be a {p.type}, got {e.type}")
            if getattr(p, "mode", "") == "ref":
                if self._is_view(a):
                    raise Unsupported(a, f"{display}() modifies its argument '{p.name}' in place, and "
                                         f"`{ast.unparse(a)}` is a view: in Python the change goes through it "
                                         "into the original array, which the C++ doesn't do",
                                      "pass a variable, or a copy if the change shouldn't reach the original")
                if not _is_lvalue(a):
                    # a fresh value (x + 1, np.zeros(n), f(x), x.copy()): Python changes the
                    # temporary and drops it; qt::lvalue gives C++ the same temporary to change
                    return f"qt::lvalue({p.type.cpp()}({e.code}))"
            if isinstance(a, ast.Name) and p.type != STR:
                passed.append(a.id)
            return e.code
        e = self.expr(a)
        self.need_scalar(e, a, f"argument '{p.name}' of {display}()")
        if not assignable(p.type, e.type):
            raise Unsupported(a, f"argument '{p.name}' of {display}() is declared {p.type} but a {e.type} is passed",
                              f"annotate '{p.name}' as float, or convert with int(...)")
        return self.promote(e, p.type).code

    def _is_view(self, a) -> bool:
        """An array expression NumPy returns as a view of another array: a slice or row
        (v[1:], A[i], A[:, j]), a transpose (A.T), np.asarray(x) (x itself), x.ravel()."""
        if isinstance(a, ast.Subscript):
            return True
        if isinstance(a, ast.Attribute) and a.attr == "T":
            return True
        if isinstance(a, ast.Call):
            q = self.qualify(a.func)
            if q in ("numpy.asarray", "numpy.ravel", "numpy.transpose", "numpy.reshape"):
                return True
            if isinstance(a.func, ast.Attribute) and a.func.attr in ("ravel", "reshape", "transpose", "view"):
                return True
        return False

    def _arity(self, node, args, lo, hi=None):
        hi = lo if hi is None else hi
        if not lo <= len(args) <= hi:
            want = str(lo) if lo == hi else f"{lo} to {hi}"
            raise Unsupported(node, f"{ast.unparse(node.func)}() takes {want} arguments here, got {len(args)}")

    def absolute(self, e: Expr) -> Expr:
        if e.type == FLOAT:
            return Expr(f"std::fabs({e.code})", P_ATOM, FLOAT)
        lo, hi = iv_of(e)
        r = (0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
        if iv_fits(r):
            return Expr(f"std::abs({self.as_type(e, INT).code})", P_ATOM, INT, iv=r)
        return Expr(f"qt::iabs({self.as_type(e, INT).code})", P_ATOM, INT, iv=iv_clip(r))  # abs(-2**63)

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
        if qual == "numpy.clip":  # a bound may be None, which is not a value
            if not node.args:
                raise Unsupported(node, "np.clip() needs an array or number to clip")
            return self._clip(node, self.expr(node.args[0]), node.args[1:],
                              {k.arg: k.value for k in node.keywords})
        if qual in STACKERS:
            return self.stack_call(node, qual)
        if qual in ("copy.deepcopy", "copy.copy"):
            if len(node.args) != 1 or node.keywords:
                raise Unsupported(node, f"{src}(x) takes one argument here")
            e = self.expr(node.args[0])
            if qual == "copy.copy" and (isinstance(e.type, StructType) or is_array(e.type)):
                raise Unsupported(node, "copy.copy is a shallow copy: arrays inside stay shared",
                                  "use copy.deepcopy (an independent copy, which is what C++ makes)")
            if not (isinstance(e.type, StructType) or is_array(e.type)):
                raise Unsupported(node, f"{src}() of a {e.type} is not needed")
            return Expr(f"{e.type.cpp()}({e.code})", P_ATOM, e.type)  # a C++ copy is a deep copy
        if qual == "numpy.einsum":
            return self.einsum_call(node)
        args = [self.expr(a) for a in node.args]
        if qual == "numpy.where" and len(args) == 3 and not node.keywords and \
                not any(is_array(a.type) or isinstance(a.type, MaskType) for a in args):
            for a, n in zip(args, node.args):
                self.need_scalar(a, n, "np.where()")
            t = join(args[1].type, args[2].type)
            t = INT if t == BOOL else t
            # NumPy evaluates both branches before choosing (no short circuit), so qt::where
            # takes all three as arguments rather than becoming `c ? x : y`
            return Expr(f"qt::where({self.truth(args[0], node).code}, {self.as_type(args[1], t).code}, "
                        f"{self.as_type(args[2], t).code})", P_ATOM, t)
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
                (a0, a1), (b0, b1) = iv_of(a), iv_of(b)
                r = (max(a0, b0), max(a1, b1)) if fn == "max" else (min(a0, b0), min(a1, b1))
                return Expr(f"std::{fn}({self.as_type(a, INT).code}, {self.as_type(b, INT).code})", P_ATOM, INT,
                            iv=r)
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
