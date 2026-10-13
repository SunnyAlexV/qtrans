"""int64 arrays (0.14): np.arange of ints, rng.integers(..., size=n), index arrays.

    idx = rng.integers(0, n, size=n)      qt::IVec idx = rng.integers(0, n, n);
    sample = x[idx]                       qt::Vec sample = qt::take(x, idx);
    k = np.arange(1, n + 1)               qt::IVec k = qt::iarange(1, n + 1);
    t = k * dt                            qt::Vec t = k.cast<double>() * dt;

An int array keeps NumPy's int64 rules: +, -, * and sum wrap around at 2**64 without an error;
// and % follow Python's signs and give 0 for a zero divisor (NumPy warns); `/`, a float
operand and the float functions (np.exp, np.mean, np.std, ...) turn it into float64 first, as
NumPy does. It prints as NumPy prints int64 arrays ([ 0  1 10]). Only the operations written
here accept one: anything else is refused, so an int array is never treated as a float array
by accident.
"""
from __future__ import annotations

import ast

from .core import P_ATOM, P_REL, Expr, wrap
from .errors import Unsupported
from .types import BOOL, FLOAT, INT, IVEC, VEC, MaskType, is_array, is_iarray

# NumPy functions whose result is float64 for an int array: the int array is converted first
FLOAT_FUNCS = {
    "numpy.exp", "numpy.log", "numpy.log2", "numpy.log10", "numpy.log1p", "numpy.expm1", "numpy.sqrt",
    "numpy.sin", "numpy.cos", "numpy.tan", "numpy.arcsin", "numpy.arccos", "numpy.arctan", "numpy.sinh",
    "numpy.cosh", "numpy.tanh", "numpy.mean", "numpy.std", "numpy.var", "numpy.median", "numpy.quantile",
    "numpy.percentile", "numpy.average", "numpy.cbrt", "numpy.reciprocal", "numpy.float64",
}
_ARITH = {ast.Add: "iadd", ast.Sub: "isub", ast.Mult: "imul", ast.FloorDiv: "ifloordiv", ast.Mod: "imod"}


class IntArrayOps:
    """Mixin for ExprTranslator."""

    def as_float_array(self, e: Expr) -> Expr:
        """An int array as float64 (NumPy's conversion: exact up to 2**53, rounded beyond)."""
        if not is_iarray(e.type):
            return e
        return Expr(f"{wrap(e, P_ATOM)}.cast<double>()", P_ATOM, VEC)

    def int_array_binop(self, node, op, L: Expr, R: Expr) -> Expr:
        floaty = isinstance(op, ast.Div) or any(x.type == FLOAT or is_array(x.type) for x in (L, R))
        if isinstance(op, ast.Pow):
            if floaty:
                return self.array_binop(node, op, self.as_float_array(L), self.as_float_array(R))
            for x in (L, R):
                if not (is_iarray(x.type) or x.type in (INT, BOOL)):
                    raise Unsupported(node, f"** of an int array and a {x.type} is not supported")
            a = L if is_iarray(L.type) else self.as_type(L, INT)
            b = R if is_iarray(R.type) else self.as_type(R, INT)
            return Expr(f"qt::ipow({a.code}, {b.code})", P_ATOM, IVEC)
        if isinstance(op, ast.MatMult):
            raise Unsupported(node, "@ with an int array is not supported yet", "convert it: idx.astype(float)")
        if floaty:  # int64 with float64, or true division: float64, as NumPy does
            return self.array_binop(node, op, self.as_float_array(L), self.as_float_array(R))
        for x in (L, R):
            if not (is_iarray(x.type) or x.type in (INT, BOOL)):
                raise Unsupported(node, f"an int array with a {x.type} is not supported")
        if type(op) not in _ARITH:
            raise Unsupported(node, f"{type(op).__name__} on int arrays is not supported")
        fn = _ARITH[type(op)]
        a = L if is_iarray(L.type) else self.as_type(L, INT)
        b = R if is_iarray(R.type) else self.as_type(R, INT)
        if not is_iarray(L.type) and fn in ("iadd", "imul"):
            a, b = b, a  # (k + idx == idx + k)
        return Expr(f"qt::{fn}({a.code}, {b.code})", P_ATOM, IVEC)

    def int_array_unary(self, node, e: Expr) -> Expr:
        if isinstance(node.op, ast.USub):
            return Expr(f"qt::ineg({e.code})", P_ATOM, IVEC)
        if isinstance(node.op, ast.UAdd):
            return e
        raise Unsupported(node, f"{type(node.op).__name__} on an int array is not supported")

    def int_array_compare(self, node, L: Expr, op, R: Expr) -> Expr:
        if any(x.type == FLOAT or is_array(x.type) for x in (L, R)):
            return self.array_compare(node, self.as_float_array(L), op, self.as_float_array(R))
        sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}.get(type(op))
        if sym is None:
            raise Unsupported(node, f"{type(op).__name__} on int arrays is not supported")
        if not all(is_iarray(x.type) or x.type in (INT, BOOL) for x in (L, R)):
            raise Unsupported(node, "an int array compared with this is not supported")
        a = L if is_iarray(L.type) else self.as_type(L, INT)
        b = R if is_iarray(R.type) else self.as_type(R, INT)
        if is_iarray(L.type) and is_iarray(R.type):
            fn = {"<": "less", "<=": "less_equal", ">": "greater", ">=": "greater_equal", "==": "equal_to",
                  "!=": "not_equal_to"}[sym]
            return Expr(f"qt::icmp({a.code}, {b.code}, std::{fn}<long long>())", P_ATOM, MaskType(1))
        return Expr(f"qt::Mask({wrap(a, P_REL)} {sym} {wrap(b, P_REL)})", P_ATOM, MaskType(1))

    def int_array_attribute(self, node, B: Expr) -> Expr:
        if node.attr == "size":
            return Expr(f"static_cast<long long>({wrap(B, P_ATOM)}.size())", P_ATOM, INT)
        if node.attr == "ndim":
            return Expr("1LL", P_ATOM, INT, value=1)
        raise Unsupported(node, f"int array attribute .{node.attr} is not supported" +
                          (" (use len(a) for the length)" if node.attr == "shape" else ""))

    def int_array_method(self, node, B: Expr, name: str) -> Expr:
        args = node.args
        if name == "astype":
            if len(args) != 1 or node.keywords:
                raise Unsupported(node, "astype(float) or astype(int) is supported")
            k = self._dtype_kind(args[0])
            if k == "float":
                return Expr(f"qt::astype_float({B.code})", P_ATOM, VEC)
            if k == "int":
                return Expr(f"qt::IVec({B.code})", P_ATOM, IVEC)
            raise Unsupported(node, "astype(float) or astype(int) is supported")
        if node.keywords or args:
            raise Unsupported(node, f"int array .{name}() with arguments is not supported")
        return self.int_array_reduce(node, name, B)

    def int_array_reduce(self, node, name, B: Expr) -> Expr:
        if name == "sum":
            return Expr(f"qt::isum({B.code})", P_ATOM, INT)
        if name == "prod":
            return Expr(f"qt::iprod({B.code})", P_ATOM, INT)
        if name in ("max", "amax"):
            return Expr(f"qt::imax({B.code})", P_ATOM, INT)
        if name in ("min", "amin"):
            return Expr(f"qt::imin({B.code})", P_ATOM, INT)
        if name == "cumsum":
            return Expr(f"qt::icumsum({B.code})", P_ATOM, IVEC)
        if name == "copy":
            return Expr(f"qt::IVec({B.code})", P_ATOM, IVEC)
        if name in ("mean", "std", "var"):
            F = self.as_float_array(B)
            return self.array_method(ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id="_", ctx=ast.Load()),
                                                                                    attr=name, ctx=ast.Load()),
                                                                args=[], keywords=[]), node), F, name)
        if name in ("argmax", "argmin"):  # (the first position of the largest / smallest, as NumPy)
            return Expr(f"qt::i{name}({B.code})", P_ATOM, INT)
        raise Unsupported(node, f"int array .{name}() is not supported yet")

    def _dtype_kind(self, n):
        src = ast.unparse(n)
        if src in ("float", "np.float64", "numpy.float64"):
            return "float"
        if src in ("int", "np.int64", "numpy.int64"):
            return "int"
        return None

    def int_array_call(self, node, qual, args):
        """np.<f>(...) where an argument is an int array (None if f isn't one handled here)."""
        name = qual.split(".")[-1]
        if qual in FLOAT_FUNCS:
            return None  # the caller converts the int arrays and goes on as for floats
        if qual in ("numpy.sum", "numpy.prod", "numpy.max", "numpy.min", "numpy.amax", "numpy.amin",
                    "numpy.cumsum", "numpy.copy", "numpy.argmax", "numpy.argmin"):
            if len(args) != 1 or node.keywords:
                raise Unsupported(node, f"np.{name}() of an int array with more arguments is not supported")
            return self.int_array_reduce(node, name, args[0])
        if qual in ("numpy.abs", "numpy.absolute"):
            return Expr(f"qt::iabs({args[0].code})", P_ATOM, IVEC)
        if qual in ("numpy.array", "numpy.asarray"):
            k = None
            for kw in node.keywords:
                if kw.arg != "dtype":
                    raise Unsupported(node, f"np.{name}(..., {kw.arg}=) is not supported")
                k = self._dtype_kind(kw.value)
                if k is None:
                    raise Unsupported(node, "dtype must be float or int here")
            if k == "float":
                return Expr(f"qt::astype_float({args[0].code})", P_ATOM, VEC)
            return Expr(f"qt::IVec({args[0].code})", P_ATOM, IVEC)  # np.array copies; asarray of int64: the same
        if qual == "numpy.sort":
            return Expr(f"qt::isorted({args[0].code})", P_ATOM, IVEC)
        raise Unsupported(node, f"np.{name}() of an int array is not supported yet",
                          "convert it first, if a float result is what you want: x.astype(float)")
