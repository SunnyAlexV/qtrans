"""Array expressions (stage 2): NumPy semantics on Eigen arrays.

Mixed into ExprTranslator. Everything here produces C++ in terms of qt::Vec / qt::Mat
(Eigen arrays, elementwise by default like NumPy) and the helpers in qtrans_arrays.hpp.

Rules that keep the numbers identical to NumPy's:
  * + - * / on arrays, sqrt and abs are exact in both, so they map straight to Eigen.
  * sum, mean, var/std, cumsum, quantile, linspace, arange use ports of NumPy's own algorithms.
  * x ** 2, x ** 0.5, x ** -1 use the same fast paths NumPy takes (square, sqrt, reciprocal).
  * Python index and slice rules (negative indices, clipping) are applied exactly.
  * Broadcasting follows NumPy: scalar with array, a Mat with a Vec whose length equals the
    number of columns (the trailing-axis rule), and two Mats whose dimensions are equal or 1
    (a keepdims=True mean, or v[:, None], against a matrix).
"""
from __future__ import annotations

import ast

from .core import P_ADD, P_ATOM, P_EQ, P_MUL, P_REL, P_UNARY, Expr, wrap
from .errors import Unsupported
from .types import BOOL, FLOAT, INT, MAT, RNG, VEC, ArrayType, MaskType, Scalar, TupleType, is_array

_ARITH = {ast.Add: ("+", P_ADD), ast.Sub: ("-", P_ADD), ast.Mult: ("*", P_MUL), ast.Div: ("/", P_MUL)}
_CMP = {ast.Lt: ("<", P_REL), ast.LtE: ("<=", P_REL), ast.Gt: (">", P_REL), ast.GtE: (">=", P_REL),
        ast.Eq: ("==", P_EQ), ast.NotEq: ("!=", P_EQ)}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}

# elementwise functions: Eigen member (exact, same as NumPy) or qt:: helper (C library)
_MEMBER = {"numpy.sqrt": "sqrt", "numpy.abs": "abs", "numpy.absolute": "abs", "numpy.fabs": "abs",
           "numpy.square": "square"}
_HELPER = {f"numpy.{n}": f"qt::{n}" for n in (
    "exp", "log", "log10", "log2", "log1p", "expm1", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "floor", "ceil", "trunc", "rint")}
_HELPER["numpy.sign"] = "qt::np_sign"
_MASK_FN = {"numpy.isnan": "isNaN", "numpy.isfinite": "isFinite", "numpy.isinf": "isInf"}
_REDUCE = {"numpy.sum": "sum", "numpy.mean": "mean", "numpy.std": "std", "numpy.var": "var",
           "numpy.max": "max", "numpy.amax": "max", "numpy.min": "min", "numpy.amin": "min",
           "numpy.argmax": "argmax", "numpy.argmin": "argmin", "numpy.cumsum": "cumsum",
           "numpy.cumprod": "cumprod", "numpy.sort": "sort", "numpy.quantile": "quantile",
           "numpy.percentile": "percentile", "numpy.median": "median"}
ARRAY_FUNCS = set(_MEMBER) | set(_HELPER) | set(_MASK_FN) | set(_REDUCE) | {
    "numpy.zeros", "numpy.ones", "numpy.full", "numpy.eye", "numpy.identity", "numpy.linspace",
    "numpy.arange", "numpy.array", "numpy.asarray", "numpy.zeros_like", "numpy.ones_like",
    "numpy.full_like", "numpy.copy", "numpy.dot", "numpy.matmul", "numpy.outer", "numpy.transpose",
    "numpy.diag", "numpy.where", "numpy.all", "numpy.any", "numpy.count_nonzero", "numpy.maximum",
    "numpy.minimum", "numpy.power", "numpy.linalg.solve", "numpy.linalg.inv", "numpy.linalg.cholesky",
    "numpy.linalg.det", "numpy.clip", "numpy.trace", "numpy.ravel", "numpy.linalg.eigh",
    "numpy.linalg.eigvalsh", "numpy.linalg.pinv"}
# functions taking a list of arrays: np.column_stack([a, b]) ...
STACKERS = {"numpy.column_stack", "numpy.vstack", "numpy.row_stack", "numpy.hstack", "numpy.concatenate"}
_BC_OPS = {"+": "qt::add_op{}", "-": "qt::sub_op{}", "*": "qt::mul_op{}", "/": "qt::div_op{}"}
# functions that make arrays from scalars: always take the array path
_CREATORS = {"numpy.zeros", "numpy.ones", "numpy.full", "numpy.eye", "numpy.identity", "numpy.linspace",
             "numpy.arange", "numpy.array", "numpy.asarray"}


# Generator methods: name -> scalar parameters (with NumPy's defaults), then `size`
_RNG_METHODS = {"random": [], "standard_normal": [], "normal": [("loc", "0.0"), ("scale", "1.0")],
                "uniform": [("low", "0.0"), ("high", "1.0")]}


class ArrayOps:
    """Mixin for ExprTranslator: everything that involves arrays."""

    # --- small helpers -------------------------------------------------------------------
    def lower_bound(self, node):
        """A proven lower bound for an int expression, or None. Used to emit `a(i)` instead of
        `a(qt::idx(i, n))` when Python's negative-index wrap-around can't happen."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None

    def _scalar_operand(self, e: Expr, node) -> Expr:
        self.need_scalar(e, node, "array arithmetic")
        return self.as_double(e)  # Eigen needs double scalars: 2 -> 2.0, n -> static_cast<double>(n)

    def _int_code(self, node) -> str:
        e = self.expr(node)
        if e.type not in (INT, BOOL):
            raise Unsupported(node, "array indices and slice bounds must be integers")
        return e.code

    def _need_array(self, e: Expr, node, what):
        if not is_array(e.type):
            raise Unsupported(node, f"{what} needs an array (Vec or Mat), got {e.type}")

    def _plain(self, e: Expr) -> Expr:
        """An Eigen expression as a concrete array, where C++ needs both sides of ?: to match."""
        return Expr(f"{e.type.cpp()}({e.code})", P_ATOM, e.type)

    # --- arithmetic ------------------------------------------------------------------------
    def array_binop(self, node, op, L: Expr, R: Expr) -> Expr:
        node = node or self.current
        if isinstance(L.type, MaskType) or isinstance(R.type, MaskType):
            if isinstance(op, (ast.BitAnd, ast.BitOr)) and L.type == R.type:
                sym = "&&" if isinstance(op, ast.BitAnd) else "||"
                return Expr(f"{wrap(L, P_ATOM)} {sym} {wrap(R, P_ATOM)}", P_REL, L.type)
            raise Unsupported(node, "only & and | combine boolean arrays", "wrap each comparison in parentheses")
        if isinstance(op, ast.MatMult):
            return self.matmul(node, L, R)
        if isinstance(op, ast.Pow):
            return self.array_power(node, L, R)
        if type(op) not in _ARITH:
            raise Unsupported(node, f"the {type(op).__name__} operator is not supported on arrays",
                              "// and % on arrays are not in stage 2")
        sym, p = _ARITH[type(op)]
        LA, RA = is_array(L.type), is_array(R.type)
        if not LA:
            L = self._scalar_operand(L, node)
        if not RA:
            R = self._scalar_operand(R, node)
        if L.type == MAT and R.type == MAT:
            # either side may be (n, 1) or (1, m): NumPy broadcasts, Eigen would refuse
            return Expr(f"qt::broadcast({L.code}, {R.code}, {_BC_OPS[sym]})", P_ATOM, MAT)
        if not (LA and RA) or L.type == R.type:
            t = L.type if LA else R.type
            return Expr(f"{wrap(L, p)} {sym} {wrap(R, p - 1)}", p, t)
        # NumPy broadcasting of a Vec across the rows of a Mat (vector length = number of columns)
        if L.type == MAT and R.type == VEC:
            return Expr(f"{wrap(L, P_ATOM)}.rowwise() {sym} {wrap(R, P_ATOM)}.transpose()", P_ADD, MAT)
        if sym in ("+", "*"):  # Vec + Mat == Mat + Vec exactly (IEEE + and * commute)
            return Expr(f"{wrap(R, P_ATOM)}.rowwise() {sym} {wrap(L, P_ATOM)}.transpose()", P_ADD, MAT)
        return Expr(f"{wrap(L, P_ATOM)}.transpose().replicate({wrap(R, P_ATOM)}.rows(), 1) {sym} "
                    f"{wrap(R, p - 1)}", p, MAT)

    def array_power(self, node, L: Expr, R: Expr) -> Expr:
        if is_array(L.type) and not is_array(R.type):
            self.need_scalar(R, node, "**")
            v = R.value
            X = wrap(L, P_ATOM)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                # the fast paths NumPy itself takes for these exponents
                if v == 2:
                    return Expr(f"{X}.square()", P_ATOM, L.type)
                if v == 0.5:
                    return Expr(f"{X}.sqrt()", P_ATOM, L.type)
                if v == -1:
                    return Expr(f"{X}.inverse()", P_ATOM, L.type)
            return Expr(f"qt::pow({L.code}, {self.as_double(R).code})", P_ATOM, L.type)
        if not is_array(L.type) and is_array(R.type):
            self.need_scalar(L, node, "**")
            return Expr(f"qt::pow({self.as_double(L).code}, {R.code})", P_ATOM, R.type)
        if L.type != R.type:
            raise Unsupported(node, "** between a Vec and a Mat is not supported")
        if L.type == MAT:
            return Expr(f"qt::broadcast({L.code}, {R.code}, qt::pow_op{{}})", P_ATOM, MAT)
        return Expr(f"qt::pow({L.code}, {R.code})", P_ATOM, L.type)

    def matmul(self, node, L: Expr, R: Expr) -> Expr:
        if not (is_array(L.type) and is_array(R.type)):
            raise Unsupported(node, "@ needs an array on both sides")
        fn, t = {(1, 1): ("dot", FLOAT), (2, 1): ("matvec", VEC), (1, 2): ("vecmat", VEC),
                 (2, 2): ("matmul", MAT)}[(L.type.rank, R.type.rank)]
        return Expr(f"qt::{fn}({L.code}, {R.code})", P_ATOM, t)

    def array_compare(self, node, L: Expr, op, R: Expr) -> Expr:
        if type(op) not in _CMP:
            raise Unsupported(node, f"`{type(op).__name__}` is not supported on arrays")
        sym, p = _CMP[type(op)]
        LA, RA = is_array(L.type), is_array(R.type)
        if LA and RA:
            if L.type != R.type:
                raise Unsupported(node, "comparing a Vec with a Mat is not supported")
            return Expr(f"{wrap(L, P_ATOM)} {sym} {wrap(R, P_ATOM)}", p, MaskType(L.type.rank))
        if not LA:  # 0.0 < a  ->  a > 0.0
            L, R, sym = R, L, _FLIP[sym]
        R = self._scalar_operand(R, node)
        return Expr(f"{wrap(L, P_ATOM)} {sym} {wrap(R, p - 1)}", p, MaskType(L.type.rank))

    # --- indexing ------------------------------------------------------------------------------
    def _index(self, node, size_code: str, named: bool) -> str:
        e = self.expr(node)
        if e.type not in (INT, BOOL):
            raise Unsupported(node, "array indices must be integers")
        if isinstance(e.value, int) and not isinstance(e.value, bool):
            if e.value >= 0:
                return str(e.value)
            if not named:
                raise Unsupported(node, "negative index into an expression", "store the array in a variable first")
            return f"{size_code} - {-e.value}"
        lb = self.lower_bound(node)
        if lb is not None and lb >= 0:
            return e.code  # provably >= 0: Python's wrap-around can't apply
        if not named:
            raise Unsupported(node, "index that may be negative, into an expression",
                              "store the array in a variable first")
        return f"qt::idx({e.code}, {size_code})"

    def _bound(self, node) -> str:
        return "qt::none" if node is None else self._int_code(node)

    def _slice(self, base: str, s: ast.Slice) -> str:
        code = f"qt::slice({base}, {self._bound(s.lower)}, {self._bound(s.upper)}"
        if s.step is not None:
            code += f", {self._int_code(s.step)}"
        return code + ")"

    def _slice2d(self, base: str, rs: ast.Slice, cs: ast.Slice, node) -> str:
        for s in (rs, cs):
            if s.step is not None:
                raise Unsupported(node, "2-D slices with a step are not supported")
        return (f"qt::block({base}, {self._bound(rs.lower)}, {self._bound(rs.upper)}, "
                f"{self._bound(cs.lower)}, {self._bound(cs.upper)})")

    @staticmethod
    def _full(s) -> bool:
        return isinstance(s, ast.Slice) and s.lower is None and s.upper is None and s.step is None

    def array_subscript(self, node, target=False):
        """Read (target=False) or write (target=True) access a[...]. For writes returns
        (lvalue code, kind) where kind is 'scalar', 'vec', 'row' or 'mat'."""
        base_node = node.value
        if isinstance(base_node, ast.Attribute) and base_node.attr == "shape" and not target:
            return self._shape_item(node)
        B = self.expr(base_node)
        if isinstance(B.type, TupleType) and not target:
            k = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if not isinstance(k, int) or not -len(B.type.elts) <= k < len(B.type.elts):
                raise Unsupported(node, "tuples can only be indexed with a constant in range")
            k %= len(B.type.elts)
            return Expr(f"std::get<{k}>({B.code})", P_ATOM, B.type.elts[k])
        if not is_array(B.type):
            raise Unsupported(node, "indexing / subscripts are only supported on arrays",
                              "lists and dicts are not supported")
        named = isinstance(base_node, ast.Name)
        if target and not named:
            raise Unsupported(node, "assignment into an expression; assign into a variable")
        parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        b = wrap(B, P_ATOM)
        if len(parts) == 1 and not isinstance(parts[0], ast.Slice):
            M = self.expr(parts[0])
            if isinstance(M, Expr) and isinstance(M.type, MaskType):
                return self._mask_subscript(node, B, M, target)
        if B.type == VEC and len(parts) == 2 and any(self._is_newaxis(q) for q in parts):
            if target:
                raise Unsupported(node, "assigning through a new axis (v[:, None] = ...) is not supported")
            other = parts[1] if self._is_newaxis(parts[0]) else parts[0]
            if not self._full(other):
                raise Unsupported(node, "a new axis is supported as v[:, None] or v[None, :]")
            fn = "as_row" if self._is_newaxis(parts[0]) else "as_col"
            return Expr(f"qt::{fn}({B.code})", P_ATOM, MAT)
        slices = [isinstance(p, ast.Slice) for p in parts]
        if any(slices) and not named:
            raise Unsupported(node, "slicing an expression", "store the array in a variable, then slice it")

        if B.type == VEC:
            if len(parts) != 1:
                raise Unsupported(node, "too many indices for a 1-D array")
            if slices[0]:
                code = self._slice(B.code, parts[0])
                return (code, "vec") if target else Expr(code, P_ATOM, VEC)
            code = f"{b}({self._index(parts[0], f'{b}.size()', named)})"
            return (code, "scalar") if target else Expr(code, P_ATOM, FLOAT)

        # Mat
        if len(parts) == 1:
            if slices[0]:
                if parts[0].step is not None:
                    raise Unsupported(node, "row slices with a step are not supported")
                code = self._slice2d(B.code, parts[0], ast.Slice(), node)
                return (code, "mat") if target else Expr(code, P_ATOM, MAT)
            code = f"{b}.row({self._index(parts[0], f'{b}.rows()', named)})"
            return (code, "row") if target else Expr(f"{code}.transpose()", P_ATOM, VEC)
        if len(parts) != 2:
            raise Unsupported(node, "too many indices for a 2-D array")
        r, c = parts
        if not slices[0] and not slices[1]:
            code = f"{b}({self._index(r, f'{b}.rows()', named)}, {self._index(c, f'{b}.cols()', named)})"
            return (code, "scalar") if target else Expr(code, P_ATOM, FLOAT)
        if self._full(r) and not slices[1]:
            code = f"{b}.col({self._index(c, f'{b}.cols()', named)})"
            return (code, "vec") if target else Expr(code, P_ATOM, VEC)
        if not slices[0] and self._full(c):
            code = f"{b}.row({self._index(r, f'{b}.rows()', named)})"
            return (code, "row") if target else Expr(f"{code}.transpose()", P_ATOM, VEC)
        if slices[0] and slices[1]:
            code = self._slice2d(B.code, r, c, node)
            return (code, "mat") if target else Expr(code, P_ATOM, MAT)
        raise Unsupported(node, "mixing an index and a partial slice (A[i, 1:3]) is not supported",
                          "slice the row first: row = A[i]; row[1:3]")

    def _is_newaxis(self, n) -> bool:
        return (isinstance(n, ast.Constant) and n.value is None) or self.qualify(n) == "numpy.newaxis"

    def _mask_subscript(self, node, B: Expr, M: Expr, target):
        """a[mask] (a copy of the selected elements) and a[mask] = scalar."""
        if target:
            if M.type.rank != B.type.rank:
                raise Unsupported(node, "assigning into rows picked by a mask (A[mask] = ...) is not supported")
            return (B.code, M.code), "mask"
        if M.type.rank == B.type.rank:
            return Expr(f"qt::select({B.code}, {M.code})", P_ATOM, VEC)
        if B.type == MAT and M.type.rank == 1:
            return Expr(f"qt::select_rows({B.code}, {M.code})", P_ATOM, MAT)
        raise Unsupported(node, "a 2-D mask can only index a 2-D array")

    def _shape_item(self, node):
        A = self.expr(node.value.value)
        self._need_array(A, node, ".shape")
        k = node.slice.value if isinstance(node.slice, ast.Constant) else None
        a = wrap(A, P_ATOM)
        dims = [f"{a}.size()"] if A.type == VEC else [f"{a}.rows()", f"{a}.cols()"]
        if not isinstance(k, int) or not -len(dims) <= k < len(dims):
            raise Unsupported(node, f".shape of a {A.type} has {len(dims)} entries; index it with a constant")
        return Expr(dims[k], P_ATOM, INT)

    def array_attribute(self, node, B: Expr) -> Expr:
        b = wrap(B, P_ATOM)
        if node.attr == "T":
            return Expr(f"{b}.transpose()", P_ATOM, MAT) if B.type == MAT else B
        if node.attr == "size":
            return Expr(f"{b}.size()", P_ATOM, INT)
        if node.attr == "ndim":
            return Expr(str(B.type.rank), P_ATOM, INT, int_lit=True, value=B.type.rank, simple=True)
        if node.attr == "shape":
            if B.type == VEC:
                return Expr(f"std::make_tuple(static_cast<long long>({b}.size()))", P_ATOM, TupleType((INT,)))
            return Expr(f"std::make_tuple(static_cast<long long>({b}.rows()), static_cast<long long>({b}.cols()))",
                        P_ATOM, TupleType((INT, INT)))
        raise Unsupported(node, f"array attribute .{node.attr} is not supported")

    # --- methods: a.sum(), a.copy(), (a > 0).any(), ... -------------------------------------------
    def array_method(self, node, B: Expr, name: str) -> Expr:
        if name in ("astype", "clip"):  # arguments are a dtype / may be None: not plain values
            args, kws = [], {}
        else:
            args = [self.expr(a) for a in node.args]
            kws = {k.arg: self.expr(k.value) for k in node.keywords}
        if isinstance(B.type, MaskType):
            if name in ("any", "all") and not args and not kws:
                return Expr(f"{wrap(B, P_ATOM)}.{name}()", P_ATOM, BOOL)
            if name == "sum" and not args and not kws:
                return Expr(f"{wrap(B, P_ATOM)}.count()", P_ATOM, INT)
            raise Unsupported(node, f"boolean-array method .{name}() is not supported")
        if name == "copy" and not args:
            return Expr(B.code, B.prec, B.type)
        if name == "astype":
            if len(node.args) != 1 or node.keywords or ast.unparse(node.args[0]) not in (
                    "float", "np.float64", "numpy.float64", "'float64'", '"float64"'):
                raise Unsupported(node, "only .astype(float) is supported (arrays are float64)")
            return Expr(B.code, B.prec, B.type)  # a copy, as astype makes
        if name in ("ravel", "flatten") and not node.args and not node.keywords:
            return Expr(f"qt::ravel({B.code})", P_ATOM, VEC) if B.type == MAT else Expr(B.code, B.prec, VEC)
        if name == "clip":
            return self._clip(node, B, node.args, {k.arg: k.value for k in node.keywords})
        if name == "trace" and not args and not kws:
            if B.type != MAT:
                raise Unsupported(node, ".trace() needs a 2-D array")
            return Expr(f"qt::trace({B.code})", P_ATOM, FLOAT)
        if name == "transpose" and not args:
            return self.array_attribute(ast.Attribute(attr="T"), B)
        if name == "dot" and len(args) == 1:
            return self.matmul(node, B, args[0])

        if name in ("sum", "mean", "std", "var", "max", "min", "argmax", "argmin", "cumsum", "cumprod"):
            return self._reduce(node, name, B, args, kws)
        raise Unsupported(node, f"array method .{name}() is not supported in stage 2")

    def _axis(self, node, args, kws, pos):
        a = args[pos] if len(args) > pos else kws.get("axis")
        if a is None:
            return None
        if not isinstance(a.value, int) or isinstance(a.value, bool):
            raise Unsupported(node, "axis must be an integer constant")
        return a.value

    def _keepdims(self, node, kws) -> bool:
        k = kws.get("keepdims")
        if k is None:
            return False
        if not isinstance(k.value, bool):
            raise Unsupported(node, "keepdims must be the constant True or False")
        return k.value

    def _kept(self, e: Expr, A: Expr, axis, keep: bool) -> Expr:
        """A reduction result with keepdims=True: (1, m) for axis 0, (n, 1) for axis 1."""
        if not keep:
            return e
        if A.type == VEC or axis is None:
            raise Unsupported(self.current, "keepdims=True is supported for reductions of a Mat over an axis")
        fn = "as_row" if axis in (0, -2) else "as_col"
        return Expr(f"qt::{fn}({e.code})", P_ATOM, MAT)

    def _reduce(self, node, name, A: Expr, args, kws) -> Expr:
        self._need_array(A, node, f"{name}()")
        extra = set(kws) - {"axis", "ddof", "keepdims"}
        if extra:
            raise Unsupported(node, f"keyword(s) {', '.join(sorted(extra))} are not supported for {name}()")
        keep = self._keepdims(node, kws)
        if keep and name not in ("sum", "mean", "std", "var"):
            raise Unsupported(node, f"keepdims is not supported for {name}()")
        if name in ("sum", "mean", "std", "var"):
            axis = self._axis(node, args, kws, 0)
            if axis is not None and not (A.type == VEC and axis in (0, -1)) and \
                    not (A.type == MAT and axis in (0, 1, -1, -2)):
                raise Unsupported(node, f"axis={axis} is out of range for a {A.type}")
            if A.type == VEC:
                axis = None
        if name in ("sum", "mean"):
            if axis is None:
                return Expr(f"qt::{name}({A.code})", P_ATOM, FLOAT)
            return self._kept(Expr(f"qt::{name}({A.code}, {axis})", P_ATOM, VEC), A, axis, keep)
        if name in ("std", "var"):
            if len(args) > 1:
                raise Unsupported(node, f"pass ddof to {name}() by keyword")
            ddof = kws.get("ddof")
            if ddof is not None and ddof.type not in (INT, BOOL):
                raise Unsupported(node, "ddof must be an integer")
            fn = "std_" if name == "std" else "var"
            if axis is None:
                return Expr(f"qt::{fn}({A.code}{', ' + ddof.code if ddof is not None else ''})", P_ATOM, FLOAT)
            d = ddof.code if ddof is not None else "0"
            return self._kept(Expr(f"qt::{fn}({A.code}, {axis}, {d})", P_ATOM, VEC), A, axis, keep)
        if name in ("cumsum", "cumprod") and not kws.keys() - {"axis"}:
            axis = self._axis(node, args, kws, 0)
            if axis is not None and len(args) > 1:
                raise Unsupported(node, f"arguments to {name}() beyond axis are not supported")
            if axis is not None and A.type == MAT:
                if axis not in (0, 1, -1, -2):
                    raise Unsupported(node, f"axis={axis} is out of range for a Mat")
                return Expr(f"qt::{name}({A.code}, {axis})", P_ATOM, MAT)
            if axis is not None and axis not in (0, -1):
                raise Unsupported(node, f"axis={axis} is out of range for a Vec")
            return Expr(f"qt::{name}({A.code})", P_ATOM, VEC)
        if args or kws:
            raise Unsupported(node, f"arguments to {name}() are not supported in stage 2")
        t = {"max": FLOAT, "min": FLOAT, "argmax": INT, "argmin": INT, "cumsum": VEC, "cumprod": VEC}[name]
        return Expr(f"qt::{name}({A.code})", P_ATOM, t)

    # --- numpy functions -------------------------------------------------------------------
    def wants_array_call(self, qual, args) -> bool:
        return qual in _CREATORS or (qual in ARRAY_FUNCS and any(
            is_array(a.type) or isinstance(a.type, MaskType) for a in args))

    def _shape_args(self, node, shape_node):
        """np.zeros(n) / np.zeros((r, c)) -> ("n",) or ("r", "c")."""
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            dims = [self._int_code(e) for e in shape_node.elts]
            if len(dims) not in (1, 2):
                raise Unsupported(node, "only 1-D and 2-D arrays are supported")
            return dims
        return [self._int_code(shape_node)]

    def _check_dtype(self, node):
        for k in node.keywords:
            if k.arg == "dtype":
                if ast.unparse(k.value) not in ("float", "np.float64", "numpy.float64"):
                    raise Unsupported(node, "only float64 arrays are supported in stage 2")
            else:
                raise Unsupported(node, f"keyword '{k.arg}' is not supported here")

    def array_literal(self, node, src) -> Expr:
        """np.array([a, b, c]) / np.array([[..], [..]]) / np.array(existing_array)."""
        self._check_dtype(node)
        if len(node.args) != 1:
            raise Unsupported(node, f"{src}() takes one argument here")
        a = node.args[0]
        if isinstance(a, (ast.List, ast.Tuple)):
            if a.elts and all(isinstance(e, (ast.List, ast.Tuple)) for e in a.elts):
                rows = []
                for row in a.elts:
                    rows.append("{" + ", ".join(self._scalar_operand(self.expr(e), e).code for e in row.elts) + "}")
                return Expr(f"qt::mat({{{', '.join(rows)}}})", P_ATOM, MAT)
            vals = [self._scalar_operand(self.expr(e), e).code for e in a.elts]
            return Expr(f"qt::vec({{{', '.join(vals)}}})", P_ATOM, VEC)
        e = self.expr(a)
        self._need_array(e, node, f"{src}()")
        return Expr(e.code, e.prec, e.type)  # a copy, as np.array(x) makes

    def array_call(self, node, qual, args, kws) -> Expr:
        src = ast.unparse(node.func)
        if qual in ("numpy.zeros", "numpy.ones"):
            self._check_dtype(node)
            dims = self._shape_args(node, node.args[0])
            fn = qual.split(".")[1]
            return Expr(f"qt::{fn}({', '.join(dims)})", P_ATOM, VEC if len(dims) == 1 else MAT)
        if qual == "numpy.full":
            self._check_dtype(node)
            dims = self._shape_args(node, node.args[0])
            v = self._scalar_operand(args[1], node.args[1])
            return Expr(f"qt::full({', '.join(dims)}, {v.code})", P_ATOM, VEC if len(dims) == 1 else MAT)
        if qual in ("numpy.eye", "numpy.identity"):
            self._check_dtype(node)
            if len(node.args) != 1:
                raise Unsupported(node, f"{src}() with more than one argument is not supported")
            return Expr(f"qt::eye({self._int_code(node.args[0])})", P_ATOM, MAT)
        if qual == "numpy.linspace":
            if set(kws) - {"num"} or not 2 <= len(args) <= 3:
                raise Unsupported(node, "linspace(start, stop[, num]) only (no endpoint=, retstep=)")
            num = args[2] if len(args) == 3 else kws.get("num")
            parts = [self._scalar_operand(args[0], node).code, self._scalar_operand(args[1], node).code]
            if num is not None:
                if num.type not in (INT, BOOL):
                    raise Unsupported(node, "linspace num must be an integer")
                parts.append(num.code)
            return Expr(f"qt::linspace({', '.join(parts)})", P_ATOM, VEC)
        if qual == "numpy.arange":
            self._check_dtype(node)
            if not 1 <= len(args) <= 3:
                raise Unsupported(node, "arange takes 1 to 3 arguments")
            self.note("arange", "np.arange of ints is stored as doubles (exact for |values| < 2^53)")
            return Expr(f"qt::arange({', '.join(self._scalar_operand(a, node).code for a in args)})", P_ATOM, VEC)
        if qual in ("numpy.array", "numpy.asarray"):
            return self.array_literal(node, src)

        # everything below takes at least one array argument
        if not args:
            raise Unsupported(node, f"{src}() needs arguments")
        A = args[0]
        if qual in ("numpy.zeros_like", "numpy.ones_like", "numpy.full_like"):
            self._need_array(A, node, src)
            a = wrap(A, P_ATOM)
            dims = f"{a}.size()" if A.type == VEC else f"{a}.rows(), {a}.cols()"
            if qual.endswith("full_like"):
                return Expr(f"qt::full({dims}, {self._scalar_operand(args[1], node).code})", P_ATOM, A.type)
            return Expr(f"qt::{qual.split('.')[1][:-5]}({dims})", P_ATOM, A.type)
        if qual == "numpy.copy":
            self._need_array(A, node, src)
            return Expr(A.code, A.prec, A.type)
        if qual in _MEMBER:
            self._need_array(A, node, src)
            return Expr(f"{wrap(A, P_ATOM)}.{_MEMBER[qual]}()", P_ATOM, A.type)
        if qual in _HELPER:
            self._need_array(A, node, src)
            return Expr(f"{_HELPER[qual]}({A.code})", P_ATOM, A.type)
        if qual in _MASK_FN:
            self._need_array(A, node, src)
            return Expr(f"{wrap(A, P_ATOM)}.{_MASK_FN[qual]}()", P_ATOM, MaskType(A.type.rank))
        if qual in _REDUCE:
            name = _REDUCE[qual]
            if isinstance(A.type, MaskType) and name == "sum" and len(args) == 1 and not kws:
                return Expr(f"{wrap(A, P_ATOM)}.count()", P_ATOM, INT)
            if name in ("sort", "median"):
                self._need_array(A, node, src)
                if len(args) > 1 or kws:
                    raise Unsupported(node, f"{src}() with extra arguments is not supported")
                if A.type == MAT:
                    raise Unsupported(node, f"{src}() of a 2-D array is not supported in stage 2")
                return Expr(f"qt::{name}({A.code})", P_ATOM, VEC if name == "sort" else FLOAT)
            if name in ("quantile", "percentile"):
                self._need_array(A, node, src)
                if len(args) != 2 or kws or A.type == MAT:
                    raise Unsupported(node, f"{src}(a, q) of a 1-D array with a scalar q is supported; "
                                            "no axis= or method=")
                q = self._scalar_operand(args[1], node)
                return Expr(f"qt::{name}({A.code}, {q.code})", P_ATOM, FLOAT)
            return self._reduce(node, name, A, args[1:], kws)
        if qual in ("numpy.dot", "numpy.matmul"):
            if len(args) != 2:
                raise Unsupported(node, f"{src}() takes two arguments")
            return self.matmul(node, args[0], args[1])
        if qual == "numpy.outer":
            for a in args:
                if a.type != VEC:
                    raise Unsupported(node, "np.outer takes two Vecs")
            return Expr(f"qt::outer({args[0].code}, {args[1].code})", P_ATOM, MAT)
        if qual == "numpy.transpose":
            self._need_array(A, node, src)
            return self.array_attribute(ast.Attribute(attr="T"), A)
        if qual == "numpy.diag":
            self._need_array(A, node, src)
            if len(args) != 1 or kws:
                raise Unsupported(node, "np.diag with an offset is not supported")
            return Expr(f"qt::diag({A.code})", P_ATOM, MAT) if A.type == VEC else \
                Expr(f"qt::diagonal({A.code})", P_ATOM, VEC)
        if qual == "numpy.where":
            if len(args) != 3 or not isinstance(A.type, MaskType):
                raise Unsupported(node, "np.where(condition, x, y) with an array condition is supported")
            branches = []
            for b in args[1:]:
                if is_array(b.type):
                    if b.type.rank != A.type.rank:
                        raise Unsupported(node, "np.where branches must have the condition's shape")
                    branches.append(b.code)
                else:
                    branches.append(self._scalar_operand(b, node).code)
            return Expr(f"qt::where({A.code}, {branches[0]}, {branches[1]})", P_ATOM,
                        VEC if A.type.rank == 1 else MAT)
        if qual in ("numpy.all", "numpy.any", "numpy.count_nonzero"):
            if not isinstance(A.type, MaskType) or len(args) != 1 or kws:
                raise Unsupported(node, f"{src}() of a boolean array (a comparison) is supported")
            fn = {"numpy.all": "all", "numpy.any": "any", "numpy.count_nonzero": "count"}[qual]
            return Expr(f"{wrap(A, P_ATOM)}.{fn}()", P_ATOM, INT if fn == "count" else BOOL)
        if qual in ("numpy.maximum", "numpy.minimum"):
            if len(args) != 2:
                raise Unsupported(node, f"{src}() takes two arguments")
            a, b = args
            if is_array(a.type) and is_array(b.type) and a.type != b.type:
                raise Unsupported(node, f"{src}() of a Vec and a Mat is not supported")
            t = a.type if is_array(a.type) else b.type
            ca = a.code if is_array(a.type) else self._scalar_operand(a, node).code
            cb = b.code if is_array(b.type) else self._scalar_operand(b, node).code
            return Expr(f"qt::{qual.split('.')[1]}({ca}, {cb})", P_ATOM, t)
        if qual == "numpy.power":
            return self.array_power(node, args[0], args[1])
        if qual == "numpy.clip":
            return self._clip(node, A, node.args[1:], {k.arg: k.value for k in node.keywords})
        if qual == "numpy.trace":
            if A.type != MAT or len(args) != 1 or kws:
                raise Unsupported(node, "np.trace(A) of a 2-D array, without offset, is supported")
            return Expr(f"qt::trace({A.code})", P_ATOM, FLOAT)
        if qual == "numpy.ravel":
            self._need_array(A, node, src)
            return Expr(f"qt::ravel({A.code})", P_ATOM, VEC) if A.type == MAT else Expr(A.code, A.prec, VEC)
        if qual.startswith("numpy.linalg."):
            return self._linalg(node, qual, args)
        raise Unsupported(node, f"{src}() is not supported on arrays in stage 2")

    def _clip(self, node, A: Expr, bound_nodes, kw_nodes) -> Expr:
        """np.clip(a, lo, hi) / a.clip(lo, hi): scalar bounds, either of which may be None."""
        names = ["a_min", "a_max"]
        if len(bound_nodes) > 2 or set(kw_nodes) - {"a_min", "a_max", "min", "max"}:
            raise Unsupported(node, "clip(a, lo, hi) with scalar bounds is supported")
        b = dict(zip(names, bound_nodes))
        for k, v in kw_nodes.items():
            b["a_" + k if not k.startswith("a_") else k] = v
        lo, hi = (None if (n := b.get(k)) is None or (isinstance(n, ast.Constant) and n.value is None)
                  else n for k in names)
        if lo is None and hi is None:
            raise Unsupported(node, "clip() needs at least one bound (NumPy raises ValueError)")
        ex = []
        for n in (lo, hi):
            if n is None:
                ex.append(None)
                continue
            e = self.expr(n)
            if is_array(e.type):
                raise Unsupported(n, "array-valued clip bounds are not supported; use scalars")
            ex.append(self._scalar_operand(e, n))
        lo_e, hi_e = ex
        if not is_array(A.type):
            self.need_scalar(A, node, "np.clip()")
            x = self.as_double(A).code
            if hi_e is None:
                return Expr(f"qt::np_maximum({x}, {lo_e.code})", P_ATOM, FLOAT)
            if lo_e is None:
                return Expr(f"qt::np_minimum({x}, {hi_e.code})", P_ATOM, FLOAT)
            return Expr(f"qt::np_clip({x}, {lo_e.code}, {hi_e.code})", P_ATOM, FLOAT)
        # NumPy: clip(a, lo, None) is np.maximum(a, lo) and clip(a, None, hi) is np.minimum(a, hi)
        if hi_e is None:
            return Expr(f"qt::maximum({A.code}, {lo_e.code})", P_ATOM, A.type)
        if lo_e is None:
            return Expr(f"qt::minimum({A.code}, {hi_e.code})", P_ATOM, A.type)
        return Expr(f"qt::clip({A.code}, {lo_e.code}, {hi_e.code})", P_ATOM, A.type)

    def stack_call(self, node, qual) -> Expr:
        """np.column_stack / vstack / hstack / concatenate of a literal list of arrays."""
        src = ast.unparse(node.func)
        if len(node.args) != 1 or not isinstance(node.args[0], (ast.List, ast.Tuple)) or not node.args[0].elts:
            raise Unsupported(node, f"{src}() of a literal list of arrays, [a, b, ...], is supported")
        axis = 0
        for k in node.keywords:
            if qual == "numpy.concatenate" and k.arg == "axis" and isinstance(k.value, ast.Constant) \
                    and k.value.value in (0, 1, -1):
                axis = k.value.value
            else:
                raise Unsupported(node, f"keyword '{k.arg}' is not supported for {src}()")
        parts = []
        for e in node.args[0].elts:
            if isinstance(e, (ast.List, ast.Tuple)):  # np.concatenate(([x0], v)): a literal list of numbers
                parts.append(self.array_literal(ast.Call(func=node.func, args=[e], keywords=[]), "np.array"))
            else:
                parts.append(self.expr(e))
        for e, n in zip(parts, node.args[0].elts):
            if not is_array(e.type):
                raise Unsupported(n, f"{src}() needs arrays (scalars are not supported here)")
        vecs = all(e.type == VEC for e in parts)
        mats = all(e.type == MAT for e in parts)
        if qual == "numpy.column_stack":
            cols = [f"qt::as_col({e.code})" if e.type == VEC else f"qt::Mat({e.code})" for e in parts]
            return Expr(f"qt::hcat({{{', '.join(cols)}}})", P_ATOM, MAT)
        if qual in ("numpy.vstack", "numpy.row_stack"):
            rows = [f"qt::as_row({e.code})" if e.type == VEC else f"qt::Mat({e.code})" for e in parts]
            return Expr(f"qt::vcat({{{', '.join(rows)}}})", P_ATOM, MAT)
        if vecs and (qual == "numpy.hstack" or axis in (0, -1)):
            return Expr(f"qt::concat({{{', '.join(f'qt::Vec({e.code})' for e in parts)}}})", P_ATOM, VEC)
        if mats:
            fn = "hcat" if qual == "numpy.hstack" or axis in (1, -1) else "vcat"
            return Expr(f"qt::{fn}({{{', '.join(f'qt::Mat({e.code})' for e in parts)}}})", P_ATOM, MAT)
        raise Unsupported(node, f"{src}() of a mix of 1-D and 2-D arrays raises ValueError in NumPy")

    def einsum_call(self, node) -> Expr:
        """np.einsum('ij,ij->i', A, B) with one or two Vec / Mat operands and a literal spec."""
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            raise Unsupported(node, "np.einsum needs its subscripts as a string literal")
        if node.keywords:
            raise Unsupported(node, "keyword arguments to np.einsum (optimize=, out=, ...) are not supported")
        spec = node.args[0].value.replace(" ", "")
        ops = [self.expr(a) for a in node.args[1:]]
        if not 1 <= len(ops) <= 2:
            raise Unsupported(node, "np.einsum with one or two operands is supported")
        if "." in spec or not all(c.isalpha() and c.islower() or c in ",->" for c in spec):
            raise Unsupported(node, "np.einsum subscripts: lower-case letters only (no ellipsis)")
        lhs, _, out = spec.partition("->")
        ins = lhs.split(",")
        if len(ins) != len(ops):
            raise Unsupported(node, f"np.einsum spec '{spec}' names {len(ins)} operands but {len(ops)} are given")
        if "->" not in spec:  # implicit mode: labels used once, alphabetically
            out = "".join(sorted(c for c in set(lhs) - {","} if lhs.count(c) == 1))
        for e, l, n in zip(ops, ins, node.args[1:]):
            if not is_array(e.type):
                raise Unsupported(n, "np.einsum operands must be arrays")
            if len(l) != e.type.rank:
                raise Unsupported(n, f"np.einsum subscripts '{l}' don't match a {e.type}")
        if len(out) > 2 or len(set(out)) != len(out) or any(c not in lhs for c in out):
            raise Unsupported(node, f"np.einsum output '{out}' is not supported (at most 2 distinct input labels)")
        self.note("einsum", "np.einsum is computed by direct summation; NumPy's kernels add in a different "
                            "order, so results agree to rounding rather than bit for bit")
        mats = [e.code for e in ops]  # qt::EinsumArg takes a Mat or Vec by reference
        ranks = ", ".join(str(e.type.rank) for e in ops)
        t = {0: FLOAT, 1: VEC, 2: MAT}[len(out)]
        return Expr(f'qt::einsum{len(out)}("{spec}", {{{", ".join(mats)}}}, {{{ranks}}})', P_ATOM, t)

    def _linalg(self, node, qual, args):
        fn = qual.rsplit(".", 1)[1]
        A = args[0]
        if A.type != MAT:
            raise Unsupported(node, f"np.linalg.{fn} needs a Mat")
        if fn in ("eigh", "eigvalsh"):
            if len(args) != 1 or node.keywords:
                raise Unsupported(node, f"np.linalg.{fn}(A) without UPLO= is supported (the lower triangle is used)")
            self.note("eigh", "np.linalg.eigh: eigenvalues agree to rounding; each eigenvector is defined only up "
                              "to sign (and rotation within a repeated eigenvalue), which LAPACK and Eigen may "
                              "choose differently. Results like V @ diag(f(w)) @ V.T agree.")
            if fn == "eigvalsh":
                return Expr(f"qt::eigvalsh({A.code})", P_ATOM, VEC)
            return Expr(f"qt::eigh({A.code})", P_ATOM, TupleType((VEC, MAT)))
        if fn == "pinv":
            rc = [k for k in node.keywords if k.arg in ("rcond", "rtol")]
            if len(args) > 2 or len(rc) != len(node.keywords) or (len(args) == 2 and rc):
                raise Unsupported(node, "np.linalg.pinv(A) or pinv(A, rcond) with a scalar rcond is supported")
            r = args[1] if len(args) == 2 else (self.expr(rc[0].value) if rc else None)
            if r is not None and is_array(r.type):
                raise Unsupported(node, "array-valued rcond is not supported")
            self.note("pinv", "np.linalg.pinv agrees with NumPy to rounding; singular values within rounding "
                              "of the rcond cutoff can be kept by one and dropped by the other")
            extra = f", {self._scalar_operand(r, node).code}" if r is not None else ""
            return Expr(f"qt::pinv({A.code}{extra})", P_ATOM, MAT)
        if fn == "solve":
            if len(args) != 2 or not is_array(args[1].type):
                raise Unsupported(node, "np.linalg.solve(A, b) with b a Vec or Mat")
            return Expr(f"qt::solve({A.code}, {args[1].code})", P_ATOM, args[1].type)
        if len(args) != 1:
            raise Unsupported(node, f"np.linalg.{fn} takes one argument")
        t = FLOAT if fn == "det" else MAT
        return Expr(f"qt::{fn}({A.code})", P_ATOM, t)

    # --- random numbers (stage 3) ---------------------------------------------------------
    def default_rng(self, node) -> Expr:
        """np.random.default_rng(seed) -> qt::Rng(seed): the same PCG64 stream, bit for bit."""
        args = list(node.args) + [k.value for k in node.keywords if k.arg == "seed"]
        if any(k.arg != "seed" for k in node.keywords) or len(args) != 1:
            raise Unsupported(node, "default_rng() needs exactly one argument, the seed",
                              "without a seed it draws fresh entropy from the OS, so runs can't be "
                              "reproduced or checked; pass an integer seed")
        seed = self.expr(args[0])
        if seed.type not in (INT, BOOL):
            raise Unsupported(node, "the seed must be an integer (SeedSequence, Generator or array seeds "
                                    "are not supported)")
        return Expr(f"qt::Rng({seed.code})", P_ATOM, RNG)

    def rng_method(self, node, R: Expr, name: str) -> Expr:
        """rng.random / standard_normal / normal / uniform, with NumPy's argument order and
        defaults; size=None gives a float, an int a Vec, a pair a Mat (filled row by row)."""
        if name not in _RNG_METHODS:
            raise Unsupported(node, f"Generator.{name}() is not supported in stage 3",
                              "supported: random, standard_normal, normal, uniform")
        params = [p for p, _ in _RNG_METHODS[name]] + ["size"]
        if len(node.args) > len(params):
            raise Unsupported(node, f"Generator.{name}() takes at most {len(params)} positional arguments")
        bound = dict(zip(params, node.args))
        for k in node.keywords:
            if k.arg == "dtype":
                if ast.unparse(k.value) not in ("float", "np.float64", "numpy.float64"):
                    raise Unsupported(node, "only float64 draws are supported")
                continue
            if k.arg not in params or k.arg in bound:
                raise Unsupported(node, f"unsupported argument '{k.arg}' for Generator.{name}()")
            bound[k.arg] = k.value
        parts = []
        for p, default in _RNG_METHODS[name]:
            if p in bound:
                e = self.expr(bound[p])
                if is_array(e.type):
                    raise Unsupported(node, f"array-valued {p} is not supported; use a scalar")
                parts.append(self._scalar_operand(e, bound[p]).code)
            else:
                parts.append(default)
        size = bound.get("size")
        dims = []
        if size is not None and not (isinstance(size, ast.Constant) and size.value is None):
            dims = self._shape_args(node, size)
        t = FLOAT if not dims else VEC if len(dims) == 1 else MAT
        return Expr(f"{wrap(R, P_ATOM)}.{name}({', '.join(parts + dims)})", P_ATOM, t)

    def norm_array(self, node, fn, x: Expr, loc, scale) -> Expr:
        """scipy.stats.norm.<fn> applied elementwise to an array."""
        if fn in ("ppf", "sf") and (loc is not None or scale is not None):
            raise Unsupported(node, f"norm.{fn} of an array with loc/scale is not supported")
        if loc is None and scale is None:
            return Expr(f"qt::norm_{fn}({x.code})", P_ATOM, x.type)
        for e in (loc, scale):
            if e is not None:
                self.need_scalar(e, node, f"norm.{fn} loc/scale")
        lc = self.as_double(loc).code if loc is not None else "0.0"
        sc = self.as_double(scale).code if scale is not None else "1.0"
        return Expr(f"qt::norm_{fn}({x.code}, {lc}, {sc})", P_ATOM, x.type)
