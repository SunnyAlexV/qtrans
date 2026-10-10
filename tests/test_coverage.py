"""Coverage added after translating the Black-Litterman app: np.errstate, np.clip, mask
indexing, new axes, keepdims and Mat-Mat broadcasting, var/std/cumsum/cumprod over an axis,
trace, stacking, scalar np.where. Compiled against NumPy on hand-picked edge cases (signed
zeros, NaN, empty selections, shapes that don't broadcast) and random inputs."""
import textwrap

import numpy as np
import pytest

from qtrans import translate_source, verify_file
from qtrans.verify import find_compiler, find_eigen

try:
    find_compiler()
    find_eigen()
    HAVE = True
except RuntimeError:
    HAVE = False

HEAD = "import math\nimport numpy as np\nfrom qtrans.typing import Vec, Mat\n"
BODY = textwrap.dedent('''
def errstate_ratio(a: Vec, b: Vec) -> Vec:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b != 0.0, a / b, 0.0)
    return r

def ann_sharpe(x: Vec, fpy: float) -> float:
    sd = np.std(x, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, np.mean(x) / sd * math.sqrt(fpy), np.nan)

def clip_both(v: Vec, lo: float, hi: float) -> Vec:
    return 1.0 / np.clip(v, lo, hi)  # 1/x shows which zero a tie returned

def clip_lo(v: Vec, lo: float) -> Vec:
    return 1.0 / np.clip(v, lo, None)

def clip_hi_method(v: Vec, hi: float) -> Vec:
    return 1.0 / v.clip(None, hi)

def clip_scalar(d: float) -> float:
    return math.copysign(1.0, float(np.clip(d, 0.0, 10.0))) * float(np.clip(d, -10.0, None))

def max_tie(v: Vec) -> Vec:
    return 1.0 / np.maximum(v, 0.0)

def min_tie(v: Vec) -> Vec:
    return 1.0 / np.minimum(0.0, v)

def astype_ravel(M: Mat) -> Vec:
    return M.astype(float).ravel() * 2.0

def col_newaxis(v: Vec, M: Mat) -> Mat:
    return M - v[:, None]

def row_newaxis(v: Vec, M: Mat) -> Mat:
    return M / v[None, :]

def demean_keepdims(X: Mat) -> Mat:
    return X - X.mean(axis=0, keepdims=True)

def scale_rows(X: Mat) -> Mat:
    return X / np.sum(X, axis=1, keepdims=True)

def var_axis0(R: Mat) -> Vec:
    return np.var(R, axis=0, ddof=1)

def std_axis1(R: Mat) -> Vec:
    return R.std(axis=1)

def cumprod_axis0(R: Mat) -> Mat:
    return np.cumprod(1.0 + R, axis=0)

def cumsum_axis1(R: Mat) -> Mat:
    return R.cumsum(axis=1)

def trace_both(S: Mat) -> float:
    return np.trace(S) + S.trace()

def mask_select(x: Vec, c: float) -> Vec:
    return x[x >= c]

def cvar(losses: Vec, alpha: float) -> float:
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)

def finite_pairs(a: Vec, b: Vec) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a2, b2 = a[ok], b[ok]
    return float(np.sum(a2 * b2))

def mask_rows(M: Mat, c: float) -> Mat:
    return M[M[:, 0] > c]

def mask_assign(w: Vec, c: float) -> Vec:
    w = w.copy()
    w[w < c] = 0.0
    w[w > 1.0] += 1.0
    return w

def column_stack_(a: Vec, b: Vec) -> Mat:
    return np.column_stack([a - a.mean(), b - b.mean(), a * b])

def vstack_(a: Vec, b: Vec) -> Mat:
    return np.vstack([a, b])

def concat_(a: Vec, b: Vec) -> Vec:
    return np.concatenate([a, b, a])

def copy_then_mutate(S: Mat, G: Mat) -> Mat:
    S = S.copy()
    S += 0.5 * (G + G.T)
    return S

def bcast_inplace(X: Mat) -> Mat:
    Y = X.copy()
    Y -= Y.mean(axis=1, keepdims=True)
    return Y

def asarray_alias(X: Vec) -> float:
    Y = np.asarray(X, dtype=float)
    return float(np.sum(Y))

def scalar_where(x: float, n: int) -> float:
    return np.where(n > 0, x / max(n, 1), -1.0)
''')
SOURCE = HEAD + BODY + textwrap.dedent('''
nan, inf = float("nan"), float("inf")
QTRANS_CASES = {
    "clip_both": [([-1.0, 0.5, 2.0, nan, -0.0, 0.0], 0.0, 1.0), ([5.0, -5.0, 0.5], 3.0, 1.0),
                  ([0.5, 2.0], nan, 1.0), ([0.5, -2.0], 0.0, nan), ([-0.0, 0.0], -0.0, 0.0),
                  ([0.0, -0.0], -0.0, 1.0), ([-0.0, 0.0], 0.0, 1.0), ([0.0, -0.0], -1.0, -0.0)],
    "clip_lo": [([-0.0, 0.0, -1.0, 2.0, nan], 0.0), ([0.0, -0.0], -0.0)],
    "clip_hi_method": [([-0.0, 0.0, 1.0, nan], 0.0), ([0.0, -0.0], -0.0)],
    "clip_scalar": [(-0.0,), (0.0,), (nan,), (11.0,), (-3.0,)],
    "max_tie": [([-0.0, 0.0, -1.0, nan],)],
    "min_tie": [([-0.0, 0.0, 1.0, nan],)],
    "col_newaxis": [([1.0, 2.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    ([1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]])],          # shapes don't broadcast
    "row_newaxis": [([1.0, 2.0, 4.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    ([1.0, 2.0], [[1.0, 2.0, 3.0]])],
    "demean_keepdims": [([[1.0, 2.0], [3.0, 5.0], [0.1, 0.7]],), ([[1.0, 2.0, 3.0]],)],
    "scale_rows": [([[1.0, 3.0], [0.0, 0.0]],)],                            # 0/0 row -> NaN in both
    "var_axis0": [([[1.0, 2.0], [3.0, 5.0], [0.1, 0.7]],), ([[1.0, 2.0]],)],  # ddof=1 on one row: NaN
    "std_axis1": [([[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]],)],
    "cumprod_axis0": [([[0.1, -0.2], [0.05, 0.3], [-0.5, 0.0]],)],
    "cumsum_axis1": [([[0.1, 0.2, 0.3], [1e16, 1.0, -1e16]],)],
    "trace_both": [([[1.0, 2.0], [3.0, 4.0]],), ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],),
                   ([[0.1 * k for k in range(20)] for _ in range(20)],)],
    "mask_select": [([1.0, 2.0, 3.0], 5.0), ([nan, 1.0, 2.0], 1.5), ([], 0.0)],
    "finite_pairs": [([1.0, nan, 3.0, inf], [2.0, 2.0, nan, 1.0]), ([1.0, 2.0], [1.0, 2.0, 3.0])],
    "mask_rows": [([[1.0, 2.0], [3.0, 4.0], [0.5, 6.0]], 0.9), ([[1.0, 2.0]], 5.0)],
    "mask_assign": [([0.5, -1.0, 2.0, 1.5, nan], 0.0)],
    "column_stack_": [([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), ([1.0, 2.0], [1.0, 2.0, 3.0])],
    "vstack_": [([1.0, 2.0], [3.0, 4.0]), ([1.0], [1.0, 2.0])],
    "concat_": [([], [1.0]), ([1.0, 2.0], [3.0])],
    "copy_then_mutate": [([[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 0.0]]),
                         ([[1.0, 2.0]], [[1.0, 2.0]])],                     # (2,2) can't go into (1,2)
    "bcast_inplace": [([[1.0, 2.0, 6.0], [3.0, 5.0, 1.0]],)],
    "scalar_where": [(3.0, 0), (3.0, 2), (-1.0, -5)],
}
''')
FUNCTIONS = [s.split("(")[0] for s in BODY.split("\ndef ")[1:]]

compiled = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("cov")
    f = d / "coverage.py"
    f.write_text(SOURCE, encoding="utf-8")
    return verify_file(f, n_random=40, build_dir=d / "build")


def fn(report, name):
    return next(f for f in report.functions if f.name == name)


@compiled
def test_all_translate_without_warnings(report):
    assert not report.translation.report.rejected
    assert report.warnings == []


@compiled
@pytest.mark.parametrize("name", FUNCTIONS)
def test_matches_numpy_bit_for_bit(report, name):
    """No BLAS or libm in any of these: every result must be identical, not just close."""
    f = fn(report, name)
    bad = [c for c in f.cases if c.outcome == "mismatch"]
    assert not bad, [(c.args, c.py, c.cpp) for c in bad[:3]]
    for c in f.cases:
        if c.outcome == "match" and c.py[0] == "ok":
            assert c.ulps == 0, (name, c.args, c.py, c.cpp)


def _given(report, name):
    return [c for c in fn(report, name).cases if c.source == "given"]


@compiled
def test_signed_zero_ties_follow_numpy(report):
    """np.maximum/np.minimum return the second argument on a tie, np.clip keeps x; 1/x shows
    which zero came back. These failed before (maximum(-0.0, 0.0) gave -0.0)."""
    py = _given(report, "max_tie")[0].py[1]          # 1 / np.maximum([-0.0, 0.0], 0.0)
    assert list(py[:2]) == [np.inf, np.inf]
    py = _given(report, "min_tie")[0].py[1]          # 1 / np.minimum(0.0, [-0.0, 0.0])
    assert list(py[:2]) == [-np.inf, np.inf]
    for name in ("max_tie", "min_tie", "clip_lo", "clip_hi_method", "clip_scalar"):
        for c in _given(report, name):
            assert c.outcome == "match", (name, c.args, c.py, c.cpp)


@compiled
def test_shape_errors_raise_like_numpy(report):
    for name, k in (("col_newaxis", 1), ("row_newaxis", 1), ("finite_pairs", 1), ("column_stack_", 1),
                    ("vstack_", 1), ("copy_then_mutate", 1)):
        c = _given(report, name)[k]
        assert c.py[0] == "exc" and c.cpp[0] == "exc" and c.outcome == "match", (name, c.py, c.cpp)


# --- generated code: what is passed how, and what is refused ------------------------------
def cpp(src):
    r = translate_source(HEAD + src, "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r.header


def rejection(src):
    r = translate_source(HEAD + src, "m")
    assert r.report.rejected, "expected a rejection"
    return r.report.rejected[0].reason


def test_copy_first_then_modify_is_by_value():
    """`S = S.copy()` before `S += ...`: the caller never sees the change, so S is passed by
    value (this was wrongly refused as 'modified in place and reassigned')."""
    h = cpp("def f(S: Mat) -> Mat:\n    S = S.copy()\n    S += 1.0\n    return S\n")
    assert "qt::Mat f(qt::Mat S)" in h


def test_modify_then_rebind_is_still_refused():
    assert "both modified in place and reassigned" in rejection(
        "def f(S: Mat) -> Mat:\n    S[0, 0] = 1.0\n    S = S * 2.0\n    return S\n")


def test_asarray_shares_memory_like_numpy():
    """np.asarray(x) of a float64 array returns x itself, so modifying the result modifies x."""
    assert "share memory" in rejection(
        "def f(x: Vec) -> float:\n    y = np.asarray(x, dtype=float)\n    y[0] = 1.0\n    return x[0]\n")
    # x = np.asarray(x) is the same array: modifying it afterwards modifies the caller's array,
    # so x is passed by reference and the rebinding disappears
    h = cpp("def f(x: Vec) -> float:\n    x = np.asarray(x)\n    x[0] = 1.0\n    return x[1]\n")
    assert "double f(qt::Vec& x)" in h and "already a float64 array" in h


def test_mask_indexing_is_a_copy_not_a_view():
    h = cpp("def f(M: Mat, c: float) -> Mat:\n    R = M[M[:, 0] > c]\n    R[0, 0] = 1.0\n    return R\n")
    assert "qt::select_rows(" in h


def test_errstate_block_is_translated_as_its_body():
    r = translate_source(HEAD + "def f(a: Vec) -> Vec:\n    with np.errstate(divide='ignore'):\n"
                                "        y = 1.0 / a\n    return y\n", "m")
    assert not r.report.rejected and "y = 1.0 / a" in r.header
    assert any("errstate" in n for f in r.report.functions for n in f.notes)


def test_other_with_blocks_are_refused():
    assert "with-blocks" in rejection("def f(a: Vec) -> float:\n    with open('x') as h:\n"
                                      "        pass\n    return 0.0\n")
