"""Stage 2, generated text: how arrays are passed, indexed, broadcast and protected from
aliasing, and what is refused. No compiler needed."""
import pytest

from qtrans import translate_source

HEAD = "import numpy as np\nfrom qtrans.typing import Vec, Mat\n\n"


def cpp(src):
    r = translate_source(HEAD + src, "m")
    bad = r.report.rejected
    assert not bad, [(f.name, f.reason) for f in bad]
    return r.header


def rejection(src):
    r = translate_source(HEAD + src, "m")
    assert r.report.rejected, "expected a rejection"
    return r.report.rejected[0].reason


# --- parameter passing follows Python's sharing semantics --------------------------------
def test_read_only_array_by_const_reference():
    assert "double f(const qt::Vec& a)" in cpp("def f(a: Vec) -> float:\n    return np.sum(a)\n")


def test_modified_array_by_reference():
    h = cpp("def f(a: Vec) -> None:\n    a[0] = 1.0\n")
    assert "void f(qt::Vec& a)" in h and "a(0) = 1.0;" in h


def test_in_place_operator_is_a_modification():
    assert "void f(qt::Vec& a, double k)" in cpp("def f(a: Vec, k: float) -> None:\n    a *= k\n")


def test_rebound_array_by_value():
    assert "qt::Vec f(qt::Vec a)" in cpp("def f(a: Vec) -> Vec:\n    a = a * 2\n    return a\n")


def test_modification_through_a_callee_propagates():
    h = cpp("def g(a: Vec) -> None:\n    a[0] = 0.0\n\ndef f(b: Vec) -> None:\n    g(b)\n")
    assert "void f(qt::Vec& b)" in h


# --- indexing and slicing ----------------------------------------------------------------
def test_negative_literal_index():
    assert "return a(a.size() - 1);" in cpp("def f(a: Vec) -> float:\n    return a[-1]\n")


def test_index_that_may_be_negative_is_wrapped():
    assert "a(qt::idx(i, a.size()))" in cpp("def f(a: Vec, i: int) -> float:\n    return a[i]\n")


def test_loop_index_proven_non_negative():
    h = cpp("def f(a: Vec) -> float:\n    s = 0.0\n    for t in range(1, len(a)):\n"
            "        s += a[t - 1]\n    return s\n")
    assert "a(t - 1)" in h and "qt::idx" not in h


def test_slices_use_python_rules():
    h = cpp("def f(a: Vec) -> Vec:\n    return a[1:] - a[:-1]\n")
    assert "qt::slice(a, 1, qt::none) - qt::slice(a, qt::none, -1)" in h


def test_matrix_rows_columns_and_elements():
    h = cpp("def f(A: Mat, i: int) -> float:\n    return A[0, 1] + np.sum(A[:, 2]) + np.sum(A[1])\n")
    # rows and columns are checked before Eigen forms the view (Python's index rules)
    assert "A(0, 1)" in h and "qt::col(A, 2)" in h and "qt::row(A, 1).transpose()" in h


def test_two_dimensional_slice():
    h = cpp("def f(A: Mat) -> Mat:\n    B = A[:, 1:]\n    return B[1:3, :]\n")
    assert "qt::block(A, qt::none, qt::none, 1, qt::none)" in h and "qt::block(B, 1, 3, qt::none, qt::none)" in h


def test_shape_entries():
    h = cpp("def f(A: Mat) -> int:\n    return A.shape[0] * A.shape[1]\n")
    # each size is below 2**45 (address space), but a product of two could leave 64 bits: checked
    assert "qt::imul(A.rows(), A.cols())" in h


# --- broadcasting and products --------------------------------------------------------------
def test_matrix_minus_vector_broadcasts_over_rows():
    # mu has A's number of columns, provably: Eigen's rowwise operation
    h = cpp("def f(A: Mat) -> Mat:\n    mu = A.mean(axis=0)\n    return A - mu\n")
    assert "A.rowwise() - mu.transpose()" in h
    # an unrelated v may have length 1 (or A one column): NumPy broadcasting, done by the helper
    h = cpp("def f(A: Mat, v: Vec) -> Mat:\n    return A - v\n")
    assert "qt::broadcast(A, v.transpose(), qt::sub_op{})" in h


def test_matmul_dispatch_by_rank():
    h = cpp("def f(A: Mat, v: Vec) -> float:\n    return v @ A @ v\n")
    assert "qt::dot(qt::vecmat(v, A), v)" in h


def test_power_fast_paths_match_numpy():
    h = cpp("def f(a: Vec) -> Vec:\n    return a ** 2 + a ** 0.5 + a ** -1 + a ** 3\n")
    assert "a.square()" in h and "a.sqrt()" in h and "a.inverse()" in h and "qt::pow(a, 3.0)" in h


def test_scalars_become_doubles_for_eigen():
    assert "2.0 * a" in cpp("def f(a: Vec) -> Vec:\n    return 2 * a\n")


# --- Eigen is lazy: read everything before overwriting --------------------------------------
def test_slice_assignment_reading_itself_is_evaluated_first():
    h = cpp("def f(V: Vec) -> None:\n    V[1:-1] = V[:-2] + V[2:]\n")
    assert "qt::slice(V, 1, -1) = (qt::slice(V, qt::none, -2) + qt::slice(V, 2, qt::none)).eval();" in h


def test_shrinking_reassignment_is_evaluated_first():
    h = cpp("def f(S: Vec) -> Vec:\n    S = S[:-1] * 2\n    return S\n")
    assert "S = (qt::slice(S, qt::none, -1) * 2.0).eval();" in h


def test_elementwise_update_needs_no_eval():
    assert ".eval()" not in cpp("def f(V: Vec) -> Vec:\n    V = V * 2 + 1\n    return V\n")


# --- refusals -----------------------------------------------------------------------------------
@pytest.mark.parametrize("src, expected", [
    ("def f(a: np.ndarray) -> float:\n    return 0.0\n", "doesn't say whether the array is 1-D or 2-D"),
    ("def f(a: Vec) -> Vec:\n    b = a\n    b[0] = 1.0\n    return b\n", "share memory"),
    ("def f(a: Vec) -> Vec:\n    b = a[1:]\n    b *= 2\n    return a\n", "share memory"),
    ("def f(a: Vec) -> None:\n    a[0] = 1.0\n    a = a * 2\n", "both modified in place and reassigned"),
    ("def f(a: Vec) -> float:\n    if a:\n        return 1.0\n    return 0.0\n", "truth value of an array"),
    ("def f(a: Vec, b: Vec) -> bool:\n    return (a > 0).all() and (b > 0)\n", "`and`/`or` on arrays"),
    ("def f(a: Vec) -> None:\n    for x in a:\n        a[0] = x\n", "while iterating"),
    ("def g(x: Vec, y: Vec) -> None:\n    x[0] = y[0]\n\ndef f(a: Vec) -> None:\n    g(a, a)\n",
     "is passed twice to g()"),
    ("def g(x: Vec) -> None:\n    x[0] = 1.0\n\ndef f(a: Vec) -> None:\n    g(a * 2)\n", "modifies its argument"),
    ("def f(a: Vec) -> float:\n    return a.reshape(2, 2).sum()\n", "not supported"),
    ("def f(A: Mat, i: int) -> Vec:\n    return A[i, 1:3]\n", "mixing an index and a partial slice"),
    ("def f(a: Vec) -> float:\n    return sum(a)\n", "built-in sum() of an array"),
])
def test_array_rejections(src, expected):
    assert expected in rejection(src)


def test_stored_comparison_is_a_mask():
    h = cpp("def f(a: Vec) -> float:\n    m = a > 0\n    return np.sum(np.where(m, a, 0.0)) / np.sum(m)\n")
    assert "qt::Mask m = a > 0.0;" in h and "m.count()" in h


def test_copy_breaks_the_alias():
    h = cpp("def f(a: Vec) -> Vec:\n    b = a.copy()\n    b[0] = 1.0\n    return b\n")
    assert "qt::Vec b = a;" in h and "b(0) = 1.0;" in h


def test_arrays_pull_in_eigen_header_only_when_used():
    assert "qtrans_arrays.hpp" in cpp("def f(a: Vec) -> float:\n    return a[0]\n")
    assert "qtrans_arrays.hpp" not in cpp("def f(x: float) -> float:\n    return x\n")
