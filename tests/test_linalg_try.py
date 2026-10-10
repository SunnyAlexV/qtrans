"""np.linalg.eigh / eigvalsh / pinv, np.einsum, and try / except np.linalg.LinAlgError, as the
Black-Litterman app uses them (nearest_psd, Ledoit-Wolf, the solve-or-pinv fallbacks). These go
through LAPACK in NumPy and Eigen here, so they agree to rounding, not bit for bit."""
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
def nearest_psd(matrix: Mat, eps: float = 1e-10) -> Mat:
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T

def eigenvalues(A: Mat) -> Vec:
    return np.linalg.eigvalsh(A)

def eig_pair_values(A: Mat) -> Vec:
    return np.linalg.eigh(A)[0]

def ledoit_wolf_identity(X: Mat) -> Mat:
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / t
    m = np.trace(S) / n
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n
    sq = np.einsum("ij,ij->i", Xc, Xc)
    term1 = float(np.dot(sq, sq))
    term2 = float(np.sum((Xc @ S) * Xc))
    term3 = t * float(np.sum(S * S))
    b_bar2 = (term1 - 2.0 * term2 + term3) / (t ** 2) / n
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    shrink = b2 / d2 if d2 > 0 else 1.0
    sigma = shrink * m * np.eye(n) + (a2 / d2 if d2 > 0 else 0.0) * S
    return sigma

def implied_confidence_weights(mu_excess: Vec, Sigma: Mat, delta: float) -> Vec:
    Sigma = np.asarray(Sigma, dtype=float)
    try:
        return np.linalg.solve(delta * Sigma, np.asarray(mu_excess, dtype=float))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(delta * Sigma) @ np.asarray(mu_excess, dtype=float)

def posterior_sigma(Sigma: Mat, P: Mat, Omega: Mat, tau: float) -> Mat:
    tauS = tau * Sigma
    A = P @ tauS @ P.T + Omega
    try:
        M = tauS - tauS @ P.T @ np.linalg.solve(A, P @ tauS)
    except np.linalg.LinAlgError:
        M = tauS - tauS @ P.T @ np.linalg.pinv(A) @ P @ tauS
    return nearest_psd(Sigma + M)

def partial_then_fail(A: Mat, b: Vec) -> tuple[float, int]:
    flag = 0
    x = 1.0
    try:
        x = 2.0
        y = np.linalg.solve(A, b)
        x = float(y[0])
        flag = 1
    except np.linalg.LinAlgError:
        flag = -1
    return x, flag

def retry_loop(A: Mat, b: Vec) -> int:
    tries = 0
    M = A.copy()
    while True:
        tries += 1
        try:
            np.linalg.solve(M, b)
            break
        except np.linalg.LinAlgError:
            M = M + np.eye(M.shape[0])
        if tries > 5:
            break
    return tries

def value_error_from_numpy(a: Vec, q: float) -> float:
    try:
        return np.quantile(a, q)
    except ValueError:
        return -1.0

def pinv_of(A: Mat) -> Mat:
    return np.linalg.pinv(A)

def pinv_rcond(A: Mat, r: float) -> Mat:
    return np.linalg.pinv(A, rcond=r)

def rowdots(A: Mat, B: Mat) -> Vec:
    return np.einsum("ij,ij->i", A, B)

def es_matmul(A: Mat, B: Mat) -> Mat:
    return np.einsum("ij,jk->ik", A, B)

def es_trace(A: Mat) -> float:
    return np.einsum("ii->", A) + np.einsum("ii", A)

def es_misc(a: Vec, A: Mat) -> tuple[float, Vec, Mat, Mat, float]:
    return (np.einsum("i,i->", a, a), np.einsum("ij,j->i", A, a), np.einsum("i,j->ij", a, a),
            np.einsum("ij->ji", A), np.einsum("ij->", A))

def es_implicit(A: Mat, B: Mat) -> Mat:
    return np.einsum("ij,jk", A, B)

nan = float("nan")
QTRANS_CASES = {
    "nearest_psd": [([[1.0, 2.0], [2.0, 1.0]],), ([[1.0, 0.0], [0.0, 1.0]],), ([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],),
                    ([[1.0, nan], [nan, 1.0]],), ([[0.0]],)],
    "eigenvalues": [([[2.0, 99.0], [1.0, 3.0]],), ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), ([[1.0, nan], [2.0, 1.0]],)],
    "implied_confidence_weights": [([0.01, 0.02], [[0.04, 0.0], [0.0, 0.09]], 2.5),
                                   ([0.01, 0.02], [[1.0, 1.0], [1.0, 1.0]], 2.5),       # singular: pinv branch
                                   ([0.01, 0.02], [[0.0, 0.0], [0.0, 0.0]], 2.5)],
    "posterior_sigma": [([[0.04, 0.01], [0.01, 0.09]], [[1.0, -1.0]], [[0.001]], 0.05),
                        ([[0.04, 0.01], [0.01, 0.09]], [[0.0, 0.0]], [[0.0]], 0.05)],  # singular A
    "partial_then_fail": [([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0]), ([[2.0, 0.0], [0.0, 4.0]], [1.0, 1.0])],
    "retry_loop": [([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0]), ([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]),
                   ([[-1.0, 0.0], [0.0, -1.0]], [1.0, 1.0])],
    "value_error_from_numpy": [([1.0, 2.0, 3.0], 1.5), ([1.0, 2.0], 0.5)],
    "pinv_of": [([[1.0, 2.0], [2.0, 4.0]],), ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],), ([[0.0, 0.0], [0.0, 0.0]],),
                ([[1.0, nan], [0.0, 1.0]],), ([[3.0]],)],
    "pinv_rcond": [([[1.0, 0.0], [0.0, 1e-8]], 1e-6), ([[1.0, 0.0], [0.0, 1e-8]], 1e-10)],
    "rowdots": [([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 1.0], [0.5, 2.0], [0.0, 1.0]]), ([[1.0, 2.0]], [[1.0, 2.0, 3.0]])],
    "es_matmul": [([[1.0, 2.0, 3.0]], [[1.0], [2.0], [3.0]]), ([[1.0, 2.0]], [[1.0, 2.0]])],
}
''')
SOURCE = HEAD + BODY
FUNCTIONS = [s.split("(")[0] for s in BODY.split("\ndef ")[1:]]
compiled = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("lin")
    f = d / "linalg_try.py"
    f.write_text(SOURCE, encoding="utf-8")
    return verify_file(f, n_random=60, build_dir=d / "build")


def fn(report, name):
    return next(f for f in report.functions if f.name == name)


def given(report, name):
    return [c for c in fn(report, name).cases if c.source == "given"]


@compiled
def test_all_translate_without_warnings(report):
    assert not report.translation.report.rejected
    assert report.warnings == []


@compiled
@pytest.mark.parametrize("name", FUNCTIONS)
def test_matches_numpy(report, name):
    bad = [c for c in fn(report, name).cases if c.outcome == "mismatch"]
    assert not bad, [(c.args, c.py, c.cpp) for c in bad[:3]]


@compiled
def test_fallback_branch_runs_on_singular_matrices(report):
    """The pinv branch must actually be taken (solve raises on the singular cases) and agree."""
    single, ones, zeros = given(report, "implied_confidence_weights")
    for c in (ones, zeros):
        assert c.outcome == "match" and c.cpp[0] == "ok"
    assert np.allclose(ones.py[1], np.linalg.pinv(2.5 * np.ones((2, 2))) @ np.array([0.01, 0.02]))


@compiled
def test_statements_before_the_error_keep_their_effect(report):
    """x = 2.0 runs, then solve raises: Python returns (2.0, -1); so must C++."""
    singular, fine = given(report, "partial_then_fail")
    assert singular.py == ("ok", (2.0, -1)) and singular.cpp == ("ok", (2.0, -1))
    assert fine.cpp[1][1] == 1


@compiled
def test_retry_loop_breaks_out_of_try(report):
    zero, ident, neg = given(report, "retry_loop")
    assert zero.cpp == zero.py == ("ok", 2)     # singular, then zeros + I solves
    assert ident.cpp == ident.py == ("ok", 1) and neg.cpp == neg.py == ("ok", 1)


@compiled
def test_eigh_reads_the_lower_triangle_and_propagates_nan(report):
    lower, rect, upper_nan = given(report, "eigenvalues")
    assert lower.outcome == "match" and np.allclose(lower.cpp[1], np.linalg.eigvalsh([[2.0, 0.0], [1.0, 3.0]]))
    assert rect.py[1] == "ValueError" and rect.cpp == ("exc", "ValueError")      # LinAlgError is a ValueError
    assert upper_nan.outcome == "match"                                          # NaN above the diagonal: ignored


@compiled
def test_pinv_edge_cases(report):
    for c in given(report, "pinv_of"):
        assert c.outcome == "match", (c.args, c.py, c.cpp)


@compiled
def test_einsum_broadcasts_size_one_dimensions(report):
    c = given(report, "es_matmul")[1]
    assert c.py == ("ok", c.py[1]) and np.array_equal(c.cpp[1], np.array([[3.0, 6.0]]))


# --- what is refused, and why --------------------------------------------------------------
def rejection(src):
    r = translate_source(HEAD + src, "m")
    assert r.report.rejected, "expected a rejection"
    return r.report.rejected[0].reason


def try_src(handler, body="return np.linalg.det(A)"):
    return f"def f(A: Mat) -> float:\n    try:\n        {body}\n    except {handler}:\n        return 0.0\n"


@pytest.mark.parametrize("handler,why", [
    ("ZeroDivisionError", "gives inf in C++"), ("ArithmeticError", "give inf in C++"),
    ("OverflowError", "gives inf in C++"), ("Exception", "carries on with inf/NaN"),
    ("(ValueError, IndexError)", "several exception types in one clause")])
def test_handlers_that_cannot_match_python_are_refused(handler, why):
    assert why in rejection(try_src(handler))


def test_bare_except_is_refused():
    assert "bare `except:`" in rejection(try_src("").replace("except :", "except:"))


def test_value_error_around_math_domain_functions_is_refused():
    """math.sqrt(-1) raises ValueError in Python but returns NaN in C++: the handler would run
    in one and not the other. Also through a called function."""
    assert "math.sqrt" in rejection(try_src("ValueError", "return math.sqrt(A[0, 0])"))
    src = ("def g(x: float) -> float:\n    return math.log(x)\n\n" + try_src("ValueError", "return g(A[0, 0])"))
    assert "math.sqrt" in rejection(src)


def test_value_error_from_numpy_is_fine():
    r = translate_source(HEAD + try_src("ValueError", "return np.quantile(A[0], 2.0)"), "m")
    assert not r.report.rejected and "catch (const std::invalid_argument&)" in r.header


def test_finally_and_else_are_refused():
    assert "finally" in rejection("def f(A: Mat) -> float:\n    try:\n        x = 1.0\n    finally:\n"
                                  "        x = 2.0\n    return x\n")


def test_using_the_exception_object_is_refused():
    assert "caught exception" in rejection("def f(A: Mat) -> float:\n    try:\n        return np.linalg.det(A)\n"
                                           "    except np.linalg.LinAlgError as e:\n        raise e\n")


# Badly scaled matrices (entries near 1e8 beside entries near 1e-2) found by the random cases.
# Rebuilding the pseudo-inverse from the SVD lost 5-6 digits here (errors 2e-10 .. 9e-9 against
# a 60-digit reference) while LAPACK's stayed near 1e-15; qt::pinv's QR route for full rank must
# stay as accurate as NumPy.
HARD = [[["0x1.e51e1f26e7a03p-5", "0x1.a4f57e3fb8139p-3", "0x1.735ba23316cc1p+3", "0x1.a6017e0f229c3p+0", "0x1.1a7daaad5c9bfp+1", "0x1.52878d43133f2p+4"], ["0x1.49413934a7f30p-3", "0x1.7a67402b94cb5p-3", "0x1.37e1d51bb5b3cp-2", "0x1.464701f9d8136p+1", "-0x1.0f54c63fb6698p+0", "0x1.c99c751e56bccp-2"], ["0x1.535799bea2f44p+3", "0x1.d1e4b72315851p-3", "0x1.21b5700848721p-2", "0x1.1530a37df7ef3p-5", "-0x1.70d8207257586p+0", "0x1.d5b8ee0020af0p-2"], ["0x1.6b075e2b541d6p-6", "0x1.6e0812e1d5e74p-1", "0x1.0000000000000p+0", "0x1.e7e984fe8857ep-1", "0x1.d68a165e76897p-5", "0x1.e07bc7437f05ep+3"], ["-0x0.0p+0", "0x1.32b45ff9fb8d6p-4", "0x1.8e496ad434e53p+2", "0x1.2f6b5dd86eea0p-1", "0x1.30f48e44d47f9p-6", "0x1.e3132099dd7abp-7"], ["0x1.7d78400000000p+26", "-0x1.cd9e309309b74p+0", "0x1.32ae47902cd91p-2", "0x1.837fc202beeb6p-3", "-0x1.b9e6d05baf4aep+0", "0x1.66f7499be437ep+1"]], [["-0x1.90c773952169ap+0", "0x1.2141317e404ffp+4", "0x1.01ecd452cd230p+0", "0x1.7038d0a886581p-4", "0x1.53267844dafcdp-3"], ["-0x1.0000000000000p+0", "0x1.7d78400000000p+26", "0x1.e5ee492ba03b7p+4", "0x1.d2d7fcf4d03ecp-2", "0x1.47fc25a5a8039p-5"], ["0x1.614d009680d98p-6", "0x1.0000000000000p-1", "-0x1.c58cd44307ce4p+0", "0x1.593753ad0b7abp-6", "0x1.8d5e1cdd4a438p-2"], ["0x1.bbf6e37d70940p+0", "0x1.ee074e96f0735p-3", "0x1.44a47937dd773p+3", "0x1.d7c38efb6be75p+1", "0x1.5798ee2308c3ap-27"], ["0x1.1fc33426ed0b1p-6", "0x1.0000000000000p-1", "0x1.d7eaec414f7a5p-6", "0x1.3ebf0ea3be2ecp-6", "-0x0.0p+0"]], [["0x1.056f2dbc37793p+2", "0x1.134e889b52c8ap+6", "0x1.273e918362629p-3"], ["0x1.0000000000000p+0", "0x1.7d78400000000p+26", "0x1.0cb9c5cf10081p+4"], ["0x1.2432e4900a3acp+0", "0x1.e3cb78e8de149p-7", "0x1.b7da0f9711003p-6"]]]


@compiled
def test_pinv_is_accurate_on_badly_scaled_matrices(tmp_path):
    mpmath = pytest.importorskip("mpmath")
    mats = [[[float.fromhex(x) for x in row] for row in m] for m in HARD]
    src = HEAD + "def p(A: Mat) -> Mat:\n    return np.linalg.pinv(A)\n\nQTRANS_CASES = {'p': [" + \
        ", ".join(f"({m!r},)" for m in mats) + "]}\n"
    f = tmp_path / "hard_pinv.py"
    f.write_text(src, encoding="utf-8")
    rep = verify_file(f, n_random=0, build_dir=tmp_path / "build")
    mpmath.mp.dps = 60
    for c in rep.functions[0].cases:
        truth = np.array((mpmath.matrix(c.args[0].tolist()) ** -1).tolist(), dtype=float)
        scale = np.abs(truth).max()
        cpp_err = np.abs(c.cpp[1] - truth).max() / scale
        np_err = np.abs(c.py[1] - truth).max() / scale
        assert cpp_err < 1e-13, (cpp_err, np_err)
