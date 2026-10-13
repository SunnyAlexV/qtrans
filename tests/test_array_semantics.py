"""Stage 2, compiled: NumPy's behaviour on exactly the inputs where a naive translation
would differ. Needs a C++ compiler and Eigen."""
import textwrap

import numpy as np
import pytest

from qtrans import verify_file
from qtrans.verify import find_compiler, find_eigen

try:
    find_compiler()
    find_eigen()
    HAVE = True
except RuntimeError:
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")

SOURCE = textwrap.dedent('''
    import numpy as np
    from qtrans.typing import Vec, Mat

    def total(a: Vec) -> float:
        return np.sum(a)

    def matrix_sums(A: Mat) -> tuple[float, Vec, Vec]:
        return np.sum(A), A.sum(axis=0), A.sum(axis=1)

    def moments(a: Vec) -> tuple[float, float, float, float]:
        return np.mean(a), np.var(a), np.std(a, ddof=1), np.median(a)

    def quantiles(a: Vec, q: float) -> tuple[float, float]:
        return np.quantile(a, q), np.percentile(a, 100 * q)

    def grid(a: float, b: float, n: int) -> Vec:
        return np.linspace(a, b, n)

    def steps(a: float, b: float, h: float) -> Vec:
        return np.arange(a, b, h)

    def running(a: Vec) -> tuple[Vec, Vec]:
        return np.cumsum(a), np.cumprod(a)

    def extremes(a: Vec) -> tuple[float, float, int, int]:
        return np.max(a), a.min(), np.argmax(a), a.argmin()

    def ordered(a: Vec) -> Vec:
        return np.sort(a)

    def slice_any(a: Vec, i: int, j: int, k: int) -> Vec:
        return a[i:j:k]

    def slice_open(a: Vec, i: int) -> tuple[Vec, Vec]:
        return a[i:], a[:i]

    def pick(a: Vec, i: int) -> float:
        return a[i]

    def fill_slice(a: Vec, i: int, j: int, v: float) -> None:
        a[i:j] = v

    def shift_in_place(V: Vec) -> None:
        V[1:] = V[:-1] + 1.0

    def shrink(S: Vec, d: float) -> Vec:
        S = S[:-1] / d
        return S

    def reverse_add(a: Vec) -> None:
        a += a[::-1]

    def ufuncs(a: Vec) -> tuple[Vec, Vec, Vec, Vec]:
        return np.maximum(a, 0.0), np.where(a > 1.0, a, -a), a ** 2 + a ** -1, np.abs(a) ** 0.5

    def broadcast(A: Mat, v: Vec) -> tuple[Mat, Mat]:
        return A - v, v / A

    def products(A: Mat, v: Vec) -> tuple[Vec, Vec, float, Mat]:
        return A @ v, v @ A, v @ v, A @ A.T

    def solve(A: Mat, b: Vec) -> Vec:
        return np.linalg.solve(A, b)

    def factor(A: Mat) -> Mat:
        return np.linalg.cholesky(A)

    _r = np.random.default_rng(5)
    _spd = (lambda X: X @ X.T + 3 * np.eye(3))(_r.normal(size=(3, 3)))
    _nan = np.array([1.0, np.nan, 3.0, np.nan])
    _ties = np.array([2.0, 5.0, 5.0, -1.0, -1.0])

    QTRANS_CASES = {
        "total": [(_r.normal(size=n) * 10.0 ** _r.uniform(-3, 3, n),) for n in (0, 1, 7, 8, 9, 16, 127, 128, 129, 1000, 4097)],
        "matrix_sums": [(_r.normal(size=s),) for s in ((1, 1), (5, 1), (40, 1), (1, 9), (20, 30), (200, 3))],
        "moments": [(_r.normal(size=n) * 7 + 3,) for n in (2, 3, 10, 129, 999)] + [(_nan,)],
        "quantiles": [(_r.normal(size=n), q) for n in (1, 2, 7, 100) for q in (0.0, 0.01, 0.5, 0.75, 0.99, 1.0)]
                     + [(_nan, 0.5), (np.array([]), 0.5), (np.array([1.0, 2.0]), 1.5)],
        "grid": [(0.0, 1.0, n) for n in (0, 1, 2, 3, 50, 101)] + [(-3.3, 7.7, 13), (2.0, 2.0, 5), (0.0, 1.0, -1)],
        "steps": [(0.0, 1.0, 0.1), (0.0, 10.0, 1.0), (1.0, 0.0, -0.25), (0.0, 1.0, 3.0), (0.0, 1.0, 0.0), (5.0, 1.0, 1.0)],
        "running": [(_r.uniform(0.9, 1.1, 50),), (np.array([]),)],
        "extremes": [(_ties,), (_nan,), (np.array([]),), (np.array([-0.0, 0.0]),)],
        "ordered": [(_nan,), (_ties,), (np.array([]),)],
        "slice_any": [(np.arange(10.0), i, j, k) for i in (-12, -3, 0, 2, 9, 15) for j in (-12, -1, 0, 5, 20)
                      for k in (-3, -1, 1, 2)] + [(np.arange(5.0), 0, 5, 0)],
        "slice_open": [(np.arange(6.0), i) for i in (-9, -2, 0, 3, 6, 9)],
        "pick": [(np.arange(5.0), i) for i in (-6, -5, -1, 0, 4, 5)],
        "fill_slice": [(np.arange(8.0), i, j, 9.5) for i, j in ((2, 5), (-3, 100), (5, 2), (-100, 1))],
        "shift_in_place": [(np.array([0.0, 10.0, 20.0, 30.0, 40.0]),), (np.array([]),)],
        "shrink": [(np.array([1.0, 2.0, 4.0, 8.0, 16.0]), 2.0)],
        "reverse_add": [(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),)],
        "ufuncs": [(np.array([-2.0, -0.0, 0.5, 1.0, 3.0, np.nan, np.inf]),)],
        "broadcast": [(_r.normal(size=(4, 3)), _r.normal(size=3)), (_r.normal(size=(4, 3)), _r.normal(size=4))],
        "products": [(_r.normal(size=(4, 4)), _r.normal(size=4)), (_r.normal(size=(3, 4)), _r.normal(size=4))],
        "solve": [(_spd, np.ones(3)), (np.ones((3, 3)), np.ones(3)), (_r.normal(size=(2, 3)), np.ones(2))],
        "factor": [(_spd,), (-_spd,), (np.zeros((0, 0)),)],
    }
''')

BIT_EXACT = ["total", "matrix_sums", "moments", "quantiles", "grid", "steps", "running", "extremes",
             "ordered", "slice_any", "slice_open", "pick", "fill_slice", "shift_in_place", "broadcast"]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("arr")
    f = d / "arrays_sem.py"
    f.write_text(SOURCE, encoding="utf-8")
    return verify_file(f, n_random=40, build_dir=d / "build")


def fn(report, name):
    return next(f for f in report.functions if f.name == name)


def test_all_translate_without_warnings(report):
    assert not report.translation.report.rejected
    assert report.warnings == []


FUNCTIONS = [s.split("(")[0] for s in SOURCE.split("\ndef ")[1:]]


@pytest.mark.parametrize("name", FUNCTIONS)
def test_matches_numpy(report, name):
    bad = [c for c in fn(report, name).cases if c.outcome == "mismatch"]
    assert not bad, [(c.args, c.py, c.cpp) for c in bad[:3]]


@pytest.mark.parametrize("name", BIT_EXACT)
def test_numpy_algorithms_reproduced_bit_for_bit(report, name):
    """Pairwise summation, NumPy's var/quantile/linspace/arange formulas, Python slicing:
    the hand-picked cases must agree to the last bit."""
    for c in fn(report, name).cases:
        if c.source == "given" and c.py[0] == "ok":
            assert c.ulps == 0 and c.rel_err == 0.0, (name, c.args)


def test_out_of_range_index_raises_like_numpy(report):
    cases = {c.args[1]: c for c in fn(report, "pick").cases if c.source == "given"}
    for i in (-6, 5):
        assert cases[i].py[1] == "IndexError" and cases[i].cpp[1] in ("IndexError", "ShapeError")
    assert cases[-1].cpp == ("ok", 4.0) and cases[-5].cpp == ("ok", 0.0)


def test_in_place_changes_reach_the_caller(report):
    c = next(c for c in fn(report, "shift_in_place").cases if c.source == "given" and len(c.args[0]) == 5)
    # NumPy reads all old values first: [0, 10, 20, 30, 40] -> [0, 1, 11, 21, 31]. Reading while
    # writing would cascade to [0, 1, 2, 3, 4].
    assert np.array_equal(c.py[1][0], np.array([0.0, 1, 11, 21, 31]))
    assert np.array_equal(c.cpp[1][0], c.py[1][0])


def test_self_referencing_updates_read_old_values(report):
    rev = next(c for c in fn(report, "reverse_add").cases if c.source == "given")
    assert np.array_equal(rev.cpp[1][0], np.array([6.0, 6, 6, 6, 6]))  # not [6, 6, 6, 12, 18]
    sh = next(c for c in fn(report, "shrink").cases if c.source == "given")
    assert np.array_equal(sh.cpp[1], np.array([0.5, 1.0, 2.0, 4.0]))


def test_linalg_errors_match(report):
    singular = next(c for c in fn(report, "solve").cases if c.source == "given" and c.args[0].shape == (3, 3)
                    and np.all(c.args[0] == 1.0))
    assert singular.py[1] == "LinAlgError" and singular.cpp[1] == "LinAlgError"  # its own class now, a ValueError
    not_pd = [c for c in fn(report, "factor").cases if c.source == "given"][1]
    assert not_pd.py[1] == "LinAlgError" and not_pd.cpp[1] == "LinAlgError"
