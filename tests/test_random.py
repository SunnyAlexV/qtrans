"""Stage 3: NumPy's random generator reproduced in C++, and the rules that keep translated
simulations drawing the same numbers in the same order."""
import textwrap

import numpy as np
import pytest

from qtrans import translate_source, verify_file
from tests.test_array_semantics import HAVE

HEAD = "import numpy as np\nfrom qtrans.typing import Vec, Mat\n\n"


def cpp(src):
    r = translate_source(HEAD + src, "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r.header


def rejection(src):
    r = translate_source(HEAD + src, "m")
    assert r.report.rejected, "expected a rejection"
    return r.report.rejected[0].reason


# --- translation --------------------------------------------------------------------------
def test_generator_created_from_seed():
    h = cpp("def f(seed: int) -> float:\n    rng = np.random.default_rng(seed)\n    return rng.standard_normal()\n")
    assert "qt::Rng rng(seed);" in h and "rng.standard_normal()" in h and '#include "qtrans_random.hpp"' in h


def test_generator_parameter_passed_by_reference():
    h = cpp("def f(n: int, rng: np.random.Generator) -> Vec:\n    return rng.normal(1.0, 2.0, n)\n")
    assert "qt::Vec f(long long n, qt::Rng& rng)" in h and "rng.normal(1.0, 2.0, n)" in h


def test_sizes_and_defaults():
    h = cpp("def f(rng: np.random.Generator) -> Mat:\n    return rng.uniform(size=(3, 4))\n")
    assert "rng.uniform(0.0, 1.0, 3, 4)" in h


@pytest.mark.parametrize("src, expected", [
    ("def f() -> float:\n    rng = np.random.default_rng()\n    return rng.random()\n", "exactly one argument"),
    ("def f() -> float:\n    np.random.seed(1)\n    return np.random.normal()\n", "legacy global random state"),
    ("def f(rng: np.random.Generator) -> float:\n    return rng.random() - rng.random()\n", "two random draws"),
    ("def f(rng: np.random.Generator) -> float:\n    g = rng\n    return g.random()\n", "only be created with"),
    ("def f(rng: np.random.Generator) -> float:\n    return rng.exponential()\n", "not supported in stage 3"),
    ("def f(rng: np.random.Generator) -> np.random.Generator:\n    return rng\n", "returning a generator"),
])
def test_random_rejections(src, expected):
    assert expected in rejection(src)


# --- compiled: the C++ generator is NumPy's generator ----------------------------------------
SOURCE = textwrap.dedent('''
    import numpy as np
    from qtrans.typing import Vec, Mat

    def normals(seed: int, n: int) -> Vec:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)

    def mixed(seed: int, n: int) -> tuple[Vec, Mat, float, float]:
        rng = np.random.default_rng(seed)
        u = rng.uniform(-3.0, 5.0, n)
        m = rng.normal(2.0, 0.5, (n, 3))
        x = rng.random()
        y = rng.normal()
        return u, m, x, y

    def bad_scale(seed: int, scale: float) -> float:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, scale)

    def bad_range(seed: int, low: float, high: float) -> float:
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high)

    def walk(n: int, rng: np.random.Generator) -> float:
        """Uses a data-dependent number of draws; the caller's generator must end up in the same place."""
        x = 0.0
        k = 0
        while k < n and x < 3.0:
            x += rng.random()
            k += 1
        return x

    def pair(a: float, b: float) -> float:
        return a - 2.0 * b

    def ordered(rng: np.random.Generator) -> float:
        a = rng.random()
        b = rng.random()
        return pair(a, b)

    QTRANS_CASES = {
        "normals": [(s, n) for s in (0, 1, 42, 2 ** 40 + 3, 2 ** 63 - 1) for n in (0, 1, 1000, 100000)] + [(-1, 3)],
        "mixed": [(s, 50) for s in (0, 7, 123456789)],
        "bad_scale": [(1, 1.0), (1, -0.0), (1, -1.0), (1, float("nan"))],
        "bad_range": [(1, 0.0, 1.0), (1, 1.0, 0.0), (1, -1e308, 1e308), (1, 2.0, 2.0)],
        "walk": [(n, s) for n in (0, 1, 5, 100) for s in (0, 1, 2)],
        "ordered": [(s,) for s in range(5)],
    }
''')

pytestmark_compiled = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("rng")
    f = d / "random_sem.py"
    f.write_text(SOURCE)
    return verify_file(f, n_random=30, build_dir=d / "build")


def fn(report, name):
    return next(f for f in report.functions if f.name == name)


@pytestmark_compiled
@pytest.mark.parametrize("name", ["normals", "mixed", "bad_scale", "bad_range", "walk", "ordered"])
def test_random_matches_numpy(report, name):
    bad = [c for c in fn(report, name).cases if c.outcome != "match"]
    assert not bad, [(c.args, c.py, c.cpp) for c in bad[:3]]


@pytestmark_compiled
def test_draws_are_bit_identical(report):
    for name in ("normals", "mixed", "walk", "ordered"):
        for c in fn(report, name).cases:
            if c.py[0] == "ok":
                assert c.rel_err == 0.0 and c.ulps == 0, (name, c.args)


@pytestmark_compiled
def test_errors_match_numpy(report):
    by_args = {c.args: c for c in fn(report, "bad_scale").cases if c.source == "given"}
    assert by_args[(1, -1.0)].cpp == ("exc", "ValueError")      # scale < 0
    assert by_args[(1, -0.0)].cpp == ("exc", "ValueError")      # NumPy checks the sign bit
    rng_cases = {c.args: c for c in fn(report, "bad_range").cases if c.source == "given"}
    assert rng_cases[(1, 1.0, 0.0)].cpp == ("exc", "ValueError")  # high - low < 0
    assert rng_cases[(1, -1e308, 1e308)].py[1] == "OverflowError" == rng_cases[(1, -1e308, 1e308)].cpp[1]
    neg = next(c for c in fn(report, "normals").cases if c.args == (-1, 3))
    assert neg.py[1] == "ValueError" and neg.cpp[1] == "ValueError"


@pytestmark_compiled
def test_monte_carlo_price_is_right(tmp_path):
    """Independent of the translation: the example's antithetic estimator lands within four
    standard errors of the Black-Scholes price, in C++ as in Python."""
    from pathlib import Path
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "examples")]
    from black_scholes import bs_price
    r = verify_file(Path(__file__).resolve().parent.parent / "examples" / "monte_carlo.py",
                    n_random=0, build_dir=tmp_path)
    cases = [c for c in next(f for f in r.functions if f.name == "european_mc").cases
             if c.cpp[0] == "ok" and c.args[5] >= 1000]
    assert cases
    for c in cases:
        S0, K, rr, sigma, T, n, seed, is_call = c.args
        price, se = c.cpp[1]
        assert abs(price - bs_price(S0, K, rr, sigma, T, 0.0, is_call)) < 4 * se
