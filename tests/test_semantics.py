"""Compile the generated C++ and check it computes what Python computes, on exactly the
inputs where C++ and Python disagree by default. Needs a C++ compiler."""
import shutil
import textwrap

import pytest

from qtrans import verify_file
from qtrans.verify import find_compiler

try:
    find_compiler()
    HAVE_CXX = True
except RuntimeError:
    HAVE_CXX = False

pytestmark = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")

EDGE_FLOATS = [0.0, -0.0, 0.1, -0.1, 0.5, 1.5, 2.5, -2.5, 3.0, -7.5, 1e-300, 1e300, 7.0, -7.0]
EDGE_INTS = [0, 1, -1, 2, -2, 3, -3, 7, -7, 10, -10]

SOURCE = textwrap.dedent('''
    import math
    import numpy as np
    from scipy.stats import norm

    def int_floordiv(a: int, b: int) -> int:
        return a // b

    def int_mod(a: int, b: int) -> int:
        return a % b

    def float_floordiv(a: float, b: float) -> float:
        return a // b

    def float_mod(a: float, b: float) -> float:
        return a % b

    def true_div(a: int, b: int) -> float:
        return a / b

    def rounding(x: float) -> int:
        return round(x)

    def huge_round(x: float) -> int:
        return round(x)

    def truncation(x: float) -> int:
        return int(x)

    def floors(x: float) -> tuple[int, float, int]:
        return math.floor(x), np.floor(x), math.ceil(x)

    def int_powers(a: int) -> tuple[int, float, int]:
        return a ** 3, a ** -2 if a != 0 else 0.0, a ** 0

    def mixed_minmax(a: int, b: float) -> tuple[float, float, int]:
        return max(a, b), min(a, b, 0.5), max(a, 3, -2)

    def np_extremes(a: float, b: float) -> tuple[float, float]:
        return np.maximum(a, b), np.minimum(a, b)

    def truthiness(x: float, n: int) -> tuple[bool, bool, bool]:
        return bool(x), not n, (x if n else -x) > 0

    def chained(a: float, b: float, c: float) -> bool:
        return a <= b < c

    def collatz_steps(n: int) -> int:
        """while, break, continue and int arithmetic together."""
        if n <= 0:
            raise ValueError("n must be positive")
        steps = 0
        while True:
            if n == 1:
                break
            steps += 1
            if n % 2 == 0:
                n //= 2
                continue
            n = 3 * n + 1
        return steps

    def stepped_sum(start: int, stop: int) -> int:
        total = 0
        for k in range(start, stop, 3):
            total += k
        for k in range(stop, start, -2):
            total -= k
        return total

    def carried_partial(n: int) -> int:
        """`last` is only assigned on some iterations; the others reuse the previous value."""
        total = 0
        for k in range(n):
            if k % 3 == 0:
                last = k
            total += last
        return total

    def carried_prev(n: int) -> int:
        """`prev` is read before it is assigned within an iteration: loop-carried."""
        acc = 0
        for k in range(n):
            if k > 0:
                acc += k * prev
            prev = k + 1
        return acc

    def fresh_each_iteration(n: int) -> float:
        """`x` is assigned before every read: may live inside the loop."""
        s = 0.0
        k = 0
        while k < n:
            x = 0.5 * k
            if x > 2:
                x = x - 1
            s += x
            k += 1
        return s

    def elif_spanning(x: float) -> float:
        if x < 0:
            y = -x
        elif x < 1:
            y = x * x
        else:
            y = 1.0
        return y

    def ppf_roundtrip(p: float) -> float:
        return norm.ppf(p)

    def norm_family(x: float, m: float, s: float) -> tuple[float, float, float, float]:
        return norm.cdf(x, m, s), norm.pdf(x, loc=m, scale=s), norm.sf(x), norm.ppf(0.25, m, s)

    def int_float_mix(n: int, x: float) -> tuple[float, float]:
        """An int on the left of / or ** with a float: must be cast explicitly (-Wconversion),
        found by translating the Black-Litterman app's newey_west_lag."""
        return n / 100.0 + x, n ** (2.0 / 9.0) + x ** 0.5

    QTRANS_CASES = {
        "int_floordiv": [(a, b) for a in INTS for b in INTS],
        "int_mod": [(a, b) for a in INTS for b in INTS],
        "float_floordiv": [(a, b) for a in FLOATS for b in FLOATS] + [(1.0, 0.1), (-1.0, 0.1)],
        "float_mod": [(a, b) for a in FLOATS for b in FLOATS],
        "true_div": [(a, b) for a in INTS for b in INTS],
        "rounding": [(x,) for x in FLOATS[:-4] + [0.5, -0.5, 4.5, 5.5, 1e15 + 0.5, 1e18]],
        "huge_round": [(1e300,), (float("inf"),), (float("nan"),), (-1e19,)],
        "truncation": [(x,) for x in FLOATS[:-4] + [-2.7, 2.7]],
        "floors": [(x,) for x in FLOATS[:-4] + [-2.7, 2.7]],
        "int_powers": [(a,) for a in INTS],
        "mixed_minmax": [(a, b) for a in INTS for b in FLOATS],
        "np_extremes": [(a, b) for a in FLOATS for b in FLOATS] + [(float("nan"), 1.0), (1.0, float("nan"))],
        "truthiness": [(x, n) for x in FLOATS for n in (0, 1)],
        "chained": [(1.0, 1.0, 2.0), (1.0, 2.0, 2.0), (2.0, 1.0, 3.0)],
        "collatz_steps": [(n,) for n in (1, 2, 3, 7, 27, 97, 871, 0)],
        "stepped_sum": [(a, b) for a in (-5, 0, 4) for b in (-6, 0, 11, 20)],
        "carried_partial": [(n,) for n in (1, 2, 4, 7, 10)],
        "carried_prev": [(n,) for n in (1, 2, 5, 9)],
        "fresh_each_iteration": [(n,) for n in (0, 3, 8)],
        "elif_spanning": [(x,) for x in (-2.0, 0.5, 3.0)],
        "ppf_roundtrip": [(p,) for p in (1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 0.001, 0.02425, 0.1, 0.25,
                                         0.5, 0.75, 0.9, 0.97575, 0.999, 1 - 1e-10, 0.0, 1.0)],
        "int_float_mix": [(n, x) for n in (0, 1, 7, 250, -8) for x in (0.0, 2.0, -4.0)],
        "norm_family": [(x, m, s) for x in (-3.0, 0.0, 1.5) for m in (0.0, 0.7)
                        for s in (0.5, 2.0, 0.0, -1.0, float("nan"))],
    }
''').replace("INTS", repr(EDGE_INTS)).replace("FLOATS", repr(EDGE_FLOATS))


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("sem")
    f = d / "semantics.py"
    f.write_text(SOURCE)
    return verify_file(f, n_random=50, build_dir=d / "build")


def test_everything_translates(report):
    assert not report.translation.report.rejected


def test_float_to_int_never_undefined(report):
    """Python: big int / OverflowError / ValueError. C++: OverflowError / OverflowError /
    ValueError. NaN and inf agree; a finite value beyond 64 bits is the documented
    limitation and raises rather than returning garbage."""
    f = next(f for f in report.functions if f.name == "huge_round")
    got = {c.args[0] if c.args[0] == c.args[0] else "nan": c.cpp for c in f.cases if c.source == "given"}
    assert got[1e300] == ("exc", "OverflowError") and got[-1e19] == ("exc", "OverflowError")
    assert got[float("inf")] == ("exc", "OverflowError") and got["nan"] == ("exc", "ValueError")


def test_no_compiler_warnings(report):
    assert report.warnings == []


@pytest.mark.parametrize("name", ["int_floordiv", "int_mod", "float_floordiv", "float_mod", "true_div",
                                  "rounding", "truncation", "floors", "int_powers", "mixed_minmax",
                                  "np_extremes", "truthiness", "chained", "collatz_steps", "stepped_sum",
                                  "ppf_roundtrip", "norm_family", "carried_partial", "carried_prev",
                                  "fresh_each_iteration", "elif_spanning", "int_float_mix"])
def test_matches_python(report, name):
    f = next(f for f in report.functions if f.name == name)
    bad = [c for c in f.cases if c.outcome == "mismatch"]
    assert not bad, [(c.args, c.py, c.cpp) for c in bad[:5]]


def test_integer_arithmetic_is_bit_exact(report):
    for name in ("int_floordiv", "int_mod", "float_floordiv", "float_mod", "rounding", "collatz_steps"):
        f = next(f for f in report.functions if f.name == name)
        assert f.max_ulps == 0, name


def test_division_by_zero_raises_like_python(report):
    f = next(f for f in report.functions if f.name == "int_floordiv")
    zero = [c for c in f.cases if c.args[1] == 0]
    assert zero and all(c.cpp == ("exc", "ArithmeticError") for c in zero)


def test_norm_ppf_matches_scipy_closely(report):
    f = next(f for f in report.functions if f.name == "ppf_roundtrip")
    assert f.max_rel_err < 1e-13


def test_negative_base_fractional_power_is_known_not_mismatch(report):
    """(-8) ** (2/9): a Python float goes complex, C (like np.float64) gives NaN. The verifier
    must call that a known float difference, and both paths must be exercised."""
    f = next(f for f in report.functions if f.name == "int_float_mix")
    given = [c for c in f.cases if c.source == "given"]
    assert any(c.outcome == "known" and c.args[0] < 0 for c in given)
    assert all(c.outcome == "match" for c in given if c.args[0] >= 0 and c.args[1] >= 0)
