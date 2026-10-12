"""Unit tests on the generated C++ text and on what gets rejected. No compiler needed."""
import pytest

from qtrans import translate_source


def cpp(src):
    r = translate_source(src, "m")
    bad = r.report.rejected
    assert not bad, [(f.name, f.reason) for f in bad]
    return r.header


def rejection(src):
    r = translate_source(src, "m")
    assert r.report.rejected, "expected a rejection"
    return r.report.rejected[0].reason


# --- readable, exact output ----------------------------------------------------------
def test_power_two_keeps_the_rounding_of_python():
    h = cpp("def f(a: float, s: float) -> float:\n    return a * s ** 2\n")
    # a * s * s would round differently from Python's a * (s ** 2)
    assert "return a * (s * s);" in h


def test_true_division_of_ints_is_float():
    h = cpp("def f(a: int, b: int) -> float:\n    return a / b\n")
    assert "static_cast<double>(a) / static_cast<double>(b)" in h


def test_floor_division_and_mod_use_python_semantics():
    h = cpp("def f(a: int, b: int) -> int:\n    return a // b + a % b\n")
    # the sum of two unknown ints could leave 64 bits: checked
    assert "qt::iadd(qt::floordiv(a, b), qt::mod(a, b))" in h


def test_int_literals_promoted_where_overflow_possible():
    h = cpp("def f() -> int:\n    return 100000 * 100000\n")
    assert "100000LL * 100000" in h


def test_variable_widened_to_double():
    r = translate_source("def f(n: int) -> float:\n    t = 0\n    for i in range(n):\n        t += 0.5\n    return t\n", "m")
    assert "double t = 0.0;" in r.header
    assert any("holds an int" in n for n in r.report.functions[0].notes)


def test_declared_before_branch_when_first_assigned_inside_it():
    h = cpp("def f(x: float) -> float:\n    if x > 0:\n        y = 1.0\n    else:\n        y = -1.0\n    return y\n")
    assert h.index("double y = 0.0;") < h.index("if (x > 0)")


def test_comments_and_docstrings_carried_over():
    h = cpp('# section header\ndef f(x: float) -> float:\n    """Doubles x."""\n    # explain\n    return 2 * x  # trailing\n')
    assert "// section header" in h and "/// Doubles x." in h
    assert "// explain" in h and "return 2 * x;  // trailing" in h


def test_declared_in_innermost_block():
    h = cpp("def f(n: int) -> float:\n    t = 0.0\n    for k in range(n):\n"
            "        pv = 1.0 / (k + 1)\n        t += pv\n    return t\n")
    assert "        double pv = 1.0 / static_cast<double>(k + 1);" in h


def test_loop_carried_variable_declared_outside_loop():
    h = cpp("def f(n: int) -> int:\n    t = 0\n    for k in range(n):\n        if k % 2 == 0:\n"
            "            last = k\n        t += last\n    return t\n")
    assert h.index("long long last = 0;") < h.index("for (long long k")


def test_tuple_return_and_structured_binding():
    h = cpp("def g(x: float) -> tuple[float, float]:\n    return x, 2 * x\n\n"
            "def f(x: float) -> float:\n    a, b = g(x)\n    return a + b\n")
    assert "std::tuple<double, double> g(double x)" in h
    assert "return {x, 2 * x};" in h and "auto [a, b] = g(x);" in h


def test_swap_goes_through_a_temporary():
    h = cpp("def f(x: float, y: float) -> float:\n    a = x\n    b = y\n    a, b = b, a\n    return a - b\n")
    assert "std::tie(a, b) = std::make_tuple(b, a);" in h


def test_underscore_target_is_ignored():
    h = cpp("def g(x: float) -> tuple[float, float]:\n    return x, x\n\n"
            "def f(x: float) -> float:\n    a, _ = g(x)\n    return a\n")
    assert "std::tie(a, std::ignore) = g(x);" in h


def test_keyword_arguments_fill_skipped_defaults():
    h = cpp("def g(a: float, b: float = 2.0, c: float = 3.0) -> float:\n    return a + b + c\n\n"
            "def f(x: float) -> float:\n    return g(x, c=5.0) + g(x)\n")
    assert "g(x, 2.0, 5.0) + g(x)" in h


def test_cpp_keywords_are_renamed():
    h = cpp("def f(new: float, default: float) -> float:\n    std = new + default\n    return std\n")
    assert "double f(double new_, double default_)" in h and "double std_ = new_ + default_;" in h


def test_math_floor_returns_int_numpy_floor_float():
    h = cpp("import math\nimport numpy as np\n\ndef f(x: float) -> float:\n"
            "    return math.floor(x) + np.floor(x)\n")
    assert "static_cast<double>(qt::to_int(std::floor(x))) + std::floor(x)" in h


def test_round_is_half_to_even():
    h = cpp("def f(x: float) -> int:\n    return round(x)\n")
    assert "std::nearbyint(x)" in h


def test_range_bound_hoisted_when_body_changes_it():
    h = cpp("def f(n: int) -> int:\n    t = 0\n    for i in range(n):\n        n -= 1\n        t += 1\n    return t\n")
    assert "const long long i_end = n;" in h and "i < i_end" in h


def test_negative_step_loop():
    h = cpp("def f(n: int) -> int:\n    t = 0\n    for k in range(n, 0, -2):\n        t += k\n    return t\n")
    assert "for (long long k = n; k > 0; k -= 2)" in h


def test_chained_comparison():
    h = cpp("def f(a: float, b: float, c: float) -> bool:\n    return a < b <= c\n")
    assert "return a < b && b <= c;" in h


def test_norm_with_loc_and_scale():
    h = cpp("from scipy.stats import norm\n\ndef f(x: float, m: float, s: float) -> float:\n"
            "    return norm.cdf(x, m, s) + norm.ppf(0.9, loc=m, scale=s)\n")
    assert "qt::norm_cdf(x, m, s)" in h and "qt::norm_ppf(0.9, m, s)" in h


def test_norm_loc_only_stays_inline():
    h = cpp("from scipy.stats import norm\n\ndef f(x: float, m: float) -> float:\n"
            "    return norm.cdf(x, m) + norm.ppf(0.1, m)\n")
    assert "qt::norm_cdf(x - m) + (qt::norm_ppf(0.1) + m)" in h


def test_includes_only_what_is_used():
    h = cpp("def f(x: float) -> float:\n    return x + 1\n")
    assert "#include <cmath>" in h and "qtrans_runtime" not in h and "<tuple>" not in h


# --- refusals: every one names the line and, where possible, a fix ----------------------
@pytest.mark.parametrize("src, expected", [
    ("def f(x):\n    return x\n", "no type annotation"),
    ("def f(x: float):\n    if x > 0:\n        return 1\n    return 2.5\n", "no return annotation"),
    ("def f(x: list) -> float:\n    return 0.0\n", "type annotation 'list'"),
    ("def f(x: float) -> float:\n    print(f'{x!a}')\n    return x\n", "the !a conversion"),
    ("def f(n: int) -> int:\n    for i in range(n):\n        pass\n    return i\n", "outside its for-loop"),
    ("def f(n: int) -> int:\n    for i in range(n):\n        i = 3\n    return n\n", "assigning to loop variable"),
    ("def f(x: float) -> float:\n    if x > 0:\n        return x\n", "can reach its end"),
    ("def f(a: int, b: int) -> int:\n    return a ** b\n", "non-constant exponent"),
    ("def f(a: float, b: float) -> bool:\n    return a and b\n", "`and`/`or` on non-bool"),
    ("def f(n: int) -> int:\n    n = n / 2\n    return n\n", "declared int but gets a float"),
    ("def f(x: float) -> float:\n    return [x][0]\n", "not supported"),
    ("def f(x: float) -> float:\n    try:\n        return 1.0 / x\n    except ZeroDivisionError:\n        return 0.0\n",
     "catching ZeroDivisionError is not supported: x / 0.0 raises it in Python but gives inf in C++"),
    ("def f(x: float) -> float:\n    raise ValueError('bad', x)\n", "several arguments"),
    ("from qtrans.typing import Vec\ndef f(v: Vec) -> float:\n    s = 0.0\n    for e in v:\n        s += e\n    return round(s, 2)\n",
     "a Python float or a NumPy float64"),
    ("def f(x: float) -> float:\n    for i in range(x):\n        pass\n    return x\n", "must be ints"),
])
def test_rejections(src, expected):
    assert expected in rejection(src)


def test_rejection_propagates_to_callers():
    r = translate_source("def bad(x: float) -> float:\n    return [x][0]\n\n"
                         "def good(x: float) -> float:\n    return bad(x) + 1\n", "m")
    reasons = {f.name: f.reason for f in r.report.rejected}
    assert "calls bad()" in reasons["good"]
    assert "good" not in r.header and "bad" not in r.header.split("namespace")[1]
