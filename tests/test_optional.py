"""Optional values (`x: float | None`, `-> float | None`, `lag: int | None = None`), as the
Black-Litterman app uses them. None crosses the verifier as its own value, Python's TypeError
on None + 1 must be reproduced, and None must be falsy."""
import textwrap

import pytest

from qtrans import translate_source, verify_file
from qtrans.verify import find_compiler, find_eigen

try:
    find_compiler()
    find_eigen()
    HAVE = True
except RuntimeError:
    HAVE = False

HEAD = "import math\nimport numpy as np\nfrom typing import Optional, Union\nfrom qtrans.typing import Vec, Mat\n"
BODY = textwrap.dedent('''
def newey_west_lag(T: int) -> int:
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

def hac_cov(Y: Mat, lag: int | None = None) -> Mat:
    T = Y.shape[0]
    L = newey_west_lag(T) if lag is None else int(lag)
    S = Y.T @ Y / T
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        G = Y[j:].T @ Y[:-j] / T
        S += w * (G + G.T)
    return S

def hit_rate_pvalue(hit_rate: Optional[float], n_independent: int) -> float | None:
    if hit_rate is None or not n_independent:
        return None
    z = (float(hit_rate) - 0.5) / math.sqrt(0.25 / float(n_independent))
    return 0.5 * math.erfc(z / math.sqrt(2))

def confidence(hit_rate: float | None, n_checks: int, shrink_k: float = 25.0) -> float:
    if hit_rate is None or n_checks <= 0:
        return 0.0
    raw = 2.0 * (float(hit_rate) - 0.5)
    if raw <= 0:
        return 0.0
    return float(np.clip(raw * (n_checks / (n_checks + shrink_k)), 0.0, 0.9))

def posterior_mu(pi: Vec, Sigma: Mat, P: Mat | None, Q: Vec, Omega: Mat, tau: float) -> Vec:
    if P is None or np.asarray(P).size == 0:
        return pi.copy()
    P = np.asarray(P, dtype=float)
    tauS = tau * Sigma
    A = P @ tauS @ P.T + Omega
    return pi + tauS @ P.T @ np.linalg.solve(A, Q - P @ pi)

def first_positive(x: Vec) -> float | None:
    found = None
    for v in x:
        if v > 0 and found is None:
            found = v
    return found

def use_result(x: Vec) -> float:
    r = first_positive(x)
    return -1.0 if r is None else r * 2.0

def none_arith(x: float | None) -> float:
    return x + 1.0  # TypeError in Python when x is None

def truthy(x: float | None) -> int:
    return 1 if x else 0

def default_none(a: float, b: float | None = None) -> float:
    return a if b is None else a * b

def call_defaults(a: float) -> float:
    return default_none(a) + default_none(a, 2.0) + default_none(a, None)

nan = float("nan")
QTRANS_CASES = {
    "hac_cov": [([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.0], [0.2, 0.2]], None),
                ([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.0], [0.2, 0.2]], 2), ([[0.1, 0.2]], 0)],
    "hit_rate_pvalue": [(None, 10), (0.6, 0), (0.6, 40), (0.5, 1)],
    "confidence": [(None, 10), (0.4, 10), (0.7, 12), (0.99, 10000), (0.7, 0)],
    "posterior_mu": [([0.05, 0.07], [[0.04, 0.01], [0.01, 0.09]], None, [0.1], [[0.001]], 0.05),
                     ([0.05, 0.07], [[0.04, 0.01], [0.01, 0.09]], [[1.0, -1.0]], [0.02], [[0.001]], 0.05),
                     ([0.05, 0.07], [[0.04, 0.01], [0.01, 0.09]], [[0.0, 0.0]], [0.02], [[0.0]], 0.05)],
    "first_positive": [([-1.0, -2.0],), ([-1.0, 0.5, 3.0],), ([],)],
    "none_arith": [(None,), (2.5,)],
    "truthy": [(None,), (0.0,), (-0.0,), (nan,), (3.0,)],
    "default_none": [(2.0,), (2.0, None), (2.0, 3.0)],
}
''')
SOURCE = HEAD + BODY
FUNCTIONS = [s.split("(")[0] for s in BODY.split("\ndef ")[1:]]
compiled = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("opt")
    f = d / "optionals.py"
    f.write_text(SOURCE)
    return verify_file(f, n_random=40, build_dir=d / "build")


def fn(report, name):
    return next(f for f in report.functions if f.name == name)


def given(report, name):
    return {c.args: c for c in fn(report, name).cases if c.source == "given"}


@compiled
def test_all_translate_without_warnings(report):
    assert not report.translation.report.rejected
    assert report.warnings == []


@compiled
@pytest.mark.parametrize("name", FUNCTIONS)
def test_matches_python(report, name):
    bad = [c for c in fn(report, name).cases if c.outcome == "mismatch"]
    assert not bad, [(c.args, c.py, c.cpp) for c in bad[:3]]


@compiled
def test_none_results_and_type_errors(report):
    hr = given(report, "hit_rate_pvalue")
    assert hr[(None, 10)].cpp == ("ok", None) and hr[(0.6, 0)].cpp == ("ok", None)
    na = given(report, "none_arith")
    assert na[(None,)].py[1] == "TypeError" and na[(None,)].cpp == ("exc", "TypeError")
    t = {k[0] if k[0] == k[0] else "nan": c.cpp for k, c in given(report, "truthy").items()}
    assert t[None] == t[0.0] == ("ok", 0) and t["nan"] == t[3.0] == ("ok", 1)


@compiled
def test_random_cases_include_none(report):
    """The verifier must exercise the None branch on random inputs too, not only given ones."""
    assert any(c.args[0] is None and c.source == "random" for c in fn(report, "confidence").cases)


def cpp(src):
    r = translate_source(HEAD + src, "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r.header


def rejection(src):
    r = translate_source(HEAD + src, "m")
    assert r.report.rejected, "expected a rejection"
    return r.report.rejected[0].reason


@pytest.mark.parametrize("ann", ["float | None", "None | float", "Optional[float]", "Union[float, None]"])
def test_all_spellings_of_optional(ann):
    h = cpp(f"def f(x: {ann}) -> float:\n    return 0.0 if x is None else x\n")
    assert "double f(std::optional<double> x)" in h and "!x.has_value()" in h


def test_optional_array_is_passed_by_const_reference():
    h = cpp("def f(P: Mat | None) -> float:\n    return 0.0 if P is None else float(np.sum(P))\n")
    assert "const std::optional<qt::Mat>& P" in h


def test_modifying_an_optional_array_is_refused():
    """std::optional<qt::Mat> holds its own copy; Python would modify the caller's array."""
    assert "None" in rejection("def f(P: Mat | None) -> float:\n    if P is None:\n        return 0.0\n"
                               "    P[0, 0] = 1.0\n    return 1.0\n")


def test_is_only_compares_with_none():
    assert "`is` is supported only" in rejection("def f(a: float, b: float) -> bool:\n    return a is b\n")


def test_a_plain_float_is_never_none():
    assert "return false;" in cpp("def f(a: float) -> bool:\n    return a is None\n")
