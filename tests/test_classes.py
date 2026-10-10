"""Stage 5: @dataclass and NamedTuple as C++ structs, and strings. Two modules are compiled
and checked against Python (an option class; curves, bonds, a mutable portfolio inside an
account), and every refusal is pinned down with its reason."""
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
compiled = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")

OPTION = textwrap.dedent('''import math
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Option:
    """A European option."""
    S: float
    K: float
    T: float
    vol: float
    r: float = 0.0
    is_call: bool = True

    def __post_init__(self):
        if self.vol <= 0:
            raise ValueError("vol must be positive")

    @property
    def forward(self) -> float:
        return self.S * math.exp(self.r * self.T)

    def d1(self) -> float:
        return (math.log(self.forward / self.K) + 0.5 * self.vol ** 2 * self.T) / (self.vol * math.sqrt(self.T))

    def price(self) -> float:
        d1 = self.d1()
        d2 = d1 - self.vol * math.sqrt(self.T)
        df = math.exp(-self.r * self.T)
        if self.is_call:
            return df * (self.forward * norm.cdf(d1) - self.K * norm.cdf(d2))
        return df * (self.K * norm.cdf(-d2) - self.forward * norm.cdf(-d1))

    @classmethod
    def atm(cls, S: float, T: float, vol: float) -> "Option":
        return cls(S, S, T, vol)


def book_value(o: Option, qty: float) -> float:
    return qty * o.price()


def bumped(o: Option, h: float) -> float:
    up = replace(o, S=o.S + h)
    dn = replace(o, S=o.S - h)
    return (up.price() - dn.price()) / (2 * h)


def make(S: float, K: float) -> Option:
    return Option(S=S, K=K, T=0.5, vol=0.2, is_call=False)


def same(a: Option, b: Option) -> bool:
    return a == b

QTRANS_CASES = {"book_value": [(Option(100.0, 95.0, 0.5, 0.2, 0.01), 3.0)],
                "make": [(100.0, 105.0)], "same": [(Option(1.0, 1.0, 1.0, 0.1), Option(1.0, 1.0, 1.0, 0.1)),
                                                   (Option(1.0, 1.0, 1.0, 0.1), Option(1.0, 1.0, 1.0, 0.1, is_call=False))]}
''')
BOOK = textwrap.dedent('''import copy
import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from qtrans.typing import Vec


class Quote(NamedTuple):
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class Curve:
    """Zero rates at pillar times, linearly interpolated, flat outside."""
    times: Vec
    rates: Vec

    def __post_init__(self):
        if len(self.times) != len(self.rates):
            raise ValueError("times and rates differ in length")
        if len(self.times) == 0:
            raise ValueError("empty curve")

    def zero(self, t: float) -> float:
        ts = self.times
        rs = self.rates
        if t <= ts[0]:
            return rs[0]
        if t >= ts[-1]:
            return rs[-1]
        for i in range(1, len(ts)):
            if t <= ts[i]:
                w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
                return rs[i - 1] + w * (rs[i] - rs[i - 1])
        return rs[-1]

    def df(self, t: float) -> float:
        return math.exp(-self.zero(t) * t)


@dataclass(frozen=True)
class Bond:
    face: float
    coupon: float
    maturity: float
    freq: int = 2
    n_periods: int = field(init=False, default=0)

    def __post_init__(self):
        if self.freq <= 0 or self.maturity <= 0 or self.maturity > 100:
            raise ValueError("bad bond")
        object.__setattr__(self, "n_periods", int(round(self.maturity * self.freq)))

    def price(self, curve: Curve) -> float:
        c = self.face * self.coupon / self.freq
        pv = 0.0
        for k in range(1, self.n_periods + 1):
            pv += c * curve.df(k / self.freq)
        return pv + self.face * curve.df(self.n_periods / self.freq)


@dataclass
class Portfolio:
    weights: Vec
    cash: float = 0.0
    trades: int = field(default=0, init=False)

    def value(self, prices: Vec) -> float:
        return float(self.weights @ prices) + self.cash

    def rebalance(self, target: Vec, prices: Vec, cost_bps: float) -> float:
        turnover = float(np.sum(np.abs(target - self.weights)))
        cost = turnover * cost_bps * 1e-4 * self.value(prices)
        self.weights = target.copy()
        self.cash -= cost
        self.trades += 1
        return cost

    def scale(self, k: float) -> None:
        self.weights *= k


@dataclass
class Account:
    port: Portfolio
    name: str = "main"

    def nav(self, prices: Vec) -> float:
        return self.port.value(prices)


def quote_stats(q: Quote) -> tuple[float, float, float, float]:
    bid, ask = q
    return q.mid, q.spread(), q[0] - bid, ask


def bond_price(b: Bond, curve: Curve) -> float:
    return b.price(curve)


def shifted(curve: Curve, h: float) -> Curve:
    return Curve(curve.times, curve.rates + h)


def run_rebalances(p: Portfolio, target: Vec, prices: Vec, n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += p.rebalance(target, prices, 5.0)
    return total


def scaled(p: Portfolio, k: float) -> Portfolio:
    q = copy.deepcopy(p)
    q.scale(k)
    return q


def renamed(acc: Account, name: str) -> Account:
    a = copy.deepcopy(acc)
    a.name = name
    return a


def nav(acc: Account, prices: Vec) -> float:
    return acc.nav(prices)


def repriced(b: Bond, coupon: float, curve: Curve) -> float:
    return replace(b, coupon=coupon).price(curve)


_T = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
_R = np.array([0.030, 0.032, 0.035, 0.038, 0.040])
_CURVE = Curve(_T, _R)
_W = np.array([0.4, 0.3, 0.2, 0.1])
_PX = np.array([101.0, 55.5, 230.0, 12.25])
QTRANS_CASES = {
    "quote_stats": [(Quote(99.5, 100.25),)],
    "bond_price": [(Bond(100.0, 0.05, 7.0), _CURVE), (Bond(100.0, 0.0, 0.5, 1), _CURVE),
                   (Bond(1000.0, 0.04, 30.0, 4), _CURVE)],
    "shifted": [(_CURVE, 0.001)],
    "run_rebalances": [(Portfolio(_W.copy(), 1000.0), np.full(4, 0.25), _PX, 3), (Portfolio(_W.copy()), _W, _PX, 0)],
    "scaled": [(Portfolio(_W.copy(), 5.0), 2.0)],
    "renamed": [(Account(Portfolio(_W.copy())), "hedge book")],
    "nav": [(Account(Portfolio(_W.copy(), 50.0)), _PX)],
    "repriced": [(Bond(100.0, 0.05, 7.0), 0.06, _CURVE)],
}
''')
STRINGS = textwrap.dedent('''LONG = "Long only"

def stance_bounds(stance: str, cap: float, n: int) -> tuple[float, float, float]:
    if stance == "Long only":
        return 0.0, cap, 1.0
    if stance == "Short only":
        return -cap, 0.0, -1.0
    return -cap, cap, 1.0

def is_known(stance: str = LONG) -> bool:
    return stance in ("Long only", "Short only", "Long-short")

def label(x: float) -> str:
    tag = "pos" if x > 0 else "neg"
    return "value: " + tag

def pair(x: float) -> tuple[float, str]:
    if not label(x):
        return 0.0, ""
    return x, label(x)

QTRANS_CASES = {"label": [(1.0,), (-0.0,)], "is_known": [("Long only",), ("long only",), ("",), ("Long-short",), ()],
                "stance_bounds": [("Short only", 0.3, 5), ("é ü 你好", 0.3, 5)]}
''')


def _verify(tmp_path_factory, src, name):
    d = tmp_path_factory.mktemp(name)
    f = d / f"{name}.py"
    f.write_text(src)
    return verify_file(f, n_random=60, build_dir=d / "build")


@pytest.fixture(scope="module")
def option(tmp_path_factory):
    return _verify(tmp_path_factory, OPTION, "options_mod")


@pytest.fixture(scope="module")
def book(tmp_path_factory):
    return _verify(tmp_path_factory, BOOK, "book_mod")


@pytest.fixture(scope="module")
def strings(tmp_path_factory):
    return _verify(tmp_path_factory, STRINGS, "strings_mod")


@compiled
@pytest.mark.parametrize("which", ["option", "book", "strings"])
def test_module_translates_and_matches(which, request):
    r = request.getfixturevalue(which)
    assert not r.translation.report.rejected, [(f.name, f.reason) for f in r.translation.report.rejected]
    assert r.warnings == []
    assert r.ok, r.to_text()


def _given(report, name):
    return [c for c in next(f for f in report.functions if f.name == name).cases if c.source == "given"]


@compiled
def test_changes_made_by_a_method_reach_the_caller(book):
    """run_rebalances(p, ...) changes p through p.rebalance(): the caller's object must end up
    the same in both (weights, cash and the trade count)."""
    c = _given(book, "run_rebalances")[0]
    assert c.outcome == "match"
    final = c.cpp[1][1]
    assert final["trades"] == 3 and list(final["weights"]) == [0.25] * 4


@compiled
def test_post_init_runs_in_the_constructor(book, option):
    b = _given(book, "bond_price")
    assert all(c.outcome == "match" for c in b)
    # random cases the Python constructor refuses (vol <= 0, bad bonds) are redrawn, not counted
    assert all(c.py[0] == "ok" or c.py[1] != "ValueError" or "vol" not in c.py[2]
               for f in option.functions for c in f.cases)


@compiled
def test_equality_and_strings(option, strings):
    same, differ = _given(option, "same")
    assert same.cpp == ("ok", True) and differ.cpp == ("ok", False)
    s = {c.args: c.cpp for c in _given(strings, "stance_bounds")}
    assert s[("Short only", 0.3, 5)] == ("ok", (-0.3, 0.0, -1.0))


# --- generated code ---------------------------------------------------------------------------
HEAD = ("import copy\nimport numpy as np\nfrom dataclasses import dataclass, field, replace\n"
        "from typing import NamedTuple\nfrom qtrans.typing import Vec, Mat\n\n")
PORT = ("@dataclass\nclass Port:\n    w: Vec\n    cash: float = 0.0\n\n"
        "    def scale(self, k: float) -> None:\n        self.w *= k\n\n"
        "    def value(self) -> float:\n        return float(np.sum(self.w)) + self.cash\n\n")


def cpp(src):
    r = translate_source(HEAD + src, "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r.header


def rejection(src, which=None):
    r = translate_source(HEAD + src, "m")
    bad = [f for f in r.report.rejected if which is None or f.name == which]
    assert bad, f"expected a rejection: {[f.name for f in r.report.functions]}"
    return bad[0].reason + " " + (bad[0].hint or "")


def test_const_methods_and_passing_modes():
    h = cpp(PORT + "def grow(p: Port) -> None:\n    p.scale(1.1)\n\n"
                   "def worth(p: Port) -> float:\n    return p.value()\n")
    assert "void scale(double k);" in h and "double value() const;" in h
    assert "void grow(Port& p)" in h and "double worth(const Port& p)" in h


def test_frozen_objects_are_never_copied():
    h = cpp("@dataclass(frozen=True)\nclass P:\n    x: float\n\ndef f(p: P) -> float:\n    return p.x\n")
    assert "double f(const P& p)" in h


def test_a_frozen_object_holding_an_array_can_still_change_inside():
    """frozen only stops reassigning fields; o.w[0] = 1 is legal Python and changes the caller's
    array, so the parameter must be a reference."""
    h = cpp("@dataclass(frozen=True)\nclass C:\n    w: Vec\n\ndef f(c: C) -> None:\n    c.w[0] = 1.0\n")
    assert "void f(C& c)" in h


@pytest.mark.parametrize("body,why", [
    ("def f(w: Vec) -> float:\n    p = Port(w)\n    p.scale(2.0)\n    return float(w[0])\n", "is part of 'p'"),
    ("def f(p: Port) -> float:\n    q = p\n    q.scale(2.0)\n    return p.value()\n", "same object"),
    ("def f(p: Port) -> float:\n    w = p.w\n    w[0] = 5.0\n    return p.value()\n", "is part of 'p'"),
    ("def f(p: Port) -> float:\n    q = replace(p, cash=1.0)\n    q.scale(2.0)\n    return p.value()\n",
     "same object"),
])
def test_sharing_that_c_plus_plus_would_break_is_refused(body, why):
    assert why in rejection(PORT + body, "f")


def test_deep_copy_breaks_the_sharing():
    h = cpp(PORT + "def f(p: Port) -> float:\n    q = copy.deepcopy(p)\n    q.scale(2.0)\n    return p.value()\n")
    assert "Port q = Port(p);" in h


@pytest.mark.parametrize("src,why", [
    ("@dataclass(frozen=True)\nclass P:\n    x: float\n\ndef f(p: P) -> None:\n    p.x = 1.0\n",
     "frozen dataclass"),
    ("@dataclass\nclass P:\n    w: Vec\n\n    def __post_init__(self):\n        self.w /= np.sum(self.w)\n",
     "changes field 'w' in place"),
    ("@dataclass\nclass P:\n    w: Vec = np.zeros(3)\n", "default_factory"),
    ("class Base:\n    pass\n\n@dataclass\nclass P(Base):\n    x: float\n", "inheritance"),
    ("class P:\n    def __init__(self, x: float):\n        self.x = x\n", "plain class"),
    ("@dataclass\nclass P:\n    rng: np.random.Generator\n", "Generator inside an object"),
    ("@dataclass\nclass P:\n    x: float\n\n    @property\n    def y(self) -> float:\n        return self.x\n\n"
     "    @y.setter\n    def y(self, v: float) -> None:\n        self.x = v\n", "setters"),
    ("@dataclass\nclass P:\n    x: float\n\n    def __eq__(self, o) -> bool:\n        return True\n",
     "changes what Python operators mean"),
])
def test_class_refusals(src, why):
    assert why in rejection(src, "f" if "def f(" in src else "class P")


def test_equality_of_objects_holding_arrays_is_refused():
    assert "holds an array" in rejection(PORT + "def f(a: Port, b: Port) -> bool:\n    return a == b\n", "f")


def test_replace_cannot_set_an_init_false_field():
    src = ("@dataclass\nclass P:\n    x: float\n    y: float = field(init=False, default=0.0)\n\n"
           "def f(p: P) -> P:\n    return replace(p, y=1.0)\n")
    assert "init=False" in rejection(src, "f")


def test_shallow_copy_is_refused():
    assert "shallow" in rejection(PORT + "def f(p: Port) -> Port:\n    return copy.copy(p)\n", "f")


def test_namedtuple_indexing_and_unpacking():
    h = cpp("class Q(NamedTuple):\n    bid: float\n    ask: float\n\n"
            "def f(q: Q) -> float:\n    a = 0.0\n    b = 0.0\n    if q.bid > 0:\n        a, b = q\n"
            "    return a + b + q[1]\n")
    assert "qtrans_unpacked" in h and "q.ask" in h


def test_strings_in_tuples_and_in():
    h = cpp("def f(s: str) -> bool:\n    return s in ('a', 'b')\n\ndef g(x: float) -> tuple[float, str]:\n"
            "    return x, 'ok' if x > 0 else 'no'\n")
    assert 's == "a" || s == "b"' in h and "std::tuple<double, std::string>" in h


def test_string_number_comparison_is_refused():
    assert "never equal" in rejection("def f(s: str) -> bool:\n    return s == 1\n")
