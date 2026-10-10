"""Plain classes (fields from __init__, which becomes the constructor), a Generator held by an
object, and the evaluation-order rule that guards both: Python evaluates left to right, C++
leaves operand and argument order unspecified."""
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

ENGINE = textwrap.dedent('''import math
import numpy as np
from qtrans.typing import Vec


class GBMEngine:
    """Simulates geometric Brownian motion; owns its random generator."""

    def __init__(self, s0: float, mu: float, sigma: float, seed: int, steps_per_year: int = 252):
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = 1.0 / steps_per_year
        self.rng = np.random.default_rng(seed)
        self.paths_run = 0

    def terminal(self, n_paths: int, years: float) -> Vec:
        steps = max(1, int(round(years / self.dt)))
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        vol = self.sigma * math.sqrt(self.dt)
        logs = np.zeros(n_paths)
        for _ in range(steps):
            z = self.rng.standard_normal(n_paths)
            logs += drift + vol * z
        self.paths_run += n_paths
        return self.s0 * np.exp(logs)

    def call_price(self, K: float, years: float, n_paths: int) -> float:
        st = self.terminal(n_paths, years)
        return math.exp(-self.mu * years) * float(np.mean(np.maximum(st - K, 0.0)))


class Counter:
    def __init__(self):
        self.n = 0
        self.total = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x

    def mean(self) -> float:
        return self.total / self.n if self.n > 0 else 0.0


def two_prices(seed: int, K: float) -> tuple[float, float, int]:
    eng = GBMEngine(100.0, 0.03, 0.2, seed)
    a = eng.call_price(K, 1.0, 2000)
    b = eng.call_price(K, 1.0, 2000)
    return a, b, eng.paths_run


def price_with(eng: GBMEngine, K: float) -> float:
    return eng.call_price(K, 0.5, 1000)


def running_mean(xs: Vec) -> float:
    c = Counter()
    for x in xs:
        c.add(x)
    return c.mean()

QTRANS_CASES = {"two_prices": [(42, 100.0), (7, 120.0)], "price_with": [(GBMEngine(100.0, 0.03, 0.25, 3), 105.0)]}
''')


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    d = tmp_path_factory.mktemp("engine")
    f = d / "engine_mod.py"
    f.write_text(ENGINE, encoding="utf-8")
    return verify_file(f, n_random=60, build_dir=d / "build")


@compiled
def test_engine_module_matches(engine):
    assert not engine.translation.report.rejected
    assert engine.warnings == []
    assert engine.ok, engine.to_text()


@compiled
def test_the_generator_inside_ends_in_the_same_state(engine):
    """price_with(eng, K) draws from eng.rng: afterwards the caller's engine must be in the same
    state on both sides, generator included (compared by its next raw output)."""
    cases = [c for f in engine.functions if f.name == "price_with" for c in f.cases]
    assert cases and all(c.outcome in ("match", "known") for c in cases)
    given = next(c for c in cases if c.source == "given")
    after = given.cpp[1][1]
    assert after["paths_run"] == 1000 and isinstance(after["rng"], int)


HEAD = "import math\nimport numpy as np\nfrom dataclasses import dataclass\nfrom qtrans.typing import Vec\n\n"


def cpp(src):
    r = translate_source(HEAD + src, "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r.header


def rejection(src, which=None):
    r = translate_source(HEAD + src, "m")
    bad = [f for f in r.report.rejected if which is None or f.name == which]
    assert bad, f"expected a rejection: {[(f.name, f.translated) for f in r.report.functions]}"
    return bad[0].reason + " " + (bad[0].hint or "")


def test_init_becomes_the_constructor_and_types_the_fields():
    h = cpp("class C:\n    def __init__(self, w: Vec, scale: float = 1.0):\n        self.w = w * scale\n"
            "        self.n = len(w)\n        self.total = 0.0\n        self.ok = True\n")
    assert "qt::Vec w{};" in h and "long long n{};" in h and "double total{};" in h and "bool ok{};" in h
    assert "C(const qt::Vec& w_, double scale = 1.0);" in h and "n = w_.size();" in h


def test_drawing_from_a_held_generator_changes_the_object():
    h = cpp("class E:\n    def __init__(self, seed: int):\n        self.rng = np.random.default_rng(seed)\n\n"
            "    def draw(self) -> float:\n        return self.rng.random()\n\n"
            "def twice(e: E) -> float:\n    a = e.draw()\n    b = e.draw()\n    return a + b\n")
    assert "double draw();" in h and "double twice(E& e)" in h and "rng = qt::Rng(seed);" in h


def test_a_frozen_object_holding_a_generator_is_not_immutable():
    h = cpp("@dataclass(frozen=True)\nclass F:\n    rng: np.random.Generator\n\n    def draw(self) -> float:\n"
            "        return self.rng.random()\n\ndef g(f: F) -> float:\n    return f.draw()\n")
    assert "double g(F& f)" in h


@pytest.mark.parametrize("src,why", [
    ("class C:\n    def __init__(self):\n        self.best = None\n", "starts as None"),
    ("class C:\n    def __init__(self, x: float):\n        self.x = self.helper(x)\n\n"
     "    def helper(self, x: float) -> float:\n        return 2 * x\n", "annotate it"),
    ("class C:\n    def __init__(self, w: Vec):\n        self.w = w\n        self.w /= np.sum(self.w)\n",
     "changes field 'w' in place"),
    ("class C:\n    x: float = 1.0\n\n    def __init__(self):\n        self.y = 2.0\n", "shared by every object"),
    ("class Base:\n    def __init__(self):\n        self.x = 1.0\n\nclass C(Base):\n    def __init__(self):\n"
     "        self.y = 1.0\n", "inheritance"),
])
def test_plain_class_refusals(src, why):
    assert why in rejection(src, "class C")


def test_annotated_fields_resolve_what_init_cannot():
    h = cpp("class C:\n    best: float | None\n\n    def __init__(self):\n        self.best = None\n")
    assert "std::optional<double> best{};" in h and "best = std::nullopt;" in h


def test_plain_class_equality_is_identity_in_python():
    assert "same object" in rejection("class C:\n    def __init__(self, x: float):\n        self.x = x\n\n"
                                      "def f(a: C, b: C) -> bool:\n    return a == b\n", "f")


COUNTER = ("class P:\n    def __init__(self, cash: float):\n        self.cash = cash\n\n"
           "    def step(self) -> float:\n        self.cash *= 1.01\n        return self.cash\n\n"
           "def push(a: Vec) -> float:\n    a[0] += 1.0\n    return a[0]\n\n")


@pytest.mark.parametrize("body,why", [
    ("def f(p: P) -> float:\n    return p.step() + p.cash\n", "also reads 'p'"),
    ("def f(a: Vec) -> float:\n    return push(a) + a[0]\n", "also reads 'a'"),
    ("def f(p: P) -> float:\n    return p.step() - p.step()\n", "two calls that change something"),
    ("def f(p: P) -> None:\n    p.cash += p.step()\n", "also reads 'p'"),
    ("def f(g: np.random.Generator) -> float:\n    return g.random() - g.random()\n", "two random draws"),
])
def test_evaluation_order_hazards_are_refused(body, why):
    """Each of these gives a different number in C++ when the compiler picks the other order."""
    assert why in rejection(COUNTER + body, "f")


def test_one_change_per_statement_is_fine():
    h = cpp(COUNTER + "def f(p: P) -> float:\n    x = p.step()\n    total = 0.0\n    total += p.step()\n"
                      "    p.cash = p.step()\n    return x + total + p.cash\n")
    assert "p.cash = p.step();" in h


def test_passing_a_generator_into_a_constructor_is_refused():
    assert "share it with the caller" in rejection(
        "class E:\n    def __init__(self, g: np.random.Generator):\n        self.rng = g\n\n"
        "def f(g: np.random.Generator) -> float:\n    e = E(g)\n    return 1.0\n", "f")
