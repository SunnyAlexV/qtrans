"""Class hierarchies (0.13): a class may derive from one class of the same file, plain or
@dataclass, abstract (abc.ABC, @abstractmethod) or not. Objects of a hierarchy live behind a
std::shared_ptr, so a Payoff variable, parameter, field or list element can hold a Call or a Put,
a call goes to the object's own method (virtual), and assigning or passing one shares the object,
as Python does. What C++ can't do the same way is refused with the reason."""
import shutil
from pathlib import Path

import pytest

from qtrans import translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).parent / "data"
compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
HEAD = "from abc import ABC, abstractmethod\nfrom dataclasses import dataclass\n\n"


@compiled
@pytest.mark.parametrize("name", ["inheritance_payoffs.py", "inheritance_shapes.py", "inheritance_legs.py"])
def test_programs_and_functions_agree_with_python(tmp_path, name):
    """Abstract and concrete bases, super().__init__ and super().method(), inherited __init__ and
    fields, lists of mixed subclasses, isinstance, type(x).__name__, `is` between objects, a
    __repr__ that dispatches, dataclass reprs per class, objects changed through a function or
    a list (compared after the call, as the caller sees them), random objects of every concrete
    class as arguments."""
    f = tmp_path / name
    shutil.copy(DATA / name, f)
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert not rep.translation.report.rejected, rep.translation.report.to_text()
    assert rep.program is not None and rep.program.identical, rep.to_text()
    assert rep.ok and rep.warnings == [], rep.to_text()


def cpp(src):
    r = translate_source(HEAD + src, "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r.header


def rejection(src, which=None):
    r = translate_source(HEAD + src, "m")
    bad = [f for f in r.report.rejected if which is None or f.name == which]
    assert bad, f"expected a rejection: {[(f.name, f.translated) for f in r.report.functions]}"
    return bad[0].reason + " " + (bad[0].hint or "")


BASE = ("class Payoff(ABC):\n    def __init__(self, strike: float):\n        self.strike = strike\n\n"
        "    @abstractmethod\n    def value(self, spot: float) -> float:\n        ...\n\n"
        "class Call(Payoff):\n    def value(self, spot: float) -> float:\n        return max(spot - self.strike, 0.0)\n\n")


def test_the_cpp_reads_as_written_by_hand():
    h = cpp(BASE + "class Capped(Call):\n    def __init__(self, strike: float, cap: float):\n"
                   "        super().__init__(strike)\n        self.cap = cap\n\n"
                   "    def value(self, spot: float) -> float:\n        return min(super().value(spot), self.cap)\n\n"
                   "def price(p: Payoff, s: float) -> float:\n    return p.value(s)\n\n"
                   "def book(s: float) -> float:\n    xs: list[Payoff] = [Call(1.0), Capped(1.0, 2.0)]\n"
                   "    return sum((p.value(s) for p in xs), 0.0)\n")
    for want in ["struct Payoff {  // abstract class Payoff(ABC)", "virtual ~Payoff() = default;",
                 "virtual double value(double spot) const = 0;  // @abstractmethod",
                 "struct Call : public Payoff {", "using Payoff::Payoff;", "double value(double spot) const override;",
                 "struct Capped : public Call {", ": Call(strike_)", "return std::min(Call::value(spot), cap);",
                 "double price(const std::shared_ptr<Payoff>& p, double s)", "return p->value(s);",
                 "std::make_shared<Call>(1.0)", "std::vector<std::shared_ptr<Payoff>>", "#include <memory>"]:
        assert want in h, want


def test_a_dataclass_subclass_takes_the_base_fields_first():
    h = cpp("@dataclass\nclass A:\n    x: float\n\n    def f(self) -> float:\n        return self.x\n\n"
            "@dataclass\nclass B(A):\n    y: float = 2.0\n\n    def f(self) -> float:\n        return self.x + self.y\n")
    assert "explicit B(double x_, double y_ = 2.0);" in h and ": A(x_), y(y_) {}" in h
    assert "virtual double f() const;" in h and "double f() const override;" in h


def test_a_variable_given_two_subclasses_holds_their_common_base():
    h = cpp(BASE + "class Put(Payoff):\n    def value(self, spot: float) -> float:\n        return max(self.strike - spot, 0.0)\n\n"
                   "def f(up: bool, s: float) -> float:\n    if up:\n        p = Call(1.0)\n    else:\n        p = Put(1.0)\n"
                   "    return p.value(s)\n")
    assert "std::shared_ptr<Payoff> p;" in h


def test_isinstance_asks_the_object():
    h = cpp(BASE + "def f(p: Payoff) -> bool:\n    return isinstance(p, Call)\n\ndef g(c: Call) -> bool:\n"
                   "    return isinstance(c, Payoff)\n")
    assert "std::dynamic_pointer_cast<Call>(p) != nullptr" in h and "return true;" in h


def test_a_method_that_changes_self_in_an_override_makes_all_of_them_non_const():
    h = cpp("class A:\n    def __init__(self):\n        self.n = 0\n\n    def step(self) -> None:\n        pass\n\n"
            "class B(A):\n    def step(self) -> None:\n        self.n += 1\n")
    assert "virtual void step();" in h and "void step() override;" in h


@pytest.mark.parametrize("src, words", [
    (BASE + "def f() -> float:\n    return Payoff(1.0).value(2.0)\n", "abstract method value"),
    (BASE + "class Bad(Call):\n    def value(self, spot: int) -> float:\n        return 1.0\n",
     "a C++ override must keep them"),
    (BASE + "class Bad(Call):\n    def value(self, s: float) -> float:\n        return 1.0\n", "other parameter names"),
    ("class A:\n    def __init__(self):\n        self.x = 1.0\n\nclass B:\n    def __init__(self):\n        self.y = 1.0\n\n"
     "class C(A, B):\n    pass\n", "multiple inheritance"),
    ("class A:\n    def __init__(self):\n        self.x = 1.0\n\n@dataclass\nclass C(A):\n    y: float\n", "mixing the two"),
    ("class A:\n    def __init__(self, x: float):\n        self.x = x\n\nclass C(A):\n    def __init__(self):\n"
     "        self.y = 1.0\n", "doesn't call super().__init__"),
    ("class A:\n    def __init__(self, x: float):\n        self.x = x\n\nclass C(A):\n    def __init__(self):\n"
     "        self.y = 1.0\n        super().__init__(2.0)\n", "must be the first statement"),
    ("class A:\n    def __init__(self):\n        self.x = 1.0\n\n    def me(self) -> 'A':\n        return self\n\n"
     "class C(A):\n    pass\n", "passed on as a value"),
    ("class A:\n    def __init__(self):\n        self.x = 1.0\n\n    @classmethod\n    def make(cls) -> 'A':\n"
     "        return cls()\n\nclass C(A):\n    pass\n", "@classmethod in a class hierarchy"),
    ("@dataclass\nclass A:\n    x: float\n\n@dataclass\nclass C(A):\n    y: float = 1.0\n\n"
     "def f(a: A, b: A) -> bool:\n    return a == b\n", "exactly the same class"),
    (BASE + "class Capped(Call):\n    def __init__(self, strike: float, cap: float):\n        super().__init__(strike)\n"
            "        self.cap = cap\n\ndef f(p: Payoff) -> float:\n    return p.cap\n", "only its subclass Capped has it"),
    ("class A:\n    def __init__(self):\n        self.x = 1.0\n\n    @abstractmethod\n    def f(self) -> float:\n"
     "        ...\n", "doesn't derive from abc.ABC"),
    (BASE + "def f(xs: list[Payoff]) -> list[Payoff]:\n    return sorted(xs)\n", "can't add or compare them"),
    (BASE + "def bump(p: Payoff) -> float:\n    p.strike += 1.0\n    return p.strike\n\n"
            "def f(p: Payoff, q: Payoff) -> float:\n    return bump(p) + q.strike\n", "also reaches"),
])
def test_refusals(src, words):
    assert words in rejection(src)
