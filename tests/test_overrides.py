"""Your own C++ for a function (<module>_overrides.hpp): used in place of qtrans's translation,
checked against the Python and against qtrans's own translation on the same inputs, flagged
when the Python changes under it; and the speed hints written in the Python."""
import textwrap

import pytest

from qtrans import overrides as ovr
from qtrans.overrides import command
from qtrans.translator import translate_file
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

PRICING = '''
import math

RATE = 0.05


def discount(t: float) -> float:
    """Discount factor."""
    return math.exp(-RATE * t)


def annuity(n: int, scale: float = 1.0) -> float:
    total = 0.0
    for t in range(1, n + 1):
        total += discount(t)
    return total * scale


def doubled_sum(x: float) -> float:
    return sum(map(lambda t: t * x, [1.0, 2.0]))


def uses(n: int) -> float:
    return annuity(n) + doubled_sum(1.5)
'''

OVERRIDES = '''
// helpers of my own
#include <cmath>

// qtrans: override annuity  python={fp}
// the same sum, written my way
inline double annuity(long long count, double s) {{
    double total = 0.0;
    for (long long t = 1; t <= count; ++t) total += discount(static_cast<double>(t));
    return total * s;
}}
// qtrans: end annuity

// qtrans: override doubled_sum
inline double doubled_sum(double x) {{
    return x * 1.0 + x * 2.0;
}}
// qtrans: end doubled_sum
'''


@pytest.fixture
def module(tmp_path):
    py = tmp_path / "pricing.py"
    py.write_text(PRICING, encoding="utf-8")
    fp = ovr.fingerprint(PRICING, "annuity")
    (tmp_path / "pricing_overrides.hpp").write_text(OVERRIDES.format(fp=fp), encoding="utf-8")
    return py


def test_your_cpp_is_used_and_reported(module):
    r = translate_file(module)
    h = r.header
    assert "inline double annuity(long long count, double s) {" in h and "total * s;" in h
    assert "double annuity(long long n, double scale = 1.0);" in h  # the default, once, in the declaration
    assert h.index("#include <cmath>") < h.index("\n\n// from pricing_overrides.hpp") < h.index("namespace pricing {")
    assert "// the same sum, written my way" in h
    status = {f.name: f.override for f in r.report.functions}
    assert status == {"discount": None, "annuity": "up to date", "doubled_sum": "not yet accepted", "uses": None}
    # qtrans can't translate doubled_sum (a lambda); your C++ fills the gap, so uses() translates too
    assert all(f.translated for f in r.report.functions)
    assert "[your C++] annuity  (up to date)" in r.report.to_text()
    assert "qtrans can't translate this function itself" in next(
        f for f in r.report.functions if f.name == "doubled_sum").notes[0]


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_your_cpp_is_checked_against_python_and_qtrans_on_the_same_inputs(module, tmp_path):
    rep = verify_file(module, n_random=40, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    f = {x.name: x for x in rep.functions}
    a = f["annuity"]
    assert a.override is not None and a.reference is not None
    assert [c.args for c in a.cases] == [c.args for c in a.reference.cases]  # the very same inputs
    assert a.vs_reference == (40, 40, 0)  # bit-identical to qtrans's own translation
    assert f["doubled_sum"].reference is None  # qtrans has no translation of its own to compare
    text = rep.to_text()
    assert "your C++ against qtrans's, on the same inputs: 40 of 40 bit-identical" in text
    assert "qtrans can't translate this function itself" in text


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_wrong_cpp_is_caught_and_cant_be_accepted(module, tmp_path):
    p = tmp_path / "pricing_overrides.hpp"
    p.write_text(p.read_text().replace("return total * s;", "return total * s * 1.000001;"))
    rep = verify_file(module, n_random=30, build_dir=tmp_path / "build")
    a = next(x for x in rep.functions if x.name == "annuity")
    assert a.count("mismatch") > 0 and a.vs_reference[1] < a.vs_reference[0]
    said = []
    assert command(module, "annuity", accept=True, out=said.append) == 1
    assert "not accepted" in said[-1]


def test_the_python_changing_is_noticed_and_comments_are_not(module):
    module.write_text(PRICING.replace("    total = 0.0\n", "    total = 0.0  # running sum\n"))
    assert next(f for f in translate_file(module).report.functions if f.name == "annuity").override == "up to date"
    # RATE is read by discount(), which annuity() calls: the C++ was written for other numbers
    module.write_text(PRICING.replace("RATE = 0.05", "RATE = 0.04"))
    r = translate_file(module)
    a = next(f for f in r.report.functions if f.name == "annuity")
    assert a.override == "out of date" and a.translated  # still used, and flagged
    assert "OUT OF DATE" in a.notes[0]


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_strict_fails_until_accepted(module, tmp_path):
    module.write_text(PRICING.replace("RATE = 0.05", "RATE = 0.04"))
    rep = verify_file(module, n_random=20, build_dir=tmp_path / "b1", strict=True)
    assert not rep.ok and rep.stale_overrides == ["annuity", "doubled_sum"]
    assert "FAIL (--strict)" in rep.to_text()
    said = []
    assert command(module, "annuity", accept=True, out=said.append) == 0, said
    assert "accepted annuity()" in said[-1]
    of = ovr.load(module)
    assert of.blocks["annuity"].fingerprint == ovr.fingerprint(module.read_text(), "annuity")
    assert of.blocks["annuity"].accepted == ovr.today()


@pytest.mark.parametrize("change, words", [
    ("inline double annuity(long long count, double s)", "inline double annuity(double count, double s)"),
    ("inline double annuity(long long count, double s)", "inline float annuity(long long count, double s)"),
    ("inline double annuity(long long count, double s)", "inline double annuity(long long count)"),
])
def test_a_head_that_doesnt_fit_the_python_is_refused(module, tmp_path, change, words):
    p = tmp_path / "pricing_overrides.hpp"
    p.write_text(p.read_text().replace(change, words))
    r = translate_file(module)
    a = next(f for f in r.report.functions if f.name == "annuity")
    assert not a.translated and "doesn't fit the Python's signature" in a.reason
    assert a.hint == "its head must be: inline double annuity(long long n, double scale)"


@pytest.mark.parametrize("text, words", [
    ("// qtrans: override annuity\ninline double annuity(long long n, double s) { return 0.0; }\n",
     "has no `// qtrans: end annuity` line"),
    ("// qtrans: end annuity\n", "without its `// qtrans: override` line"),
    ("// qtrans: override a\nx\n// qtrans: end b\n", "closes the block for a()"),
    ("// qtrans: override nosuch\ninline double nosuch() { return 1.0; }\n// qtrans: end nosuch\n",
     "not a function of this module"),
])
def test_bad_blocks_are_reported_not_ignored(module, tmp_path, text, words):
    (tmp_path / "pricing_overrides.hpp").write_text(text)
    r = translate_file(module)
    assert any(words in n for n in r.report.module_notes), r.report.module_notes


def test_override_command_starts_from_the_translation_or_a_stub(tmp_path):
    py = tmp_path / "pricing.py"
    py.write_text(PRICING)
    said = []
    assert command(py, "annuity", out=said.append) == 0
    assert command(py, "doubled_sum", out=said.append) == 0
    of = ovr.load(py)
    a, d = of.blocks["annuity"], of.blocks["doubled_sum"]
    assert a.head == "inline double annuity(long long n, double scale)"
    assert "        total += discount(static_cast<double>(t));" in a.body  # qtrans's own translation, clean
    assert a.fingerprint == ovr.fingerprint(PRICING, "annuity")
    assert any("not written yet" in line for line in d.body) and any("lambda" in line for line in d.body)
    with pytest.raises(ovr.OverrideError, match="already has a block for annuity"):
        command(py, "annuity", out=said.append)  # a second block for it is refused
    listing = []
    command(py, show=True, out=listing.append)
    assert any("annuity" in x and "up to date" in x for x in listing)
    command(py, "doubled_sum", remove=True, out=said.append)
    assert "doubled_sum" not in ovr.load(py).blocks


def test_fingerprint_follows_meaning():
    base = "R = 1.0\n\ndef g(x: float) -> float:\n    return x * R\n\ndef f(x: float) -> float:\n    \"\"\"doc\"\"\"\n    return g(x) + 1.0\n"
    fp = ovr.fingerprint(base, "f")
    assert ovr.fingerprint(base.replace('"""doc"""', '"""other doc"""'), "f") == fp
    assert ovr.fingerprint(base.replace("    return g(x) + 1.0", "    return g(x) + 1.0  # note"), "f") == fp
    assert ovr.fingerprint(base.replace("R = 1.0", "R = 2.0"), "f") != fp      # a constant it reaches
    assert ovr.fingerprint(base.replace("x * R", "x * R * R"), "f") != fp     # a function it calls
    assert ovr.fingerprint(base.replace("+ 1.0", "+ 2.0"), "f") != fp


HINTS = '''
import math


def payoff(s: float, k: float) -> float:  # qtrans: inline
    return max(s - k, 0.0)


# qtrans: hot
def total(n: int, x: float) -> float:
    t = 0.0
    for i in range(n):  # qtrans: unroll 4
        if i > 1000000:  # qtrans: unlikely
            t -= 1.0
        t += payoff(x * i, 1.0)
    return t


def other(n: int) -> float:  # qtrans: noinline
    s = 0.0
    # qtrans: unroll 8
    while n > 0:
        s += math.sqrt(n)
        n -= 1
    return s


def bad(n: int) -> float:
    s = 0.0
    for i in range(n):  # qtrans: simd
        s += math.sqrt(i)
    return s  # qtrans: unrol 4
'''


def test_hints_become_portable_macros_and_unsafe_ones_are_refused(tmp_path):
    py = tmp_path / "hinted.py"
    py.write_text(HINTS)
    r = translate_file(py)
    h = r.header
    assert "#define QT_UNROLL(n) QT_PRAGMA_(GCC unroll n)" in h and "#elif defined(_MSC_VER)" in h
    assert "QT_FORCE_INLINE double payoff(double s, double k) {" in h
    assert "QT_HOT inline double total(long long n, double x) {" in h
    assert "QT_NOINLINE inline double other(long long n) {" in h
    assert "    QT_UNROLL(4)  // qtrans: unroll 4\n    for (long long i = 0; i < n; ++i) {" in h
    assert "QT_UNROLL(8)" in h and "if (QT_UNLIKELY(i > 1000000)) {" in h
    notes = "\n".join(r.report.module_notes)
    assert "`# qtrans: simd`: refused: SIMD reductions add in a different order" in notes
    assert "`# qtrans: unrol 4`: unknown qtrans hint" in notes
    assert all(f.translated for f in r.report.functions)  # a refused hint never blocks the function


@pytest.mark.parametrize("src, words", [
    ("def f(n: int) -> int:  # qtrans: unroll 4\n    return n\n", "doesn't apply to a def"),
    ("def f(n: int) -> int:\n    for i in range(n):  # qtrans: hot\n        pass\n    return n\n", "doesn't apply to a loop"),
    ("def f(n: int) -> int:\n    x = n  # qtrans: unroll 4\n    return x\n", "isn't on a loop, an if or a def"),
    ("def f(n: int) -> int:\n    for i in range(n):  # qtrans: unroll 100\n        pass\n    return n\n", "N from 2 to 64"),
    ("def f(n: int) -> int:  # qtrans: fast-math\n    return n\n", "results change"),
])
def test_misplaced_hints_say_so(tmp_path, src, words):
    py = tmp_path / "m.py"
    py.write_text(src)
    notes = "\n".join(translate_file(py).report.module_notes)
    assert words in notes, notes


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_hints_compile_cleanly_and_change_no_number(tmp_path):
    py = tmp_path / "hinted.py"
    py.write_text(HINTS)
    rep = verify_file(py, n_random=40, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert all(f.count("mismatch") == 0 and f.max_ulps == 0 for f in rep.functions)


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_hints_in_a_program_compile_too(tmp_path):
    py = tmp_path / "prog.py"
    py.write_text("def total(n: int) -> float:\n    t = 0.0\n    for i in range(n):  # qtrans: unroll 4\n"
                  "        t += i * 0.5\n    return t\n\nprint(total(10))\n")
    rep = verify_file(py, n_random=10, build_dir=tmp_path / "build")
    assert rep.program is not None and rep.program.identical and rep.warnings == [], rep.to_text()
    assert "#define QT_UNROLL" in translate_file(py).program
