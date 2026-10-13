"""Lists of floats, ints and strings as std::vector: what they compute, what they print, and
the sharing a C++ copy can't keep, refused. Programs are run both ways (Python and the compiled
C++) and their output compared character for character; functions taking and returning lists
are checked by qtrans verify on random lists, empty ones included."""
import textwrap

import pytest

from qtrans.translator import translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

PROGRAMS = {
    "basics": '''
        prices = [100.0, 101.5, 99.25, 102.0]
        returns = []
        for i in range(1, len(prices)):
            returns.append(prices[i] / prices[i - 1] - 1)
        print(returns)
        print(len(returns), sum(returns), min(prices), max(prices), sorted(prices, reverse=True))
        sq = [r * r for r in returns if r > 0]
        print(sq, prices[-1], prices[1:3], prices[::-1], prices[-2:], prices[:10])
        names = ["call", "put"]
        for n in names:
            print(n, n in names, "fwd" in names, repr(n))
        counts = [1, 2, 3]
        counts[0] += 5
        counts.append(7)
        counts.insert(0, -1)
        print(counts, sum(counts), counts.index(7), counts.count(2), list(range(4)))
        last = counts.pop()
        first = counts.pop(0)
        print(last, counts, first, [5, 6, 7].pop(1))
        both = counts + [10, 20]
        both.extend([30])
        both += [40]
        both.sort(reverse=True)
        print(both, sorted(names), [len(x) for x in names], [])
        ''',
    "sums": '''
        empty: list[float] = []
        tiny = []
        for i in range(3):
            tiny.append(0.1)
        print(sum(empty + [0.0]), sum([0.1, 0.2, 0.3]), sum(tiny), sum([0.5] * 3), [1.5] * 2, 2 * [7])
        xs = []
        n = 0
        for i in range(n):
            xs.append(i * 0.5)
        print(sum(xs), [x + 1 for x in xs])
        s = sum(xs)
        print(s, f"{s:.2f}")
        big = [1e16, 1.0, -1e16]
        print(sum(big), sum([0.1] + [0.2]))
        ''',
    "numpy_elements": '''
        import numpy as np

        v = np.array([1.5, 2.0, 3.25])
        xs = [v[0], v[2]]
        ys = [float(x) for x in v]
        print(xs, ys, sum(ys), max(xs), xs[0], repr(xs[0]))
        ''',
    "errors": '''
        xs = [1.0, 2.0]
        print(xs[1])
        print(xs[5])
        ''',
}


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
@pytest.mark.parametrize("name", list(PROGRAMS))
def test_list_programs_print_what_python_prints(tmp_path, name):
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(PROGRAMS[name]).lstrip(), encoding="utf-8")
    rep = verify_file(f, n_random=10, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    p = rep.program
    assert p.identical, rep.to_text() + "\n--- Python:\n" + p.py_out[-3000:] + "\n--- C++:\n" + p.cpp_out[-3000:]
    assert rep.warnings == [] and p.py_code == p.cpp_code == (1 if name == "errors" else 0)


FUNCTIONS = '''
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def cumulative(xs: list[float]) -> list[float]:
    out = []
    total = 0.0
    for x in xs:
        total += x
        out.append(total)
    return out


def top(xs: list[int], k: int) -> list[int]:
    return sorted(xs, reverse=True)[:k]


def labels(n: int) -> list[str]:
    return [f"p{i}" for i in range(n)]


RATES = [0.01, 0.02, 0.05]


def discounted(t: float) -> list[float]:
    return [1.0 / (1.0 + r) ** t for r in RATES]


def last_or(xs: list[float], d: float) -> float:
    if len(xs) == 0:
        return d
    return xs[-1]
'''


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_functions_on_lists_agree_with_python(tmp_path):
    f = tmp_path / "lists_mod.py"
    f.write_text(FUNCTIONS, encoding="utf-8")
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert {fc.name for fc in rep.functions} == {"mean", "cumulative", "top", "labels", "discounted", "last_or"}
    assert all(fc.count("mismatch") == 0 and len(fc.cases) == 60 for fc in rep.functions)


def test_lists_read_like_cpp():
    h = translate_source(FUNCTIONS, "m").header
    assert "inline const std::vector<double> RATES = {0.01, 0.02, 0.05};" in h
    assert "inline std::vector<double> cumulative(const std::vector<double>& xs) {" in h
    assert "    for (double x : xs) {" in h and "        out.push_back(total);" in h
    assert "return qt::py::slice(qt::py::sorted(xs, true), qt::py::none, k);" in h
    assert "return qt::py::item(xs, -1LL);" in h


@pytest.mark.parametrize("src, line, words", [
    ("a = [1.0]\nb = a\nb.append(2.0)\nprint(a)\n", 2, "one list in Python"),
    ("def f(xs: list[float]) -> None:\n    xs.append(1.0)\n\nv = [2.0]\nf(v)\nprint(v)\n", 2, "changes the caller's list"),
    ("xs = [1.0, 2.0]\nfor x in xs:\n    xs.append(x)\n", 2, "while going through it"),
    ("xs = [1, 2.5]\nprint(xs)\n", 1, "several types"),
    ("xs = [1.0]\nxs.append(2)\nprint(xs)\n", 2, "keeps each element's type"),
    ("R = [1.0]\n\ndef f() -> float:\n    return R[0]\n\nR.append(2.0)\nprint(f())\n", 6, "module's list"),
    ("xs = [[1.0], [2.0]]\nprint(xs)\n", 1, "list of list[float]"),
    ("xs = [1.0, 2.0]\nprint(xs, xs.pop())\n", 2, "changes 'xs', and also reads 'xs'"),
    ("xs = []\nprint(len(xs))\n", 2, "xs: list[float] = []"),
    ("def f(xs: list[float]) -> list[float]:\n    return xs\n\nprint(f([1.0]))\n", 2, "the same object"),
])
def test_sharing_and_mixed_types_are_refused_at_their_line(src, line, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    assert bad, r.report.to_text()
    first = bad[0]
    assert first.reason_line == line and (words in first.reason or words in (first.hint or "")), first.reason


def test_a_copy_breaks_the_sharing():
    r = translate_source("a = [1.0]\nb = list(a)\nb.append(2.0)\nc = a.copy()\nprint(a, b, c)\n", "m", "m.py")
    assert r.report.functions[0].translated


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_sum_from_a_start(tmp_path):
    f = tmp_path / "sums2.py"
    f.write_text(textwrap.dedent("""
        def total(xs: list[float], base: float) -> float:
            return sum(xs, base)


        def count(ns: list[int]) -> int:
            return sum(ns, 10)


        empty: list[float] = []
        print(sum(empty, 0.0), sum([0.1, 0.2, 0.3], 0.0), sum([1e16, 1.0, -1e16], 0.5))
        print(sum((x * 2 for x in [0.1, 0.7]), 0.0), sum([0.1, 0.2], 0), total([0.1] * 10, 1.0), count([1, 2]))
        """).lstrip())
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.ok and rep.program.identical and rep.warnings == [], rep.to_text()
