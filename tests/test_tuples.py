"""Lists of tuples (std::vector<std::tuple<...>>), `for a, b in pairs:`, and zip / enumerate /
d.items() read to the end by list() / sorted() / dict() / min() / max(). Run both ways and the
output compared character for character; functions checked on random lists of tuples."""
import textwrap

import pytest

from qtrans.translator import translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

PROGRAM = '''
trades = [("AAPL", 190.5, 10), ("MSFT", 410.25, 3)]
trades.append(("GOOG", 140.0, 7))
for name, price, qty in trades:
    print(f"{name:<5} {price:>8.2f} x {qty}")
print(trades[0], trades[-1][1], len(trades), sorted(trades), max(trades), min(trades))
pairs = [(1.0, 2.0), (3.0, 0.5), (1.0, 1.0)]
print(sorted(pairs), sorted(pairs, reverse=True), [a * b for a, b in pairs], {str(a): b for a, b in pairs})
prices = {"b": 2.0, "a": 1.0}
print(list(zip(["x", "y"], [0.5, 0.25])), sorted(prices.items()), list(prices.items()), list(enumerate([7, 8], 1)))
for k, v in sorted(prices.items()):
    print(k, v)
print(dict(zip(["a", "b", "a"], [1.0, 2.0, 3.0])), dict(sorted(prices.items())), max(zip([1, 2], [5, 3])))
for i, (a, b) in enumerate(pairs):
    print(i, a + b)
'''

FUNCTIONS = '''
def notional(trades: list[tuple[str, float, int]]) -> float:
    total = 0.0
    for name, price, qty in trades:
        total += price * qty
    return total


def pnl(fills: list[tuple[float, float]]) -> list[float]:
    return [(sell - buy) * 100.0 for buy, sell in fills]


def best(quotes: list[tuple[float, str]]) -> tuple[float, str]:
    return max(quotes)


def ranked(xs: list[float]) -> list[tuple[int, float]]:
    return sorted(enumerate(xs), reverse=True)
'''


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_tuple_program_prints_what_python_prints(tmp_path):
    f = tmp_path / "tuples.py"
    f.write_text(textwrap.dedent(PROGRAM).lstrip())
    rep = verify_file(f, n_random=5, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    assert rep.program.identical and rep.warnings == [], rep.to_text() + rep.program.py_out + rep.program.cpp_out


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_tuple_functions_agree_with_python(tmp_path):
    f = tmp_path / "tuples_mod.py"
    f.write_text(FUNCTIONS)
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert {fc.name for fc in rep.functions} == {"notional", "pnl", "best", "ranked"}
    assert all(fc.count("mismatch") == 0 and len(fc.cases) == 60 for fc in rep.functions)


def test_tuples_read_like_cpp():
    h = translate_source(FUNCTIONS, "m").header
    assert "inline double notional(const std::vector<std::tuple<std::string, double, long long>>& trades) {" in h
    assert "auto [name, price, qty] = trades[static_cast<std::size_t>(i)];" in h
    assert "return qt::py::sorted(qt::py::enumerate(xs), true);" in h


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_ordering_tuples_with_nan_is_refused_at_run_time(tmp_path):
    f = tmp_path / "nan.py"
    f.write_text("def top(xs: list[tuple[float, float]]) -> tuple[float, float]:\n    return max(xs)\n\n"
                 "print(top([(float('nan'), 1.0), (float('nan'), 2.0)]))\n")
    r = translate_source(f.read_text(), "nan", str(f))
    assert r.program is not None
    rep = verify_file(f, n_random=0, build_dir=tmp_path / "build")
    assert "ordering a list with NaN in it is not reproduced" in rep.program.cpp_err


@pytest.mark.parametrize("src, words", [
    ("pairs = [(1.0, 2.0)]\nfor a, b in pairs:\n    pairs.append((b, a))\n", "changes the length of 'pairs'"),
    ("z = zip([1.0], [2.0])\nprint(list(z))\n", "zip"),
    ("xs = [(1.0, [2.0])]\nprint(xs)\n", "needs a number, but got a list[float]"),
])
def test_tuple_refusals(src, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    assert bad and words in bad[0].reason, r.report.to_text()


KEYS = '''
def by_abs(xs: list[float]) -> list[float]:
    return sorted(xs, key=abs)


def by_second(ps: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return sorted(ps, key=lambda p: p[1], reverse=True)


def biggest(d: dict[str, float]) -> str:
    return max(d, key=d.get)


def shortest(ws: list[str]) -> str:
    return min(ws, key=len, default="")


def score(x: float) -> float:
    return (x - 1.0) * (x - 1.0)


def closest(xs: list[float]) -> float:
    return min(xs, key=score, default=0.0)


def sort_in_place(xs: list[int]) -> list[int]:
    ys = list(xs)
    ys.sort(key=lambda v: -v)
    return ys
'''


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_key_functions_agree_with_python(tmp_path):
    """Keys computed once per element, in order; a stable sort (reverse too); min / max the first
    best: checked on random lists with ties, empty ones included."""
    f = tmp_path / "keys.py"
    f.write_text(KEYS)
    rep = verify_file(f, n_random=80, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert all(fc.count("mismatch") == 0 for fc in rep.functions)


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_key_in_a_program(tmp_path):
    f = tmp_path / "keyprog.py"
    f.write_text(textwrap.dedent('''
        prices = {"AAPL": 190.5, "MSFT": 410.25, "GOOG": 140.0}
        ranked = sorted(prices.items(), key=lambda kv: kv[1], reverse=True)
        print(ranked, max(prices, key=prices.get), min(prices, key=lambda k: prices[k]))
        xs = [3.0, -1.0, 2.0, -3.0, 1.0]
        xs.sort(key=abs)
        empty: list[float] = []
        print(xs, sorted([("b", 1), ("a", 1), ("c", 0)], key=lambda t: t[1]), max(empty, default=-1.0))
        ''').lstrip())
    rep = verify_file(f, n_random=5, build_dir=tmp_path / "build")
    assert rep.program is not None and rep.program.identical and rep.warnings == [], rep.to_text()


@pytest.mark.parametrize("src, words", [
    ("xs = [1.0]\nprint(sorted(xs, key=lambda a, b: a))\n", "takes one argument"),
    ("d = {'a': 1.0}\nks = ['a', 'b']\nprint(max(ks, key=d.get))\n", "over d itself"),
    ("xs = [1.0]\nprint(min(xs, default=0))\n", "default= is a int"),
])
def test_key_refusals(src, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    assert bad and words in bad[0].reason, r.report.to_text()
