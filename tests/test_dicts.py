"""Dicts as qt::py::dict: keys kept in insertion order, KeyError, printing as Python prints a
dict. Programs are run both ways (Python and the compiled C++) and their output compared
character for character; functions taking and returning dicts are checked by qtrans verify on
random dicts, empty ones included. What a C++ copy or loop would do differently is refused."""
import textwrap

import pytest

from qtrans.translator import translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

PROGRAM = """
import numpy as np

prices = {"AAPL": 190.5, "MSFT": 410.25}
prices["GOOG"] = 140.0
prices["AAPL"] += 1.5
print(prices, len(prices), prices["MSFT"], "AAPL" in prices, "X" not in prices)
print(prices.get("X", -1.0), sum(prices.values()), sorted(prices), max(prices.values()), list(prices))
counts = {}
for w in ["a", "b", "a"]:
    counts[w] = counts.get(w, 0) + 1
print(counts, sum(counts.values()))
del prices["MSFT"]
g = prices.pop("GOOG")
z = prices.pop("Z", 0.5)
print(prices, g, z)
for k in prices:
    print(k, prices[k])
for k, v in prices.items():
    prices[k] = v * 2
    print(k, v, prices[k])
weights = {1: 0.25, 2: 0.75}
print(weights, {k: v * 100 for k, v in weights.items()}, {n: len(n) for n in ["ab", "c"]})
print([f"{k}={v:.2f}" for k, v in weights.items()], sum(v for v in weights.values()))
names = {0: "call", 1: "put"}
names[0] = "CALL"
names[2] = "fwd"
print(names, names[2], repr(names[0]))
e: dict[str, float] = {}
e.setdefault("x", 1.0)
e.setdefault("x", 2.0)
e.update({"y": 3.0, "x": 4.0})
f = e.copy()
f["z"] = 0.0
print(e, f, dict(e) == e)
arr = np.array([1.5, 2.0])
nd = {"a": arr[0], "b": arr[1]}
print(nd, nd["a"])
for k in list(e):
    if e[k] > 3.5:
        del e[k]
print(e)
e.clear()
print(e)
print(prices["nope"])
"""


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_dict_program_prints_what_python_prints(tmp_path):
    f = tmp_path / "dicts.py"
    f.write_text(PROGRAM.lstrip(), encoding="utf-8")
    rep = verify_file(f, n_random=10, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    p = rep.program
    assert p.identical, rep.to_text() + "\n--- Python:\n" + p.py_out[-3000:] + "\n--- C++:\n" + p.cpp_out[-3000:]
    assert rep.warnings == [] and p.py_code == p.cpp_code == 1  # the last line: KeyError, as in Python


FUNCTIONS = """
def total(d: dict[str, float]) -> float:
    s = 0.0
    for k, v in d.items():
        s += v
    return s


def scaled(d: dict[str, float], k: float) -> dict[str, float]:
    return {name: v * k for name, v in d.items()}


def histogram(xs: list[int]) -> dict[int, int]:
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def label(codes: dict[int, str], c: int) -> str:
    return codes.get(c, "?")


def biggest(d: dict[str, float]) -> str:
    best = ""
    for k in d:
        if best == "" or d[k] > d[best]:
            best = k
    return best
"""


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_dict_functions_agree_with_python(tmp_path):
    f = tmp_path / "dicts_mod.py"
    f.write_text(FUNCTIONS, encoding="utf-8")
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert {fc.name for fc in rep.functions} == {"total", "scaled", "histogram", "label", "biggest"}
    assert all(fc.count("mismatch") == 0 and len(fc.cases) == 60 for fc in rep.functions)


def test_dicts_read_like_cpp():
    h = translate_source(FUNCTIONS, "m").header
    assert "inline double total(const qt::py::dict<std::string, double>& d) {" in h
    assert "for (const std::string& k : d.keys()) {" in h and "v = d.at(k);" in h
    assert "qt::py::dict<long long, long long> out = {};" in h
    assert "out[x] = qt::iadd(out.get(x, 0LL), 1LL);" in h
    assert "return codes.get(c, \"?\");" in h
    p = translate_source(PROGRAM, "m", "m.py").program
    assert 'qt::py::dict<std::string, double> prices = {{"AAPL", 190.5}, {"MSFT", 410.25}};' in p
    assert 'prices.at("AAPL") += 1.5;' in p and 'prices.erase("MSFT");' in p


@pytest.mark.parametrize("src, line, words", [
    ("d = {'a': 1.0}\nfor k in d:\n    d['b'] = 2.0\n", 2, "dictionary changed size during iteration"),
    ("d = {'a': 1.0}\nfor k in d:\n    del d[k]\n", 2, "changed size"),
    ("d = {'a': 1.0}\nfor k, v in d.items():\n    d.pop(k)\n", 2, "changed size"),
    ("d = {'a': 1, 'b': 2.5}\nprint(d)\n", 1, "several types"),
    ("d = {'a': 1.0}\nd['b'] = 2\nprint(d)\n", 2, "keeps each value's type"),
    ("d = {'a': 1.0}\nprint(d.get('a'))\n", 2, "returns None"),
    ("d = {1: 1.0}\nprint(d[1.0])\n", 2, "float key"),
    ("d = {'a': 1.0}\nprint(d, d.pop('a'))\n", 2, "changes 'd', and also reads 'd'"),
    ("a = {'a': 1.0}\nb = a\nb['c'] = 2.0\nprint(a)\n", 2, "one dict in Python"),
    ("def f(d: dict[str, float]) -> None:\n    d['x'] = 1.0\n\nv = {'a': 2.0}\nf(v)\n", 2, "passed in"),
    ("d = {}\nprint(len(d))\n", 2, "d: dict[str, float] = {}"),
    ("d = {(1, 2): 1.0}\nprint(d)\n", 1, "keys of type"),
    ("d = {'a': [1.0]}\nprint(d)\n", 1, "values of type"),
])
def test_what_a_cpp_dict_would_do_differently_is_refused_at_its_line(src, line, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    assert bad, r.report.to_text()
    first = next((f for f in bad if f.reason_line == line), bad[-1])
    assert first.reason_line == line and (words in first.reason or words in (first.hint or "")), first.reason
