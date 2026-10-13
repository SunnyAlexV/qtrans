"""enumerate() and zip() loops (and comprehensions, and generators read by sum / min / max /
any / all / join): rewritten as index loops, run both ways (Python and the compiled C++) with
the output compared character for character; refused, with the reason, where an index loop
would not read what Python's iterator reads."""
import ast
import textwrap

import pytest

from qtrans import loops
from qtrans.translator import translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

PROGRAMS = {
    "enumerate_zip": '''
        prices = [100.0, 101.5, 99.25, 102.0]
        for i, p in enumerate(prices):
            print(i, p)
        for day, p in enumerate(prices, start=1):
            print(f"day {day}: {p:.2f}")
        for a, b in zip(prices[1:], prices[:-1]):
            print(a / b - 1)
        names = ["call", "put", "fwd"]
        strikes = [90.0, 110.0]
        for n, k in zip(names, strikes):  # stops at the shorter one
            print(n, k)
        for i, (n, k) in enumerate(zip(names, strikes)):
            print(i, n, k)
        last = -1
        for last, p in enumerate(prices):
            pass
        print(last)
        for _, p in enumerate(prices):
            print(p * 2)
        total = 0.0
        for i, p in enumerate(prices):
            for j, q in enumerate(prices):
                if j > i:
                    total += p * q
        print(total)
        for i, n in enumerate(names):
            if n == "put":
                print("found at", i)
                break
        for i, p in enumerate(prices):
            prices[i] = p * 2  # changing an element (not the length) is read back as Python does
        print(prices)
        for x, y, z in zip([1, 2, 3], (4, 5, 6), [7, 8]):
            print(x + y + z)
        start = 10
        for i, p in enumerate(prices, start):
            print(i, end=" ")
        print()
        ''',
    "comprehensions": '''
        xs = [1.0, -2.0, 3.0]
        ws = [0.5, 0.25, 0.25]
        names = ["a", "b"]
        print([x * w for x, w in zip(xs, ws)], [f"{n}{i}" for i, n in enumerate(names)])
        print([i for i, x in enumerate(xs) if x > 0], [x + i for i, x in enumerate(xs, 1)])
        print(sum(x * w for x, w in zip(xs, ws)), max(abs(x) for x in xs), min(x for x in xs if x > 0))
        print(", ".join(f"{x:.1f}" for x in xs), "-".join(names), "".join([]), sorted(x * x for x in xs))
        print(sum(1 for x in xs if x > 0), list(x / 2 for x in xs))
        print(any(x < 0 for x in xs), all(x > 0 for x in xs), any(a > b for a, b in zip(xs, ws)))
        zs = [0.0, 2.0]
        print(any(1 / z > 0.1 for z in zs if z != 0), all(z == 0 or 1 / z > 0 for z in zs))
        for i in range(2):
            print(i, [i * 2 for i in range(3)])
        print([b - a for a, b in zip(xs, xs[1:])], [x for x, y in zip(xs[2:-1], xs)], [x for x in xs[5:]])
        ''',
    "arrays": '''
        import numpy as np

        v = np.array([1.5, 2.0, 3.25])
        w = np.array([2.0, 4.0, 8.0])
        for i, x in enumerate(v):
            print(i, x, repr(x))
        for a, b in zip(v, w):
            print(a * b)
        print([float(a - b) for a, b in zip(v, w)], sum(a * b for a, b in zip(v, w)))
        print([float(b / a - 1) for a, b in zip(v, v[1:])], sum(b - a for a, b in zip(v[:-1], v[1:])))
        ''',
}


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
@pytest.mark.parametrize("name", list(PROGRAMS))
def test_loop_programs_print_what_python_prints(tmp_path, name):
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(PROGRAMS[name]).lstrip(), encoding="utf-8")
    rep = verify_file(f, n_random=10, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    p = rep.program
    assert p.identical, rep.to_text() + "\n--- Python:\n" + p.py_out[-3000:] + "\n--- C++:\n" + p.cpp_out[-3000:]
    assert rep.warnings == [] and p.py_code == p.cpp_code == 0


FUNCTIONS = '''
def weighted(xs: list[float], ws: list[float]) -> float:
    total = 0.0
    for x, w in zip(xs, ws):
        total += x * w
    return total


def first_negative(xs: list[float]) -> int:
    for i, x in enumerate(xs):
        if x < 0:
            return i
    return -1


def changes(xs: list[float]) -> list[float]:
    return [b - a for a, b in zip(xs, xs[1:])]


def ranks(xs: list[int]) -> list[int]:
    out = []
    for i, x in enumerate(sorted(xs)):
        out.append(i * x)
    return out


def all_positive(xs: list[float]) -> bool:
    return all(x > 0 for x in xs)
'''


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_loop_functions_agree_with_python(tmp_path):
    f = tmp_path / "loops_mod.py"
    f.write_text(FUNCTIONS, encoding="utf-8")
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert {fc.name for fc in rep.functions} == {"weighted", "first_negative", "changes", "ranks", "all_positive"}
    assert all(fc.count("mismatch") == 0 and len(fc.cases) == 60 for fc in rep.functions)


def test_loops_read_like_cpp():
    h = translate_source(FUNCTIONS, "m").header
    # zip: one index up to the shorter length, each element read by it, unchecked (always in range)
    assert "for (long long i = 0, i_end = std::min(static_cast<long long>(xs.size()), " \
           "static_cast<long long>(ws.size())); i < i_end; ++i) {" in h
    assert "        double x = xs[static_cast<std::size_t>(i)];\n        double w = ws[static_cast<std::size_t>(i)];\n" in h
    # enumerate: its own index is the loop index
    assert "for (long long i = 0, i_end = static_cast<long long>(xs.size()); i < i_end; ++i) {" in h
    assert "item(" not in h.split("first_negative")[1].split("changes")[0]
    # a sequence that is computed (sorted(xs), xs[1:]) is computed once, before the loop
    assert "std::vector<long long> x_values = qt::py::sorted(xs);" in h
    # all(): a loop that stops at the first element that decides
    assert "for (double x : xs) if (!(x > 0)) return false; return true;" in h


def test_the_rewrite_is_python_that_means_the_same():
    src = "for i, (a, b) in enumerate(zip(xs, ys), 1):\n    print(i, a, b)\n"
    out = ast.unparse(loops.rewrite(ast.parse(src)))
    assert out == ("for j in range(min(len(xs), len(ys))):\n    i = j + 1\n    a = xs[j]\n    b = ys[j]\n"
                   "    print(i, a, b)")
    for name, prog in PROGRAMS.items():  # the rewritten programs print the same in Python itself
        tree = ast.parse(textwrap.dedent(prog))
        assert compile(loops.rewrite(tree), name, "exec")


@pytest.mark.parametrize("src, line, words", [
    ("xs = [1.0]\nfor i, x in enumerate(xs):\n    xs.append(x)\n", 2, "changes the length of 'xs' (append)"),
    ("xs = [1.0]\nfor a, b in zip(xs, xs):\n    xs = [2.0]\n", 2, "assigns 'xs'"),
    ("xs = [1.0]\nfor a, b in zip(xs, xs, strict=True):\n    print(a)\n", 2, "strict=True"),
    ("xs = [1.0]\nfor i, x in zip(range(3), xs):\n    print(i)\n", 2, "range() inside"),
    ("xs = [1.0]\nfor a, b in zip(xs):\n    print(a)\n", 2, "unpack zip() into 1 names"),
    ("xs = [1.0]\nfor a, b in xs:\n    print(a)\n", 2, "cannot unpack a float value into 2 names"),
    ("xs = [1.0]\nprint([a * b for a, b in zip(xs, sorted(xs))])\n", 2, "take names or plain slices"),
    ("def zip(a, b):\n    return a\n\nxs = [1.0]\nfor a, b in zip(xs, xs):\n    print(a)\n", 5, "its own enumerate, zip"),
])
def test_loops_an_index_cannot_follow_are_refused_at_their_line(src, line, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    assert bad, r.report.to_text()
    first = bad[-1]
    assert first.reason_line == line and (words in first.reason or words in (first.hint or "")), first.reason


ELSE = '''
def find(xs: list[float], target: float) -> int:
    for i, x in enumerate(xs):
        if x == target:
            break
    else:
        return -1
    return i


def first_negative(xs: list[float]) -> float:
    found = 0.0
    for x in xs:
        if x < 0:
            found = x
            break
    else:
        found = 1.0
    return found


def count_down(n: int) -> int:
    steps = 0
    while n > 0:
        n -= 1
        steps += 1
        for k in range(3):
            if k == 1:
                break  # leaves the inner loop only: the outer loop's else still runs
    else:
        steps += 100
    return steps
'''


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_for_and_while_else_agree_with_python(tmp_path):
    f = tmp_path / "loop_else.py"
    f.write_text(ELSE)
    rep = verify_file(f, n_random=80, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    h = translate_source(ELSE, "m").header
    assert "bool broke = false;" in h and "if (!broke) {  // the loop's else" in h
    # no break of its own (only the inner loop's): no flag, the else simply follows the loop
    assert "// the loop's else: it has no break, so this always runs after it\n    steps = qt::iadd(steps, 100LL);" in h


def test_a_loop_variable_read_in_the_else_is_refused():
    r = translate_source("def f(n: int) -> int:\n    for i in range(n):\n        pass\n    else:\n        return i\n"
                         "    return 0\n", "m")
    assert not r.report.functions[0].translated and "used outside its for-loop" in r.report.functions[0].reason
