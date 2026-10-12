"""Programs: a file with code that runs becomes <module>.cpp with an int main(), and running it
prints exactly what `python file.py` prints. Each case here is run both ways by qtrans verify
(the Python file, and the compiled program) and the two outputs compared character for
character, exit status too."""
import textwrap

import pytest

from qtrans.translator import translate_source
from qtrans.verify import printed_numbers_differ, verify_file
from tests.test_semantics import HAVE_CXX

PROGRAMS = {
    "two_lines": "A = 1+1\nprint(A)\n",
    "guarded": '''
        """A small simulation."""
        import numpy as np
        from qtrans.typing import Vec

        N_PATHS = 1000


        def payoff(s: Vec, k: float) -> Vec:
            return np.maximum(s - k, 0.0)


        def simulate(seed: int) -> Vec:
            rng = np.random.default_rng(seed)
            return 100.0 * np.exp(0.2 * rng.standard_normal(N_PATHS))


        if __name__ == "__main__":
            # run it
            s = simulate(42)
            p = payoff(s, 100.0)
            price = float(np.mean(p))
            print(f"price: {price:.6f}")
            print("max", np.max(p), repr(np.max(p)))
            for k in range(3):
                print(k, k * price)
        ''',
    "def_main": '''
        import sys


        def main():
            total = 0
            for i in range(1, 11):
                total += i * i
            print("sum of squares:", total)


        if __name__ == "__main__":
            sys.exit(main())
        ''',
    "number_formats": '''
        import math
        import numpy as np

        print(0.1 + 0.2, 1e16, 1 / 3, -0.0, float("inf"), 2 ** 0.5, 1e-5, 123456789.0 * 1000)
        print(10 / 4, 7 // 2, -7 // 2, 7 % -3, 2 ** 62, -(2 ** 62) + 1)
        print(True, 3 > 2, not True, None)
        x = math.pi
        print(f"{x:.3f}|{x:10.4e}|{x:+.2%}|{1234567.891:,.2f}|{255:#x}|{255:08b}|{-42:>6}|{'ab':*^7}|{x!r}")
        print("{} and {:.1f} and {name}".format(1, 2.25, name="n"), end="!\\n")
        print("a", "b", "c", sep="-")
        print()
        v = np.linspace(0.0, 1.0, 5)
        print(v)
        print(np.eye(3))
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        print(A @ A, A.T)
        print(repr(v), repr(np.sum(v)), (np.mean(v), 2))
        big = np.arange(2000.0)
        print(big)
        print(repr(big))
        print(np.array([1e-10, 1.0, 1e10]))
        s = "it's"
        print(s, repr(s), str(3) + "!", len(v))
        ''',
    "percent_and_round": '''
        import numpy as np

        x, n, s = 1234.56789, 42, "it's"
        print("%.2f|%8.3e|%-6d|%05d|%s|%r|%10s|%-10s|%x|%%|%d|%+.1f|%g|%G|%o|%#x|% d" % (
            x, x, n, n, s, s, s, s, n, x, x, x, 1e-7, n, n, n))
        print("%5.1f%%" % 12.345, "%c" % 65, "%s" % 2.5, "%s" % True, "%r" % "a")
        print(round(2.675, 2), round(1.85, 1), round(0.125, 2), round(1234.5, -2), round(-4.0, -1), round(2.5))
        v = np.array([1.85, 51.555])
        print(round(v[0], 1), round(float(v[0]), 1), round(v[1], 2), round(1250, -2), round(1350, -2))
        print(f"{round(np.mean(v), 3)!r}")
        ''',
    "loops_and_widening": '''
        import math

        total = 0
        for r in [0.01, 0.05, -0.1]:
            for n in [1, 2, 10]:
                total += math.exp(r * n)
                print(r, n, f"{math.exp(r * n):.6f}")
        for name in ["call", "put"]:
            print(name, end=" ")
        print()
        print(total, round(total, 4))
        ''',
    "no_annotations": '''
        import math
        import numpy as np


        def add(a, b):
            return a + b


        def bs_call(S, K, T, r, sigma):
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


        def norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


        def mean_of(v):
            return float(np.mean(v))


        def fact(n):
            if n <= 1:
                return 1
            return n * fact(n - 1)


        def main():
            print(add(2, 3), add(10, -4))
            price = bs_call(100.0, 100.0, 1.0, 0.05, 0.2)
            print(f"call: {price:.4f}")
            print(mean_of(np.array([1.0, 2.0, 4.0])), fact(10))


        if __name__ == "__main__":
            main()
        ''',
    "exception": '''
        def check(n: int) -> int:
            if n < 0:
                raise ValueError(f"n must be >= 0, got {n}")
            return n


        print("start")
        print(check(3))
        print(check(-2))
        print("never printed")
        ''',
    "exit_status": '''
        import sys

        print("leaving with 3")
        sys.exit(3)
        ''',
    "random_loop": '''
        import numpy as np

        rng = np.random.default_rng(7)
        total = 0.0
        for i in range(5):
            total += rng.random()
            print(i, total)
        z = rng.standard_normal(4)
        print(z, z[0], repr(z[1]))
        print(f"{float(np.std(z)):.12f}")
        ''',
    "printing_function": '''
        import sys


        def show(x: float) -> float:
            print("in show:", x)
            return x * 2


        def shout(msg: str) -> None:
            print(msg, "!", file=sys.stderr)
            print("said", msg)


        print("out:", show(1.5))
        shout("hello")
        y = show(show(0.25))
        print(y)
        ''',
}


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
@pytest.mark.parametrize("name", list(PROGRAMS))
def test_the_program_prints_what_python_prints(tmp_path, name):
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(PROGRAMS[name]).lstrip(), encoding="utf-8")
    rep = verify_file(f, n_random=20, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    p = rep.program
    assert p.identical, rep.to_text() + "\n--- Python:\n" + p.py_out[-3000:] + "\n--- C++:\n" + p.cpp_out[-3000:]
    assert rep.ok and rep.warnings == [], rep.to_text()
    expected_status = {"exception": 1, "exit_status": 3}.get(name, 0)
    assert p.py_code == p.cpp_code == expected_status


def test_a_script_reads_like_hand_written_cpp():
    r = translate_source("A = 1+1\nprint(A)\n", "two", "two.py")
    assert r.program is not None and not r.library
    body = r.program[r.program.index("int main()"):]
    assert body == "int main() {\n    long long A = 1LL + 1;\n    std::cout << A << '\\n';\n}\n"
    assert "namespace" not in r.program and "#pragma once" not in r.program
    assert "#include <iostream>" in r.program and "qtrans" not in r.program.split("int main")[0].split("\n", 2)[2]


def test_print_and_format_read_naturally():
    src = textwrap.dedent('''
        def f(x: float, k: int, n: int, ok: bool) -> str:
            print("x =", x, "k =", k)
            print(f"{x:.2f} ({k} of {n}) {ok}")
            print("next:", k + 1)
            return f"run {k}"
        ''')
    h = translate_source(src, "m").header
    assert 'std::cout << "x = " << qt::py::str(x) << " k = " << k << \'\\n\';' in h
    assert ('std::cout << qt::py::format(x, ".2f") << " (" << k << " of " << n << ") " '
            '<< (ok ? "True" : "False") << \'\\n\';') in h
    # k + 1 can overflow (OverflowError, where Python's int grows): the line is made first, so
    # that an exception leaves nothing of it written, as in Python
    assert 'std::cout << ("next: " + std::to_string(qt::iadd(k, 1LL)) + "\\n");' in h
    assert 'return "run " + std::to_string(k);' in h


def test_what_isnt_translated_says_why_at_its_line():
    r = translate_source("import numpy as np\n\nvalues = [1.0, 2.0]\nprint(values)\n", "m", "m.py")
    prog = next(f for f in r.report.functions if f.name == "(program)")
    assert not prog.translated and prog.reason_line == 3 and "lists" in prog.reason
    r = translate_source("print(twice(2))\n\n\ndef twice(x: int) -> int:\n    return 2 * x\n", "m", "m.py")
    prog = next(f for f in r.report.functions if f.name == "(program)")
    assert not prog.translated and "NameError" in prog.reason and prog.reason_line == 1
    # repr of a value that is a Python float on some runs and a NumPy float64 on others
    src = ("import numpy as np\nv = np.ones(3)\nx = 0.0\nfor e in v:\n    x += e\nprint(repr(x))\n")
    prog = next(f for f in translate_source(src, "m", "m.py").report.functions if f.name == "(program)")
    assert not prog.translated and "np.float64" in prog.reason and prog.reason_line == 6
    # `total = 0`, then floats added: a Python int until the first float arrives, shown as 0, not 0.0
    src = "n = 3\ntotal = 0\nfor i in range(n):\n    total += 0.5\nprint(total)\n"
    prog = next(f for f in translate_source(src, "m", "m.py").report.functions if f.name == "(program)")
    assert not prog.translated and prog.reason_line == 5 and "total = 0.0" in prog.hint
    ok = "total = 0\nfor v in [0.5, 1.5]:\n    total += v\nprint(total, f'{total:.1f}')\n"  # the loop surely runs
    assert translate_source(ok, "m", "m.py").report.functions[0].translated
    # `return 0` from a float function: a Python int on that path, printed as 0 (also through a
    # local, another function, or a tuple element); a float spec writes both the same
    f = "def f(x: float) -> float:\n    if x < 0:\n        return 0\n    return x\n\n\n"
    for use, line in (("print(f(-1.0))", 7), ("y = f(-1.0)\nprint(y)", 8), ("print(max(2.5 - 3.0, 0))", 7)):
        prog = next(x for x in translate_source(f + use + "\n", "m", "m.py").report.functions if x.name == "(program)")
        assert not prog.translated and prog.reason_line == line and "0.0 rather than 0" in prog.hint, (use, prog.reason)
    assert translate_source(f + "print(f'{f(-1.0):.2f}', f(-1.0) / 2)\n", "m", "m.py").report.functions[-1].translated
    # a bad format spec: Python raises, so qtrans refuses it rather than guess
    prog = next(f for f in translate_source("x = 2.5\nprint(f'{x:d}')\n", "m", "m.py").report.functions)
    assert not prog.translated and "ValueError" in prog.reason


def test_a_library_is_not_a_program():
    r = translate_source("TOL = 1e-8\n\n\ndef f(x: float) -> float:\n    return x + TOL\n", "m", "m.py")
    assert r.program is None and "(program)" not in {f.name for f in r.report.functions}


def test_output_comparison_allows_only_rounding():
    assert printed_numbers_differ("x 0.30000000000000004\n", "x 0.3000000000000001\n", 1e-10) == (None, 1)
    assert printed_numbers_differ("price 8.554200\n", "price 8.554201\n", 1e-10)[0] == 1  # 1e-7 apart
    assert printed_numbers_differ("v [ 1.5e-17 2. ]\n", "v [ 2.5e-17 2. ]\n", 1e-10) == (None, 1)  # atol
    assert printed_numbers_differ("[1.  2.5]\n", "[1.  2.6]\n", 1e-10)[0] == 1
    assert printed_numbers_differ("a 1\nb\n", "a 1\nc\n", 1e-10)[0] == 2
    assert printed_numbers_differ("n = 3\n", "n = 3\nextra\n", 1e-10)[0] == 2


def test_types_come_from_the_calls_when_left_out():
    src = textwrap.dedent('''
        def scale(x, k=2):
            return x * k


        def area(w, h):
            return w * h


        print(scale(1.5), scale(2.5, 3), area(2, 3))
        ''')
    r = translate_source(src, "m", "m.py")
    sigs = {n: s for n, s in r.signatures.items()}
    assert [p.type.name for p in sigs["scale"].params] == ["float", "int"] and sigs["scale"].ret.name == "float"
    assert [p.type.name for p in sigs["area"].params] == ["int", "int"] and sigs["area"].ret.name == "int"
    note = next(f for f in r.report.functions if f.name == "scale").notes[0]
    assert note == "types from how the file calls it (no annotations): x: float, k: int, -> float"
    # one call gives an int, another a float: Python computes with each as it is; refused
    r = translate_source("def half(x):\n    return x // 2\n\n\nprint(half(7), half(7.0))\n", "m", "m.py")
    half = next(f for f in r.report.functions if f.name == "half")
    assert not half.translated and "int (line 5)" in half.reason and "float (line 5)" in half.reason
    # never called: nothing shows its types
    r = translate_source("def f(x):\n    return x\n", "m", "m.py")
    assert "no call in this file shows its type" in r.report.functions[0].reason
