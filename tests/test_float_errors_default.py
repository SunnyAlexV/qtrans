"""1.0: Python's float errors everywhere. A float division by zero raises ZeroDivisionError, an
overflowing ** or math.exp OverflowError, math.sqrt(-1) ValueError, in the C++ as in Python, in
every function, not only inside a `try` that may catch them; `# qtrans: ieee` gives that up for
a function (or a module) and is reported. Where two parts of an expression can raise, the left
one is worked out first, as Python does, so the same error comes out. And qtrans verify no
longer counts "Python raised, the C++ didn't" as a known difference in a function that raises
Python's errors itself: there it is a mismatch."""
import shutil
import textwrap
from pathlib import Path

import pytest

from qtrans.translator import translate_source
from qtrans.verify import Case, classify, verify_file
from tests.test_semantics import HAVE_CXX

compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")


def cpp(src):
    r = translate_source(textwrap.dedent(src), "m")
    assert not r.report.rejected, [(f.name, f.reason) for f in r.report.rejected]
    return r


def test_errors_are_pythons_without_a_try():
    h = cpp("""
        import math

        def f(a: float, b: float, x: float) -> float:
            return a / b + math.log(x) + x ** 3.5
        """).header
    assert "qt::py::truediv(a, b)" in h and "qt::py::math_log(x)" in h and "qt::py::power(x, 3.5)" in h


def test_ieee_on_a_def_or_a_module_and_reported():
    src = """
        import math

        def fast(a: float, b: float) -> float:  # qtrans: ieee
            return a / b + math.exp(a)

        def exact(a: float, b: float) -> float:
            return a / b
        """
    r = cpp(src)
    assert "inline double fast(double a, double b) {\n    return a / b + std::exp(a);" in r.header
    assert "inline double exact(double a, double b) {\n    return qt::py::truediv(a, b);" in r.header
    notes = {f.name: " ".join(f.notes) for f in r.report.functions}
    assert "qtrans: ieee" in notes["fast"] and "qtrans: ieee" not in notes["exact"]
    assert r.signatures["exact"].float_errors and not r.signatures["fast"].float_errors
    whole = cpp('"""A module."""\n# qtrans: ieee\n' + textwrap.dedent(src).replace("  # qtrans: ieee", ""))
    assert "qt::py::" not in whole.header.split("namespace m")[1]


def test_ieee_has_no_effect_where_an_except_may_catch_the_error():
    h = cpp("""
        def ratio(a: float, b: float) -> float:  # qtrans: ieee
            return a / b

        def safe(a: float, b: float) -> float:
            try:
                return ratio(a, b)
            except ZeroDivisionError:
                return 0.0
        """).header
    assert "inline double ratio(double a, double b) {\n    return qt::py::truediv(a, b);" in h


def test_a_value_that_may_be_numpys_keeps_numpys_result_and_says_so():
    """np.exp gives a NumPy float64, max(..., 0.0) maybe a Python float: Python raises on some
    runs and gives inf on others. Not refused (it was, inside a try): NumPy's result, reported,
    and verify may count those inputs as known (float_errors is off)."""
    r = cpp("""
        import numpy as np

        def g(x: float, y: float) -> float:
            return max(np.exp(x), 0.0) / y
        """)
    note = " ".join(r.report.functions[0].notes)
    assert "may hold a NumPy number or a Python float" in note
    assert not r.signatures["g"].float_errors


def test_no_check_for_an_int_divisor_known_not_to_be_zero():
    h = cpp("""
        def h(n: int) -> float:
            t = 0.0
            for k in range(n):
                t += 1.0 / (k + 1)
            return t
        """).header
    assert "t += 1.0 / static_cast<double>(k + 1);" in h


def test_the_left_side_first_where_both_may_raise():
    h = cpp("""
        import math

        def d1(S: float, K: float, sigma: float, T: float) -> float:
            return math.log(S / K) / (sigma * math.sqrt(T))

        def add(a: float, b: float) -> float:
            return a + b

        def g(x: float, y: float) -> float:
            return add(x, y) + add(y, x) + d1(x, y, 1.0, 1.0) * d1(y, x, 1.0, 1.0)
        """).header
    assert ("    const double left = qt::py::math_log(qt::py::truediv(S, K));\n"
            "    return qt::py::truediv(left, sigma * qt::py::math_sqrt(T));") in h
    assert "add(x, y) + add(y, x)" in h  # add can't raise: nothing to order


def test_array_expressions_are_never_put_in_a_temporary():
    """sigma ** 2 may raise, i ** 2 is an array: an Eigen expression can't leave a lambda
    (it would point at values gone), and NumPy arrays don't raise anyway (lattice_pde's bug)."""
    h = cpp("""
        import numpy as np
        from qtrans.typing import Vec

        def coeffs(sigma: float, n: int) -> Vec:
            i = np.arange(1, n)
            return 0.5 * (sigma ** 2 * i ** 2 - 0.01 * i)
        """).header
    assert "[&]" not in h and "const double left" not in h


def test_verify_counts_a_missing_error_as_a_mismatch_where_python_errors_are_kept():
    def case(cpp_side):
        c = Case("f", (1.0, 0.0), "")
        c.py = ("exc", "ZeroDivisionError", "float division by zero")
        c.cpp = cpp_side
        return c
    c = case(("ok", float("inf")))
    classify(c, None, 1e-10, 1e-13, exact_floats=True)
    assert c.outcome == "mismatch"
    c = case(("ok", float("inf")))
    classify(c, None, 1e-10, 1e-13, exact_floats=False)
    assert c.outcome == "known"  # a function marked `# qtrans: ieee`
    c = case(("exc", "ZeroDivisionError", "float division by zero"))
    classify(c, None, 1e-10, 1e-13, exact_floats=True)
    assert c.outcome == "match"


PROGRAM = '''
import math


def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def ratio(a: float, b: float) -> float:
    return a / b


def fast(a: float, b: float) -> float:  # qtrans: ieee
    return a / b


QTRANS_CASES = {
    "d1": [(0.0, -0.0, 0.05, 0.2, -1.0), (100.0, 100.0, 0.05, 0.2, -1.0), (100.0, 100.0, 0.05, 0.0, 1.0),
           (100.0, 90.0, 0.05, 0.2, 1.0), (0.0, 1.0, 0.05, 0.2, 1.0)],
    "ratio": [(1.0, 0.0), (-1.0, -0.0), (0.0, 0.0), (1.0, 4.0)],
    "fast": [(1.0, 0.0), (1.0, 4.0)],
}

for b in [0.0, 4.0]:
    try:
        print(ratio(2.0, b))
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)
print(fast(1.0, 4.0))
print(ratio(1.0, 0.0))
'''


@compiled
def test_python_errors_come_out_of_the_cpp(tmp_path):
    """Every error case matches (the same class and message), in Python's order for d1; the
    program prints the same and ends the same way (an uncaught ZeroDivisionError: exit 1);
    only the ieee function has known cases."""
    f = tmp_path / "errs.py"
    f.write_text(PROGRAM, encoding="utf-8")
    rep = verify_file(f, n_random=40, build_dir=tmp_path / "build")
    by = {fc.name: fc for fc in rep.functions}
    assert by["d1"].count("mismatch") == 0 and by["d1"].count("known") == 0, rep.to_text()
    assert by["ratio"].count("known") == 0 and by["ratio"].count("mismatch") == 0, rep.to_text()
    assert by["fast"].count("known") >= 1
    d1_cases = [c for c in by["d1"].cases if c.args == (0.0, -0.0, 0.05, 0.2, -1.0)]
    assert d1_cases and d1_cases[0].py[1] == "ZeroDivisionError" and d1_cases[0].outcome == "match"
    assert rep.program is not None and rep.program.identical, rep.to_text()
    assert rep.program.py_code == rep.program.cpp_code == 1
    # the last line of Python's traceback, in the running Python's words ("float division by zero"
    # up to 3.13, "division by zero" in 3.14)
    assert rep.program.cpp_err.strip() == rep.program.py_err.strip().splitlines()[-1]
    assert rep.program.cpp_err.startswith("ZeroDivisionError: ")


# --- what an independent review of 1.0 found, each kept as a test ------------------------------
REVIEW = Path(__file__).parent / "data" / "review10"


def test_a_temporary_never_hides_a_field_or_renames_a_target():
    r = cpp("""
        import math
        from dataclasses import dataclass

        @dataclass
        class W:
            a: float
            b: float
            left: float
            lhs: float

            def m(self) -> float:
                t = math.log(self.a) / math.sqrt(self.b)
                return t + self.left

        def f(p: W, a: float, b: float) -> None:
            p.lhs = math.log(a) / math.sqrt(b)
        """)
    h = r.header
    m = h[h.index("inline double W::m() const {"):]
    assert "const double left2 = qt::py::math_log(a);" in m and "return t + left;" in m  # left: the field
    assert "p.lhs = qt::py::truediv(left, qt::py::math_sqrt(b));" in h


@compiled
@pytest.mark.parametrize("name", ["order_cases", "graph_cases"])
def test_pythons_order_and_exactness_in_the_reviewed_cases(name, tmp_path):
    """The object of a method call before its arguments; an argument that changes something
    before one that raises; xs[k] += v finding xs[k] first; calls by their C++ names (overload
    versions, a renamed `delete`, std::make_shared<Sub>); and float_errors through recursion
    (f and g call an ieee function: not exact, so their inf cases are known, not mismatches)."""
    f = tmp_path / f"{name}.py"
    shutil.copy(REVIEW / f"{name}.py", f)
    rep = verify_file(f, n_random=20, build_dir=tmp_path / "build")
    assert not rep.translation.report.rejected
    by = {fc.name: fc for fc in rep.functions}
    assert all(fc.count("mismatch") == 0 for fc in rep.functions), rep.to_text()
    if name == "order_cases":
        assert all(by[k].count("known") == 0 for k in ("method_order", "effect_then_raise", "call_order", "aug_index"))
    else:
        sigs = rep.translation.signatures
        assert sigs["overload_order"].float_errors and sigs["ctor_order"].float_errors
        assert not any(sigs[k].float_errors for k in ("f", "g", "keyword_caller", "mk", "delete", "k"))
        assert by["ctor_order"].count("known") == 0 and by["overload_order"].count("known") == 0


def test_a_try_around_another_modules_ieee_or_mixed_function_is_refused(tmp_path):
    from qtrans.project import translate_project
    shutil.copy(REVIEW / "dep.py", tmp_path / "dep.py")
    shutil.copy(REVIEW / "xmain.py", tmp_path / "xmain.py")
    p = translate_project([tmp_path / "xmain.py"])
    ok = {f.name: f.translated for f in p.modules["xmain"].report.functions}
    assert ok == {"safe": False, "safe_mixed": False, "safe_exact": True}


def test_a_comprehension_variable_over_floats_is_a_python_float():
    h = cpp("def f(a: float, xs: list[float]) -> list[float]:\n    return [a / x for x in xs]\n").header
    assert "out.push_back(qt::py::truediv(a, x))" in h


def test_complex_overflow_is_the_complex_difference():
    c = Case("f", (-1e308, 2.5), "")
    c.py = ("exc", "OverflowError", "complex exponentiation")
    c.cpp = ("ok", float("nan"))
    classify(c, None, 1e-10, 1e-13, exact_floats=True)
    assert c.outcome == "known"


def test_a_try_around_another_modules_methods_and_constructors(tmp_path):
    """C.ratio is ieee, C.exact raises Python's errors, D's __post_init__ is ieee: only the try
    around C.exact can catch what Python's except catches. (self.a in C is C's own field: the
    `self.a = ...` in D's __post_init__ doesn't make it maybe-NumPy.)"""
    from qtrans.project import translate_project
    shutil.copy(REVIEW / "dep2.py", tmp_path / "dep2.py")
    shutil.copy(REVIEW / "main2.py", tmp_path / "main2.py")
    p = translate_project([tmp_path / "main2.py"])
    ok = {f.name: f.translated for f in p.modules["main2"].report.functions}
    assert ok == {"safe": False, "safe_exact": True, "safe_ctor": False}


def test_a_library_call_mixing_a_change_and_an_error_is_refused():
    r = translate_source(textwrap.dedent("""
        from dataclasses import dataclass

        @dataclass
        class P:
            v: float

        def bump(p: P) -> float:
            p.v += 1.0
            return 0.0

        def f(p: P, a: float, b: float) -> float:
            return max(bump(p), a / b)

        def ok(p: P, a: float, b: float) -> float:
            return max(a / b, 1.0) + max(bump(p), 0.0)
        """), "m")
    why = {f.name: f.reason for f in r.report.rejected}
    assert set(why) == {"f"} and "changes something" in why["f"]
    assert "const double left = std::max(qt::py::truediv(a, b), 1.0);\n    return left + std::max(bump(p), 0.0);" in r.header


def test_the_element_first_even_when_the_right_side_reads_the_list():
    h = cpp("""
        import math

        def f(xs: list[float], k: int, a: float) -> float:
            ys = list(xs)
            ys[k] += math.log(ys[0] + a)
            return ys[0]
        """).header
    assert "auto& target = qt::py::item(ys, k);\n    target += qt::py::math_log(qt::py::item(ys, 0) + a);" in h


def test_storing_into_a_list_field_is_refused_not_a_crash():
    r = translate_source(textwrap.dedent("""
        class A:
            def __init__(self):
                self.xs = [0.0, 0.0]

            def bump(self, k: int) -> None:
                self.xs[k] += 1.0
        """), "m")
    assert any("held in a field is not supported yet" in f.reason for f in r.report.rejected)


def test_a_cached_python_header_missing_a_message_is_written_again():
    """qtrans_python.hpp is written once per Python version into the user's cache. One written by
    an earlier qtrans may lack a message the runtime now uses (fsum's, say): it is not accepted,
    so it is written again, whatever its FACTS number says."""
    from qtrans import pyfacts
    text = pyfacts.SHIPPED.read_text(encoding="utf-8")
    assert pyfacts._version_of(text) is not None
    stale = "\n".join(l for l in text.splitlines() if "fsum_overflow" not in l)
    assert pyfacts._version_of(stale) is None


def test_a_read_that_may_find_a_variable_unset_raises_as_pythons():
    """Found by the fuzzer: `n` set only in the else branch, read after; Python raises
    UnboundLocalError when x > 2.0, the C++ returned 0. Now such a variable carries a flag, and
    the read raises Python's error, with its words; a variable always set first carries none.
    Paths as Python runs them: a loop may run zero times, `while True` leaves only through
    break, a for-else's break skips the else, a handler may run before the try's assignment."""
    flagged = {
        "if_one_side": "def f(x: float) -> float:\n    if x > 2.0:\n        pass\n    else:\n        n = 3.0\n    return n\n",
        "loop_may_not_run": "def f(xs: list[float]) -> float:\n    for x in xs:\n        y = x\n    return y\n",
        "break_skips_else": "def f(xs: list[float]) -> float:\n    for x in xs:\n        if x > 0.0:\n            break\n"
                            "    else:\n        r = 1.0\n    return r\n",
        "handler_leaves_it": "def f(p: float) -> float:\n    try:\n        y = 1.0 / p\n    except ZeroDivisionError:\n"
                             "        pass\n    return y\n",
    }
    plain = {
        "both_sides": "def f(x: float) -> float:\n    if x > 2.0:\n        n = 1.0\n    else:\n        n = 3.0\n    return n\n",
        "while_true": "def f(p: float) -> float:\n    while True:\n        n = p\n        if n > 1.0:\n            break\n"
                      "        p += 1.0\n    return n\n",
        "raise_path": "def f(x: float) -> float:\n    if x > 2.0:\n        n = 1.0\n    else:\n"
                      "        raise ValueError('x')\n    return n\n",
        "handler_sets_it": "def f(p: float) -> float:\n    try:\n        y = 1.0 / p\n    except ZeroDivisionError:\n"
                           "        y = 0.0\n    return y\n",
    }
    for name, src in flagged.items():
        r = translate_source(src, "m")
        assert not r.report.rejected and "qt::py::bound(" in r.header, name
    for name, src in plain.items():
        r = translate_source(src, "m")
        assert not r.report.rejected and "qt::py::bound(" not in r.header and "_set = false" not in r.header, name
    r = translate_source("def f(v: Vec) -> float:\n    for i in range(2):\n        w = v\n    return float(w[0])\n"
                         .replace("def f", "from qtrans.typing import Vec\n\ndef f"), "m")
    assert any("may be read here before it is assigned" in f.reason for f in r.report.rejected)  # an array: refused


@compiled
def test_unbound_reads_agree_with_python(tmp_path):
    """Values and errors: UnboundLocalError with Python's words (caught and printed, too), and a
    program's top level ending in NameError, exit status 1."""
    f = tmp_path / "unbound_cases.py"
    shutil.copy(REVIEW / "unbound_cases.py", f)
    rep = verify_file(f, n_random=30, build_dir=tmp_path / "build")
    assert not rep.translation.report.rejected
    assert all(fc.count("mismatch") == 0 and fc.count("known") == 0 for fc in rep.functions), rep.to_text()
    assert rep.program.identical and rep.program.py_code == rep.program.cpp_code == 1
    assert rep.program.cpp_err.strip() == rep.program.py_err.strip().splitlines()[-1]


def test_aug_division_raises_for_python_floats_everywhere():
    """Found by the fuzzer: `p0 /= 0.0` was C++'s /= (inf); for a Python float it raises, in a
    variable, a list element, a dict value and a field alike. NumPy elements keep inf."""
    h = cpp("""
        from dataclasses import dataclass
        import numpy as np
        from qtrans.typing import Vec

        @dataclass
        class P:
            x: float

        def a(p0: float, d: float) -> float:
            p0 /= d
            return p0

        def b(xs: list[float], k: int, d: float) -> float:
            ys = list(xs)
            ys[k] /= d
            return ys[0]

        def c(p: P, d: float) -> float:
            p.x /= d
            return p.x

        def g(v: Vec, d: float) -> float:
            w = v.copy()
            w[0] /= d
            return float(w[0])
        """).header
    assert "p0 = qt::py::truediv(p0, d);" in h
    assert "auto& target = qt::py::item(ys, k);\n    target = qt::py::truediv(target, d);" in h
    assert "p.x = qt::py::truediv(p.x, d);" in h
    assert "w(0) /= d;" in h or "/= d;" in h.split("inline double g")[1]
