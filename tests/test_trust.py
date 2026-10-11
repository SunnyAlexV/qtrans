"""0.9 trust work: rectangular and empty arrays, int overflow, the stress profile, timeouts."""
import textwrap
from pathlib import Path

import pytest

from qtrans import translate_source, verify_file
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).resolve().parent / "data"


def cpp(src):
    return translate_source(textwrap.dedent(src), "m").header


def write(tmp_path, name, src):
    p = tmp_path / f"{name}.py"
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return p


# --- int overflow -------------------------------------------------------------------------------------
def test_int_arithmetic_is_plain_where_it_provably_fits():
    h = cpp('''
        from qtrans.typing import Vec

        def f(x: Vec, n: int) -> float:
            t = 0.0
            for i in range(n):
                t += x[i + 1] * 2
            return t + len(x) * 3 + 100000 * 100000
        ''')
    assert "i + 1" in h                    # i < n <= 2**63 - 1, so i + 1 fits
    assert "x.size() * 3" in h             # an array's size is below 2**45
    assert "100000LL * 100000" in h        # literals: known exactly
    assert "qt::i" not in h


def test_int_arithmetic_that_could_overflow_is_checked():
    h = cpp('''
        def f(a: int, b: int) -> int:
            c = a * b + 1
            c += a
            return -c + abs(b) + a ** 3
        ''')
    for fn in ("qt::imul(a, b)", "qt::iadd(", "c = qt::iadd(c, a);", "qt::ineg(c)", "qt::iabs(b)", "qt::ipow(a, 3)"):
        assert fn in h, fn


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_overflow_raises_where_python_ints_grow(tmp_path):
    p = write(tmp_path, "ints", '''
        def mul(a: int, b: int) -> int:
            return a * b

        def neg(a: int) -> int:
            return -a

        def fdiv(a: int, b: int) -> int:
            return a // b

        def md(a: int, b: int) -> int:
            return a % b

        def pw(a: int) -> int:
            return a ** 5

        QTRANS_CASES = {"mul": [(2**40, 2**40), (-3, 7)], "neg": [(-2**63,), (5,)],
                        "fdiv": [(-2**63, -1), (-7, 2)], "md": [(-2**63, -1), (-7, 2)], "pw": [(10**4,), (-3,)]}
        ''')
    r = verify_file(p, n_random=20, build_dir=tmp_path / "b")
    assert r.ok, r.to_text()
    got = {f.name: [(c.py, c.cpp) for c in f.cases if c.source == "given"] for f in r.functions}
    assert got["mul"][0][1] == ("exc", "IntOverflow") and got["mul"][1][1] == ("ok", -21)
    assert got["neg"][0][1] == ("exc", "IntOverflow")
    assert got["fdiv"][0][1] == ("exc", "IntOverflow") and got["fdiv"][1][1] == ("ok", -4)
    assert got["md"][0][1] == ("ok", 0)            # LLONG_MIN % -1: 0 in Python, undefined in plain C++
    assert got["pw"][0][1] == ("exc", "IntOverflow") and got["pw"][1][1] == ("ok", -243)
    known = sum(f.count("known") for f in r.functions)
    assert known >= 4  # the overflows: Python's int grows, C++ raises OverflowError


# --- rectangular and empty arrays ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_rectangular_shapes_are_found_and_checked(tmp_path):
    r = verify_file(DATA / "shapes.py", n_random=120, build_dir=tmp_path)
    assert r.ok, r.to_text()
    for f in r.functions:
        assert f.shape_plans, f"{f.name}: no non-square shape found"
        if f.name == "outer_fill":  # two vectors: of different lengths
            assert any(len(c.args[0]) != len(c.args[1]) for c in f.cases)
            continue
        shapes = {tuple(a.shape) for c in f.cases for a in c.args if hasattr(a, "ndim") and a.ndim == 2}
        assert any(s[0] != s[1] for s in shapes), f.name
        if f.name != "blank_like":  # (indexes row 0)
            assert any(0 in s for s in shapes), f"{f.name}: no empty matrix tried"


def test_shape_plans_follow_what_python_accepts():
    import numpy as np
    import qtrans.verify as V
    from qtrans.project import translate_project
    p = translate_project([DATA / "shapes.py"])
    mods = V.load_project(p)
    sigs = p.modules["shapes"].signatures
    plans = V.shape_plans(mods["shapes"].project, sigs["project"], np.random.default_rng(0))
    assert ("nm", "m") in plans and ("mn", "n") in plans   # A (n x m) @ x (m): both orientations
    assert all(not (a == "nm" and b == "n") for a, b in plans)   # never a mismatched pair


# --- stress profile and timeouts ----------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_stress_profile_and_edge_cases(tmp_path):
    p = write(tmp_path, "st", '''
        import math
        import numpy as np
        from qtrans.typing import Mat, Vec

        def ratio(a: float, b: float) -> float:
            return a / b if b != 0.0 else math.copysign(1.0, a)

        def quad(A: Mat, x: Vec) -> float:
            return float(x @ A @ x)
        ''')
    r = verify_file(p, n_random=300, build_dir=tmp_path / "b", profile="stress")
    assert r.ok, r.to_text()
    floats = [a for f in r.functions for c in f.cases for a in c.args if isinstance(a, float)]
    assert any(a != a for a in floats) and any(abs(a) == float("inf") for a in floats)
    edge = {f.name: f.count("edge") for f in r.functions}
    assert edge["ratio"] == 0  # scalar code stays strict: no 'edge' allowance


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_a_hanging_python_call_times_out(tmp_path, monkeypatch):
    import qtrans.verify as V
    monkeypatch.setattr(V, "PYTHON_CASE_SECONDS", 1.0)
    p = write(tmp_path, "hang", '''
        def spin(x: float) -> float:
            while x > 50.0:
                x = x + 1.0
            return x

        QTRANS_CASES = {"spin": [(1.0,), (60.0,), (2.0,)]}
        ''')
    r = verify_file(p, n_random=0, build_dir=tmp_path / "b")
    f = r.functions[0]
    assert f.timeouts == 2 and [c.args for c in f.cases] == [(1.0,)]  # ordinary input hung: the rest left out
    assert r.ok and "left out" in r.to_text()
