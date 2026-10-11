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


# --- NumPy broadcasting of length-1 dimensions --------------------------------------------------------
def test_plain_eigen_where_sizes_are_provably_equal():
    h = cpp('''
        import numpy as np
        from qtrans.typing import Mat, Vec

        def diffs(p: Vec) -> Vec:
            return p[1:] / p[:-1] - 1.0          # both slices have len(p) - 1 elements

        def demean(A: Mat) -> Mat:
            mu = A.mean(axis=0)
            return A - mu                        # mu has A's number of columns

        def gbm(S0: float, n_paths: int, n_steps: int, seed: int) -> Vec:
            rng = np.random.default_rng(seed)
            S = np.full(n_paths, S0)
            for _ in range(n_steps):
                z = rng.standard_normal(n_paths)
                S = S * np.exp(0.1 * z)          # every value of S has n_paths elements
            return S

        def lattice(V: Vec, p: float, n: int) -> Vec:
            for _ in range(n):
                V = p * V[1:] + (1 - p) * V[:-1]  # V shrinks, but both slices are of this V
            return V

        def normalise(w: Vec) -> Vec:
            w = w / w.sum()
            return w - 0.5 * w
        ''')
    assert "qt::broadcast" not in h and "qt::store" not in h, h
    assert "qt::slice(p, 1, qt::none) / qt::slice(p, qt::none, -1)" in h
    assert "A.rowwise() - mu.transpose()" in h
    assert "S = S * qt::exp(0.1 * z);" in h


def test_broadcasting_helper_where_a_side_may_have_length_1():
    h = cpp('''
        import numpy as np
        from qtrans.typing import Mat, Vec

        def sub(a: Vec, b: Vec) -> Vec:
            return a - b

        def rebound(x: Vec, y: Vec, n: int) -> Vec:
            s = x.copy()
            for _ in range(n):
                s = s + x
                s = y * 2.0                      # s takes y's size: no proof that s and x agree
            return s - x

        def fill(v: Vec, w: Vec) -> Vec:
            out = v.copy()
            out[1:3] = w
            out += w
            return out
        ''')
    assert "return qt::broadcast(a, b, qt::sub_op{});" in h
    assert "return qt::broadcast(s, x, qt::sub_op{});" in h
    assert "qt::store(qt::slice(out, 1, 3), w);" in h and "qt::broadcast_into(out, w, qt::add_op{});" in h


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_length_1_arrays_broadcast_like_numpy(tmp_path):
    p = write(tmp_path, "bc", '''
        import numpy as np
        from qtrans.typing import Mat, Vec

        def sub(a: Vec, b: Vec) -> Vec:
            return a - b

        def add_rows(A: Mat, b: Vec) -> Mat:
            return A + b

        def acc(a: Vec, b: Vec) -> Vec:
            c = a.copy()
            c += b
            return c

        def count_below(a: Vec, b: Vec) -> float:
            return float(np.sum(a < b))

        def floor_at(a: Vec, b: Vec) -> Vec:
            return np.maximum(a, b)

        def pick(a: Vec, b: Vec) -> Vec:
            return np.where(a > 1.0, a, b)

        def fill(v: Vec, w: Vec) -> Vec:
            out = v.copy()
            out[1:3] = w
            return out

        def set_row(A: Mat, v: Vec) -> Mat:
            B = A.copy()
            B[0] = v
            return B

        one, three = np.array([2.0]), np.array([1.0, 2.0, 3.0])
        QTRANS_CASES = {"sub": [(one, three), (three, one)], "add_rows": [(np.ones((2, 3)), one), (np.ones((2, 1)), three)],
                        "acc": [(three, one), (one, three)], "count_below": [(one, three)],
                        "floor_at": [(three, one)], "pick": [(three, one), (one, three)],
                        "fill": [(np.zeros(5), one), (np.zeros(5), three)], "set_row": [(np.ones((2, 3)), one)]}
        ''')
    r = verify_file(p, n_random=60, build_dir=tmp_path / "b")
    assert r.ok, r.to_text()
    given = {f.name: [(c.py[0], c.cpp[0]) for c in f.cases if c.source == "given"] for f in r.functions}
    assert given["sub"] == [("ok", "ok")] * 2 and given["add_rows"] == [("ok", "ok")] * 2
    assert given["acc"] == [("ok", "ok"), ("exc", "exc")]   # c (length 1) += length 3: NumPy refuses
    assert given["fill"] == [("ok", "ok"), ("exc", "exc")]  # 3 values into a slice of 2: refused
    for f in r.functions:  # the verifier's own shape probe tries length 1 too
        assert any(any(a.ndim and 1 in a.shape for a in c.args if hasattr(a, "ndim")) for c in f.cases
                   if c.source != "given"), f.name


def test_an_outcome_that_a_nudge_flips_is_a_knife_edge(tmp_path):
    """python_flips: does a rounding-sized nudge change whether Python raises? Tested on exact
    knife edges rather than on LAPACK, whose verdict on a singular matrix (the case this exists
    for) depends on the LAPACK build: that is the point, and why it can't be pinned in a test."""
    import numpy as np
    import qtrans.verify as V
    from qtrans.project import translate_project
    p = write(tmp_path, "edge", '''
        def at_one(x: float) -> float:
            if x == 1.0:
                raise ValueError("Singular matrix")  # only exactly 1.0, as an exactly-zero pivot
            return x

        def below_two(x: float) -> float:
            if x < 2.0:
                raise ValueError("Singular matrix")  # raises however 1.0 is rounded
            return x
        ''')
    project = translate_project([p])
    mod = V.load_project(project)["edge"]
    sigs = project.modules["edge"].signatures
    for name, args, flips in (("at_one", (1.0,), True), ("at_one", (3.0,), False), ("below_two", (1.0,), False)):
        fn = getattr(mod, name)
        c = V.Case(name, args, "given")
        c.py = V.run_python(fn, args, sigs[name])
        assert V.python_flips(fn, c, sigs[name], np.random.default_rng(1)) is flips, (name, args)
