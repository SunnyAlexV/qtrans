"""int64 arrays (0.14): np.arange of ints, rng.integers, index arrays (x[idx], the bootstrap),
int arithmetic as NumPy's (wrapping at 2**64, floor division and modulo with Python's signs and 0
for a zero divisor), conversion to float64 where NumPy converts, and NumPy's printing."""
import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from qtrans import translate_source
from qtrans.translator import ARRAYS_HEADER, PRINT_HEADER, PYTHON_HEADER, RANDOM_HEADER, RUNTIME_HEADER, UNICODE_HEADER
from qtrans.verify import find_compiler, find_eigen, verify_file

DATA = Path(__file__).parent / "data"
try:
    find_compiler()
    find_eigen()
    HAVE = True
except RuntimeError:
    HAVE = False
compiled = pytest.mark.skipif(not HAVE, reason="no C++ compiler or Eigen")


def _build(tmp_path, source):
    for h in (RUNTIME_HEADER, PYTHON_HEADER, ARRAYS_HEADER, RANDOM_HEADER, PRINT_HEADER, UNICODE_HEADER):
        shutil.copy(h, tmp_path / h.name)
    shutil.copy(DATA / source, tmp_path / "probe.cpp")
    exe = tmp_path / ("probe.exe" if sys.platform == "win32" else "probe")
    subprocess.run(find_compiler().command(tmp_path / "probe.cpp", exe, tmp_path, [find_eigen()]), check=True,
                   capture_output=True)
    return exe


@compiled
def test_integers_draws_as_numpy_does(tmp_path):
    """Generator.integers: ranges below and above 2**32, the full 64-bit range, a single value,
    size 0 (no bounds check, as in NumPy), NumPy's errors, and the generator's state afterwards,
    the half word kept for the next 32-bit draw included."""
    exe = _build(tmp_path, "integers_probe.cpp")
    r = random.Random(3)
    cases = []
    for _ in range(4000):
        lo = r.choice([0, -5, 3, -2 ** 62, 2 ** 31, -(2 ** 63)])
        span = r.choice([1, 2, 3, 7, 1000, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1, 2 ** 40, 2 ** 62, 2 ** 63 - 1, 2 ** 64 - 1])
        hi = min(lo + span, 2 ** 63 - 1)
        if r.random() < 0.05:
            hi = max(lo - r.choice([0, 3]), -(2 ** 63))  # an empty range: NumPy's error
        cases.append((r.randrange(2 ** 40), lo, hi, r.choice([-1, 0, 1, 5, 17])))
    out = subprocess.run([str(exe)], input="".join(f"x {s} {lo} {hi} {n}\n" for s, lo, hi, n in cases),
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(cases)
    bad = []
    for (s, lo, hi, n), o in zip(cases, out):
        g = np.random.default_rng(s)
        try:
            v = g.integers(lo, hi) if n < 0 else g.integers(lo, hi, size=n)
        except ValueError as e:
            if o != f"ValueError: {e}":
                bad.append((s, lo, hi, n, o, str(e)))
            continue
        vals = str(int(v)) if n < 0 else "".join(f"{int(x)} " for x in v)
        rnd = g.random()
        st = g.bit_generator.state
        u = st["uinteger"] if st["has_uint32"] else int(g.bit_generator.random_raw()) & 0xFFFFFFFF
        got_vals, rest = o.split("| ")
        got_rnd, got_u = rest.split()
        if (got_vals, float(got_rnd), int(got_u)) != (vals, rnd, u):
            bad.append((s, lo, hi, n, o, f"{vals}| {rnd!r} {u}"))
    assert not bad, f"{len(bad)} of {len(cases)} differ: {bad[:3]}"


def _np_case(op, a, b, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if op == "add": return str(a + b)
        if op == "sub": return str(a - b)
        if op == "mul": return str(a * b)
        if op == "floordiv": return str(a // b)
        if op == "mod": return str(a % b)
        if op == "floordivk": return str(a // np.int64(k))
        if op == "modk": return str(a % np.int64(k))
        if op == "pow": return str(a ** b)
        if op == "powk": return str(a ** np.int64(k))
        if op == "kpow": return str(np.int64(k) ** a)
        if op == "neg": return str(-a)
        if op == "abs": return str(np.abs(a))
        if op == "sum": return str(int(a.sum()))
        if op == "cumsum": return str(np.cumsum(a))
        if op == "max": return str(int(a.max()))
        if op == "argmax": return str(int(a.argmax()))
        if op == "repr": return repr(a)
        if op == "str": return str(a)
        if op == "arange": return str(np.arange(int(a[0]), int(a[1]), int(a[2])))
    raise ValueError(op)


@compiled
def test_arithmetic_and_printing_are_numpys(tmp_path):
    exe = _build(tmp_path, "int_ops_probe.cpp")
    r = random.Random(11)
    big = [2 ** 63 - 1, -(2 ** 63), 2 ** 62, -(2 ** 40), 0, -1, 1]
    lines, want = [], []
    ops = ["add", "sub", "mul", "floordiv", "mod", "floordivk", "modk", "pow", "powk", "kpow", "neg", "abs", "sum", "cumsum", "max",
           "argmax", "repr", "str", "arange"]
    for _ in range(3000):
        op = r.choice(ops)
        n = r.choice([0, 1, 3, 8, 30, 1100]) if op in ("repr", "str") else r.choice([1, 3, 6])
        def draw():
            if r.random() < 0.25:
                return [r.choice(big + [r.randint(-50, 50)]) for _ in range(n)]
            return [r.randint(-1000, 1000) * r.choice([1, 1, 1, 10 ** 9]) for _ in range(n)]
        a, b = draw(), draw()
        if op == "arange":
            a = [r.randint(-20, 20), r.randint(-20, 20), r.choice([1, 2, 3, -1, -4, 7])]
        if r.random() < 0.2 and op in ("floordiv", "mod"):
            b = [0] * n
        k = r.choice([0, 1, -1, 3, -7, 2 ** 40])
        if op in ("pow", "powk", "kpow"):  # exponents: small, large, and now and then negative
            e = lambda: r.choice([0, 1, 2, 3, 5, 63, 64, 2 ** 40 + 1, -1] if r.random() < 0.1 else [0, 1, 2, 3, 7, 62, 65])
            if op == "pow":
                b = [e() for _ in range(n)]
            elif op == "powk":
                k = e()
            else:
                a = [e() for _ in range(n)]
        A, B = np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)
        try:
            w = _np_case(op, A, B, k)
        except Exception as e:  # noqa: BLE001 - errors are results too
            w = f"{type(e).__name__}: {e}"
        lines.append(f"{op} {len(a)} {' '.join(map(str, a))} {len(b)} {' '.join(map(str, b))} {k}")
        want.append(w.replace("\n", "|"))
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True).stdout.splitlines()
    bad = [(ln, w, o) for ln, w, o in zip(lines, want, out) if w != o]
    assert len(out) == len(lines) and not bad, f"{len(bad)} of {len(lines)} differ: {bad[:3]}"


@compiled
def test_a_bootstrap_and_int_array_code_agree_with_python(tmp_path):
    """rng.integers for the bootstrap's indexes, x[idx].mean(), np.arange of ints with + - * //
    %, lam ** j, np.floor(x).astype(int), a loop over an int array, printing: every function
    verified on random arrays (huge values included), and the program's output identical."""
    f = tmp_path / "int_arrays_bootstrap.py"
    shutil.copy(DATA / "int_arrays_bootstrap.py", f)
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert not rep.translation.report.rejected, rep.translation.report.to_text()
    assert rep.program is not None and rep.program.identical, rep.to_text()
    assert rep.ok and rep.warnings == [], rep.to_text()


HEAD = "import numpy as np\nfrom qtrans.typing import Vec, IVec\n\n"


def test_the_cpp_reads_as_written_by_hand():
    h = translate_source(HEAD + "def f(x: Vec, seed: int) -> float:\n    rng = np.random.default_rng(seed)\n"
                                "    idx = rng.integers(0, len(x), size=len(x))\n    return float(x[idx].mean())\n\n"
                                "def g(n: int) -> IVec:\n    return np.arange(n) * 2 + 1\n", "m").header
    assert "qt::IVec idx = rng.integers(0, static_cast<long long>(x.size()), x.size());" in h
    assert "qt::mean(qt::take(x, idx))" in h
    assert "qt::iadd(qt::imul(qt::iarange(n), 2LL), 1LL)" in h


@pytest.mark.parametrize("src, words", [
    ("def f() -> float:\n    a = np.array([[1, 2], [3, 4]])\n    return float(a.sum())\n", "2-D int array"),
    ("def f(x: Vec, i: IVec) -> None:\n    x[i] = 0.0\n", "through an index array"),
    ("def f(i: IVec) -> IVec:\n    return i ** \"2\"\n", "on strings only"),
    ("def f(n: int) -> float:\n    return float(np.full(n, True).sum())\n", "bool array"),
    ("def f(i: IVec) -> IVec:\n    return np.unique(i)\n", "not supported yet"),
])
def test_refusals(src, words):
    r = translate_source(HEAD + src, "m")
    assert any(words in f.reason for f in r.report.rejected), [(f.name, f.reason) for f in r.report.rejected]


def test_bindings_take_and_give_int64_arrays(tmp_path):
    """The compiled module takes int64 NumPy arrays (and refuses float ones, which Python would
    treat differently), returns int64 arrays with np.int64 elements, and agrees with Python."""
    pytest.importorskip("pybind11")
    if not HAVE:
        pytest.skip("no C++ compiler or Eigen")
    from qtrans.bind import build_extension, check_bindings
    f = tmp_path / "int_arrays_bootstrap.py"
    shutil.copy(DATA / "int_arrays_bootstrap.py", f)
    ext = build_extension(f, out_dir=tmp_path, build_dir=tmp_path / "build")
    rep = check_bindings(f, n_random=40, ext=ext)
    assert rep.ok, rep.to_text()
    m = ext.load()
    r = m.mod_pattern(3, 2)
    assert r.dtype == np.int64 and isinstance(r[0], np.int64) and r.tolist() == [8, 5, 3, 0, -2, -5]
    with pytest.raises(TypeError, match="int64"):
        m.count_in(np.array([1.0, 2.0]), 1, 3)
