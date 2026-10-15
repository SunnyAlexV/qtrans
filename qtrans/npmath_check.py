"""Checks qtrans_npmath.hpp against the running NumPy, bit for bit.

    python -m qtrans.npmath_check [--n 200000] [--seed 0]     (also: tools/npmath_check.py)

Builds qtrans/runtime/npmath_probe.cpp for the running NumPy (QTRANS_NUMPY_VERSION from it), then
  1. asks the C++ which code it runs for each function, and compares that with what NumPy runs,
     worked out from NumPy's release, the platform and the CPU features NumPy itself reports
     (numpy's __cpu_features__, which honours NPY_DISABLE_CPU_FEATURES);
  2. for every function where NumPy runs its own code (SVML, Tang's kernels, its tanh), compares
     the C++ with NumPy on every input: arrays and single numbers, all bit patterns, the edges;
     np.power and the ** operator for each exponent NumPy treats specially, and random ones.
Where both run the C library they are compared too, and must match where the C++ is built for
the same C library as NumPy's (Linux, where the runtime binds the symbol versions NumPy's wheels
call, and macOS); elsewhere (Windows, where NumPy uses the MSVC runtime) they are counted.

tests/test_npmath.py runs this on every platform of CI, `qtrans doctor` on yours, and
tools/npmath_versions.py under each NumPy release from 1.21 to 2.5.
"""
from __future__ import annotations

import argparse
import os
import platform
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

PROBE = Path(__file__).resolve().parent / "runtime" / "npmath_probe.cpp"

UNARY = ["exp", "log", "exp2", "log2", "log10", "expm1", "log1p", "cbrt", "tan", "arcsin", "arccos", "arctan",
         "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "sin", "cos"]
BINARY = ["power", "arctan2"]


# the C++ and NumPy call the same C library here, so even NumPy's C-library path must match
SAME_LIBM = sys.platform.startswith("linux") or sys.platform == "darwin"


def version_number(v: str) -> int:
    from .npfacts import version_number as vn
    return vn(v)


def cpu_features() -> dict:
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as f
    except ImportError:
        from numpy.core._multiarray_umath import __cpu_features__ as f
    return dict(f)


def _levels() -> dict | None:
    """{function: 'v4' | 'f' | 'v3' | 'arm' | 'base'}: the kernel level NumPy dispatches each
    function to, from numpy.lib.introspect (NumPy 2.0 and later), the ground truth."""
    try:
        from numpy.lib.introspect import opt_func_info
    except ImportError:
        return None
    names = {"arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "power", "arctan2"} | set(UNARY)
    info = opt_func_info(func_name="^(" + "|".join(sorted(names)) + ")$", signature="float64")
    # on 64-bit ARM the baseline itself is NEON (ASIMD) with native FMA, which NumPy's own tanh
    # kernel is built for: there "baseline" runs that kernel (macOS ARM, Linux aarch64)
    arm64 = platform.machine().lower() in ("arm64", "aarch64")
    out = {}
    for fn, sigs in info.items():
        cur = next(iter(sigs.values()))["current"].upper()
        if cur.startswith("BASELINE"):
            out[fn] = "arm" if arm64 else "base"
        elif cur in ("X86_V4",) or cur.startswith("AVX512_") and cur not in ("AVX512_KNL", "AVX512_KNM"):
            out[fn] = "v4"
        elif cur in ("AVX512F", "AVX512_KNL", "AVX512_KNM"):
            out[fn] = "f"
        elif cur in ("X86_V3", "AVX2", "FMA3"):
            out[fn] = "v3"
        else:
            out[fn] = "arm"  # NEON, ASIMD, VSX, ...: a SIMD level with native FMA
    return out


def expected_paths() -> dict:
    """What NumPy runs for each function here, from its release, the platform and the kernel level
    it dispatches to (the table measured on NumPy 1.21 to 2.5; qtrans_npmath.hpp encodes it in C++)."""
    v = version_number(np.__version__)
    machine = platform.machine().lower()
    x86 = machine in ("x86_64", "amd64")
    linux, windows = sys.platform.startswith("linux"), sys.platform == "win32"
    lv = _levels()
    if lv is None:  # NumPy 1.x: from the CPU features it reports
        f = cpu_features()
        level = "v4" if x86 and f.get("AVX512_SKX") else "f" if x86 and f.get("AVX512F") else \
            "v3" if (x86 and f.get("AVX2") and f.get("FMA3")) else "arm" if machine in ("arm64", "aarch64") else "base"
        lv = {fn: level for fn in UNARY + BINARY}
    out = {}
    for fn in UNARY + BINARY:
        level = lv.get(fn, "base")
        p = "libm"
        svml = False
        if linux and x86 and level == "v4":
            if v >= 20000:
                svml = fn not in ("tanh", "sin", "cos")
            elif v >= 12200:
                svml = {"exp": v >= 12300, "log": v >= 12300, "sin": v < 12500, "cos": v < 12500,
                        "power": v >= 12400, "arctan2": v >= 12400, "tanh": v < 12300}.get(fn, True)
        if svml:
            p = "svml_ha" if v >= 20000 else "svml_la"
        elif fn in ("exp", "log") and x86 and level in ("f", "v4") and not windows:
            if not linux or v < 12300 or level == "f":
                p = "numpy_tang"
        elif fn == "tanh" and v >= 12300 and level in ("v3", "v4", "arm"):
            p = "numpy_own"
        out[fn] = p
    return out


def build(work: Path, cxx: str | None = None) -> Path:
    from .translator import NPMATH_HEADERS
    from .verify import find_compiler
    for h in NPMATH_HEADERS:
        (work / h.name).write_bytes(h.read_bytes())
    exe = work / ("npmath_probe.exe" if os.name == "nt" else "npmath_probe")
    c = find_compiler(cxx)
    cmd = c.command(PROBE, exe, work, [])
    cmd.append(f"-DQTRANS_NUMPY_VERSION={version_number(np.__version__)}" if c.kind != "msvc"
               else f"/DQTRANS_NUMPY_VERSION={version_number(np.__version__)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        raise RuntimeError(f"building the probe failed:\n{' '.join(map(str, cmd))}\n{r.stderr}{r.stdout}")
    return exe


def run(exe: Path, head: str, *arrays) -> np.ndarray:
    data = (head + "\n").encode() + b"".join(np.ascontiguousarray(a, dtype=np.float64).tobytes() for a in arrays)
    r = subprocess.run([str(exe)], input=data, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"probe failed on {head!r}: {r.stderr.decode(errors='replace')}")
    return np.frombuffer(r.stdout, dtype=np.float64)


def cxx_paths(exe: Path) -> dict:
    r = subprocess.run([str(exe)], input=b"paths\n", capture_output=True)
    return dict(line.split() for line in r.stdout.decode().splitlines() if line.strip())


def inputs(fn: str, rng, n: int) -> np.ndarray:
    k = n // 4
    bits = rng.integers(0, 2**64, k, dtype=np.uint64).view(np.float64)  # every kind of value, NaN payloads too
    edge = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 1.0, -1.0, 0.5, -0.5, 2.0, 5e-324, -5e-324,
                     2.2250738585072014e-308, 1e-300, 1e300, -1e300, 709.782712893384, 709.7827128933841,
                     -708.3964185322641, -745.1332191019411, -745.1332191019412, 1.0 - 2**-53, 1.0 + 2**-52,
                     np.pi / 2, np.pi, 1e22, 0.9375, 1.06640625]
                    + list(np.array([0x7ff4000000000001, 0xfff4000000000001, 0x7ffc00000000beef, 0xfffc00000000beef,
                                     0x7ff0000000000001, 0xffffffffffffffff], dtype=np.uint64).view(np.float64)))
    # signalling and payload NaNs of both signs: which code quiets a NaN, keeps its sign, keeps its payload
    dom = {"log": np.exp(rng.uniform(-745, 709, k)), "log2": np.exp(rng.uniform(-745, 709, k)),
           "log10": np.exp(rng.uniform(-745, 709, k)), "log1p": np.expm1(rng.uniform(-40, 700, k)),
           "arccosh": 1 + np.exp(rng.uniform(-40, 700, k)), "arcsin": rng.uniform(-1, 1, k),
           "arccos": rng.uniform(-1, 1, k), "arctanh": rng.uniform(-1, 1, k), "exp2": rng.uniform(-1080, 1030, k),
           "tanh": rng.uniform(-25, 25, k), "sin": rng.uniform(-1e4, 1e4, k), "cos": rng.uniform(-1e4, 1e4, k),
           "tan": rng.uniform(-1e4, 1e4, k)}
    wide = dom.get(fn, rng.uniform(-750, 750, k))
    near = rng.uniform(-3, 3, k)
    return np.concatenate([wide, near, bits, edge, rng.uniform(0.9, 1.1, n - 3 * k - len(edge))])


def hexbits(v) -> str:
    return f"{np.float64(v).view(np.uint64):016x}"


def layouts(exe: Path, fn: str, x: np.ndarray, j: int) -> str:
    """One value that differs, in several layouts: NumPy on a number, alone, first or second of a
    pair, at each place of an 8-value array, at its own place in the array; the C++ the same way."""
    f = getattr(np, fn)
    v = x[j]
    out = [f"np.{fn}(number)={hexbits(f(np.float64(v)))}", f"[x]={hexbits(f(np.array([v]))[0])}",
           f"[x,1]={hexbits(f(np.array([v, 1.0]))[0])}", f"[1,x]={hexbits(f(np.array([1.0, v]))[1])}"]
    by_place = []
    for k in range(8):
        a = np.ones(8)
        a[k] = v
        by_place.append(hexbits(f(a)[k]))
    out.append("in 8, places 0-7=" + ",".join(by_place))
    lo = max(0, j - 4)
    out.append(f"x[{lo}:{j + 5}] place {j - lo}={hexbits(f(x[lo:j + 5])[j - lo])}")
    out.append(f"strided={hexbits(f(np.repeat(x[j:j + 1], 2)[::2])[0])}")
    try:
        cx = [hexbits(run(exe, f"scalar {fn} 1", np.array([v]))[0]), hexbits(run(exe, f"unary {fn} 1", np.array([v]))[0]),
              ",".join(hexbits(run(exe, f"unary {fn} 8", np.where(np.arange(8) == k, v, 1.0))[k]) for k in range(8))]
        out.append(f"c++: number={cx[0]} [x]={cx[1]} in 8, places 0-7={cx[2]}")
    except Exception as e:  # noqa: BLE001 - a report, not a check
        out.append(f"c++: {e}")
    return "; ".join(out)


def same_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.view(np.uint64) == b.view(np.uint64)


def check(n: int = 200_000, seed: int = 0, cxx: str | None = None, verbose: bool = True) -> dict:
    """{'paths': (expected, got), 'cases': [(name, count, differ, exact_required)], 'failures': [...]}"""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    work = Path(tempfile.mkdtemp(prefix="qtrans_npmath_"))
    exe = build(work, cxx)
    got = cxx_paths(exe)
    want = expected_paths()
    failures = [f"{fn}: the C++ runs {got.get(fn)}, NumPy runs {p}" for fn, p in want.items() if got.get(fn) != p]
    cases = []

    def record(name, ref, out, exact, x=None, fn=None):
        bad = ~same_bits(ref, out)
        cases.append((name, len(ref), int(bad.sum()), exact))
        if exact and bad.any():
            i = np.flatnonzero(bad)[:3]
            msg = (f"{name}: {int(bad.sum())} of {len(ref)} differ, e.g. x={x[i] if x is not None else '?'} "
                   f"numpy={ref[i]} c++={out[i]}")
            # the bits, and how NumPy and the C++ treat that value in other layouts, so one report says why
            for j in i:
                msg += f"\n    at {j} of {len(ref)}: x={hexbits(x[j]) if x is not None else '?'} numpy={hexbits(ref[j])} c++={hexbits(out[j])}"
                if fn is not None and x is not None:
                    msg += "\n    " + layouts(exe, fn, x, j)
            failures.append(msg)

    for fn in UNARY:
        x = inputs(fn, rng, n)
        exact = want[fn] != "libm" or SAME_LIBM
        record(f"np.{fn}(array)", getattr(np, fn)(x), run(exe, f"unary {fn} {len(x)}", x), exact, x, fn)
        xs = x[:: max(1, len(x) // 3000)]
        ref = np.array([getattr(np, fn)(np.float64(v)) for v in xs])
        record(f"np.{fn}(number)", ref, run(exe, f"scalar {fn} {len(xs)}", xs), exact, xs, fn)
    # np.power and np.arctan2 of two arrays
    base = np.concatenate([np.abs(inputs("exp", rng, n // 2)), rng.uniform(0, 10, n // 2)])
    ex = np.concatenate([rng.uniform(-50, 50, len(base) // 2),
                         rng.integers(0, 2**64, len(base) - len(base) // 2, dtype=np.uint64).view(np.float64)])
    exact = want["power"] != "libm" or SAME_LIBM
    record("np.power(array, array)", np.power(base, ex), run(exe, f"binary power {len(base)}", base, ex), exact, base)
    y = inputs("arctan", rng, n)
    xx = rng.permutation(inputs("arctan", rng, n))
    record("np.arctan2(array, array)", np.arctan2(y, xx), run(exe, f"binary arctan2 {len(y)}", y, xx),
           want["arctan2"] != "libm" or SAME_LIBM, y)
    # a scalar exponent: the ** operator and np.power, for each exponent NumPy treats specially
    xb = base[: n // 4]
    for s in (2.0, 0.5, -1.0, 0.0, 1.0, 3.0, 1.5, -2.0, 0.25, 1e-3, float(rng.uniform(-5, 5))):
        record(f"array ** {s}", xb ** s, run(exe, f"power op {s!r} {len(xb)}", xb), exact, xb)
        record(f"np.power(array, {s})", np.power(xb, s), run(exe, f"power ufunc {s!r} {len(xb)}", xb), exact, xb)
        one = xb[:2000]
        record(f"np.power(number, {s})", np.array([np.power(np.float64(v), s) for v in one]),
               run(exe, f"power ufunc {s!r} {len(one)}", one), exact, one)
    for b in (2.0, 10.0, 0.5, np.e):
        e = rng.uniform(-30, 30, n // 8)
        record(f"{b} ** array", b ** e, run(exe, f"base {b!r} {len(e)}", e), exact, e)
    if verbose:
        print(f"NumPy {np.__version__} on {platform.system()} {platform.machine()}")
        for fn in UNARY + BINARY:
            print(f"  {fn:8s} NumPy runs {want[fn]:10s} the C++ runs {got.get(fn)}")
        for name, cnt, bad, ex_ in cases:
            print(f"  {name:28s} {cnt:>9,} values  {bad:>7,} differ  {'(must be 0)' if ex_ else '(two C libraries)'}")
        print("FAIL:\n  " + "\n  ".join(failures) if failures else "OK: every case NumPy runs its own code for is bit-identical")
    return {"paths": (want, got), "cases": cases, "failures": failures}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cxx")
    a = ap.parse_args(argv)
    return 1 if check(a.n, a.seed, a.cxx)["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
