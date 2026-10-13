"""math.fsum in C++ is CPython's own algorithm (qtrans_runtime.hpp): the same correctly rounded
sum, the same half-even rounding across partials, the same errors, on lists built to be hard
(cancellation, huge and tiny values together, ties, overflow, inf and nan)."""
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qtrans.translator import PYTHON_HEADER, RUNTIME_HEADER
from qtrans.verify import find_compiler
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).parent / "data"
compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")


def _lists():
    g = random.Random(2026)
    fixed = [[], [-0.0], [-0.0, -0.0], [1e-16, 1.0, 1e16], [1e308, 1e308], [1e308, -1e308, 1e308],
             [math.inf, -math.inf], [math.nan, 1.0], [math.inf, 1.0, math.inf], [-math.inf, 5.0],
             [0.1] * 10, [1.0, 1e100, 1.0, -1e100], [2.0 ** 53, 1.0, -(2.0 ** 53)], [5e-324, 5e-324, -5e-324],
             [1.7976931348623157e308, 1.0, -1.7976931348623157e308], [1.0, -0.5 ** 53, 0.5 ** 106]]
    out = list(fixed)
    for _ in range(3000):
        n = g.randrange(0, 40)
        kind = g.randrange(4)
        if kind == 0:
            xs = [g.uniform(-1, 1) for _ in range(n)]
        elif kind == 1:
            xs = [g.choice([1.0, -1.0, 0.5, 1e16, -1e16, 1e-16, 3.0, 2.0 ** 52]) * g.choice([1, 1, 3, 7]) for _ in range(n)]
        elif kind == 2:
            xs = [math.ldexp(g.uniform(-1, 1), g.randrange(-1074, 1020)) for _ in range(n)]
        else:  # values that cancel to almost nothing
            ys = [g.uniform(-1e6, 1e6) for _ in range(n)]
            xs = ys + [-y for y in ys] + [g.uniform(-1e-9, 1e-9)]
            g.shuffle(xs)
        out.append(xs)
    return out


@compiled
def test_fsum_is_pythons(tmp_path):
    for h in (RUNTIME_HEADER, PYTHON_HEADER):
        shutil.copy(h, tmp_path / h.name)
    shutil.copy(DATA / "fsum_probe.cpp", tmp_path / "probe.cpp")
    exe = tmp_path / ("probe.exe" if sys.platform == "win32" else "probe")
    subprocess.run(find_compiler().command(tmp_path / "probe.cpp", exe, tmp_path), check=True, capture_output=True)
    lists = _lists()
    out = subprocess.run([str(exe)], input="\n".join(" ".join(x.hex() for x in xs) for xs in lists) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(lists)
    bad = []
    for xs, got in zip(lists, out):
        try:
            want = math.fsum(xs).hex()
        except Exception as e:  # noqa: BLE001 - errors are results too
            want = f"{type(e).__name__}: {e}"
        got = got.strip()
        if got.startswith(("0x", "-0x")) or got in ("inf", "-inf", "nan", "-nan"):
            got = float.fromhex(got).hex() if "x" in got else float(got).hex()
        if want != got and not (want == "nan" and got in ("nan", "-nan")):
            bad.append((xs[:6], want, got))
    assert not bad, f"{len(bad)} of {len(lists)} differ: {bad[:3]}"
