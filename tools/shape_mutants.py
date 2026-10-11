"""Why the verifier tries rectangular and empty arrays: two deliberate shape bugs that square
inputs can't see.

    python tools/shape_mutants.py

Each mutant is applied to a copy of qtrans in a temporary folder (the package itself is not
touched), then tests/data/shapes.py is verified twice: with square inputs only (every Vec of n,
every Mat n x n, as before 0.9) and with the rectangular shape plans.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTANTS = {
    "np.zeros_like of a matrix built transposed (cols x rows)": (
        "qtrans/arrays.py",
        'dims = f"{a}.size()" if A.type == VEC else f"{a}.rows(), {a}.cols()"',
        'dims = f"{a}.size()" if A.type == VEC else f"{a}.cols(), {a}.rows()"'),
    "np.mean(A, axis=...) divides by the other dimension": (
        "qtrans/runtime/qtrans_arrays.hpp",
        "    const Index count = (axis == 0 || axis == -2) ? a.rows() : a.cols();",
        "    const Index count = (axis == 0 || axis == -2) ? a.cols() : a.rows();"),
}

RUN = r'''
import sys
import qtrans.verify as V
from qtrans import verify_project
if sys.argv[1] == "square":
    V.shape_plans = lambda *a, **k: []
r = verify_project([sys.argv[2]], n_random=200, build_dir=sys.argv[3])
print(len(r.mismatches))
'''


def main():
    rows = []
    for what, (rel, old, new) in MUTANTS.items():
        tmp = Path(tempfile.mkdtemp(prefix="qtrans_mutant_"))
        shutil.copytree(ROOT / "qtrans", tmp / "qtrans", ignore=shutil.ignore_patterns("__pycache__"))
        f = tmp / rel
        src = f.read_text(encoding="utf-8")
        assert old in src, f"mutation site not found in {rel}"
        f.write_text(src.replace(old, new), encoding="utf-8")
        counts = []
        for mode in ("square", "rect"):
            out = subprocess.run([sys.executable, "-c", RUN, mode, str(ROOT / "tests" / "data" / "shapes.py"),
                                  str(tmp / f"build_{mode}")], cwd=tmp, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
            if out.returncode != 0:
                raise RuntimeError(out.stderr[-2000:])
            counts.append(int(out.stdout.strip().splitlines()[-1]))
        rows.append((what, *counts))
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'mutant':58} {'square only':>12} {'with rectangular':>17}")
    for what, sq, rect in rows:
        print(f"{what:58} {sq:>12} {rect:>17}")
    return 0 if all(sq == 0 and rect > 0 for _, sq, rect in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
