"""Backs the README's Export numbers: exports the Black-Litterman app (examples/bl_app, four
modules) as a CMake package, records Python's results (50 cases per function, seed 0), and
builds and tests it with CMake.

    python tools/export_bl_app.py [out_folder]
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qtrans.export import export  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="bl_app_cpp_"))
    for line in export(ROOT / "examples" / "bl_app", out, name="bl_app", cases=50):
        print(line)
