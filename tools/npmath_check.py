"""Checks the runtime's NumPy maths against the running NumPy, bit for bit: qtrans.npmath_check.

    python tools/npmath_check.py [--n 200000] [--seed 0]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrans.npmath_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
