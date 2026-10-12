"""python -m tools.linalg_check [--seed S] [--per-band K] [--out FILE.md]: see __init__."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.linalg_check import check, report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.linalg_check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-band", type=int, default=4, help="matrices per size, condition number and kind")
    ap.add_argument("--out", default=None, help="also write the report here (Markdown)")
    args = ap.parse_args(argv)
    cases, cxx = check(args.seed, args.per_band)
    text, failures = report(cases, cxx, args.seed, args.per_band)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
