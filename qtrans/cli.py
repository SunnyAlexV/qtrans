"""Command line:

    python -m qtrans translate FILE.py [-o OUT_DIR]
    python -m qtrans verify FILE.py [--cases N] [--seed S] [--bench] [--rtol X] [--atol X]
    python -m qtrans bind FILE.py [-o OUT_DIR] [--check] [--cases N] [--bench]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .translator import translate_file
from .verify import verify_file


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qtrans", description="Translate typed quant Python into readable C++.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="write <module>.hpp and a report")
    t.add_argument("file")
    t.add_argument("-o", "--out", default=None, help="output folder (default: cpp_out next to the file)")

    v = sub.add_parser("verify", help="compile the C++ and check it against Python")
    v.add_argument("file")
    v.add_argument("--cases", type=int, default=200, help="random cases per function (default 200)")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--rtol", type=float, default=1e-10)
    v.add_argument("--atol", type=float, default=1e-13)
    v.add_argument("--bench", action="store_true", help="also time Python vs C++ per call")
    v.add_argument("--cxx", default=None, help="C++ compiler to use")
    v.add_argument("--build", default=None, help="folder for the generated driver and binary")

    b = sub.add_parser("bind", help="compile the C++ into a Python module, <module>_cpp (needs pybind11)")
    b.add_argument("file")
    b.add_argument("-o", "--out", default=None, help="where to put the module (default: next to the file)")
    b.add_argument("--check", action="store_true", help="check the compiled module against the Python")
    b.add_argument("--cases", type=int, default=200, help="random cases per function for --check (default 200)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--rtol", type=float, default=1e-10)
    b.add_argument("--atol", type=float, default=1e-13)
    b.add_argument("--bench", action="store_true", help="with --check: also time Python vs compiled per call")
    b.add_argument("--build", default=None, help="folder for the generated sources and build files")

    args = ap.parse_args(argv)
    if args.cmd == "bind":
        from .bind import build_extension, check_bindings
        ext = build_extension(args.file, out_dir=args.out, build_dir=args.build)
        print(f"wrote {ext.path} (built in {ext.seconds:.1f} s): import {ext.name}")
        print(f"bound {len(ext.binder.bound)} function(s) and {len(ext.binder.bound_classes)} class(es)")
        if not args.check:
            for k, ns in ext.binder.notes.items():
                for n in ns:
                    print(f"  {k}: {n}")
            return 0
        report = check_bindings(args.file, n_random=args.cases, seed=args.seed, rtol=args.rtol, atol=args.atol,
                                bench=args.bench, ext=ext)
        print(report.to_text())
        return 0 if report.ok else 1
    if args.cmd == "translate":
        out = Path(args.out) if args.out else Path(args.file).resolve().parent / "cpp_out"
        result = translate_file(args.file, out)
        print(result.report.to_text())
        print(f"wrote {out / (Path(args.file).stem + '.hpp')}")
        return 0 if not result.report.rejected else 1
    report = verify_file(args.file, n_random=args.cases, seed=args.seed, rtol=args.rtol,
                         atol=args.atol, bench=args.bench, compiler=args.cxx, build_dir=args.build)
    print(report.to_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
