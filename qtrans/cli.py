"""Command line:

    python -m qtrans translate FILE.py [FILE.py ...| FOLDER] [-o OUT_DIR]
    python -m qtrans verify FILE.py [...] [--cases N] [--seed S] [--bench] [--rtol X] [--atol X] [--all]
                                        [--profile strict|stress]
    python -m qtrans bind FILE.py [...] [-o OUT_DIR] [--check] [--cases N] [--bench] [--all]

A file that imports other modules next to it (`from bl_core import nearest_psd`) is translated
together with them, as a project: one header per module. A folder means every module in it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .project import translate_project
from .verify import verify_project


def _inputs(files):
    return files[0] if len(files) == 1 and Path(files[0]).is_dir() else files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qtrans", description="Translate typed quant Python into readable C++.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="write <module>.hpp (one per module) and a report")
    t.add_argument("files", nargs="+", help="Python files, or one folder")
    t.add_argument("-o", "--out", default=None, help="output folder (default: cpp_out next to the first file)")

    v = sub.add_parser("verify", help="compile the C++ and check it against Python")
    v.add_argument("files", nargs="+", help="Python files, or one folder")
    v.add_argument("--cases", type=int, default=200, help="random cases per function (default 200)")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--rtol", type=float, default=1e-10)
    v.add_argument("--atol", type=float, default=1e-13)
    v.add_argument("--bench", action="store_true", help="also time Python vs C++ per call")
    v.add_argument("--all", action="store_true", help="also check the modules the files import")
    v.add_argument("--profile", choices=["strict", "stress"], default="strict",
                   help="stress: also inf, NaN, subnormals, the largest doubles and longer arrays")
    v.add_argument("--cxx", default=None, help="C++ compiler to use")
    v.add_argument("--build", default=None, help="folder for the generated driver and binary")

    b = sub.add_parser("bind", help="compile the C++ into a Python module, <module>_cpp (needs pybind11)")
    b.add_argument("files", nargs="+", help="Python files, or one folder")
    b.add_argument("-o", "--out", default=None, help="where to put the module (default: next to the file)")
    b.add_argument("--check", action="store_true", help="check the compiled module against the Python")
    b.add_argument("--cases", type=int, default=200, help="random cases per function for --check (default 200)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--rtol", type=float, default=1e-10)
    b.add_argument("--atol", type=float, default=1e-13)
    b.add_argument("--bench", action="store_true", help="with --check: also time Python vs compiled per call")
    b.add_argument("--all", action="store_true", help="with --check: also check the modules the files import")
    b.add_argument("--build", default=None, help="folder for the generated sources and build files")

    args = ap.parse_args(argv)
    inputs = _inputs(args.files)
    if args.cmd == "bind":
        from .bind import build_extension, check_bindings
        ext = build_extension(inputs, out_dir=args.out, build_dir=args.build)
        print(f"wrote {ext.path} (built in {ext.seconds:.1f} s): import {ext.name}")
        print(f"bound {len(ext.bound)} function(s) and {len(ext.bound_classes)} class(es)")
        if not ext.single:
            print(f"  one submodule per module: {', '.join(f'{ext.name}.{m}' for m in ext.project.order)}")
        if not args.check:
            for k, ns in ext.notes.items():
                for n in ns:
                    print(f"  {k}: {n}")
            return 0
        report = check_bindings(inputs, n_random=args.cases, seed=args.seed, rtol=args.rtol, atol=args.atol,
                                bench=args.bench, ext=ext, all_modules=args.all)
        print(report.to_text())
        return 0 if report.ok else 1
    if args.cmd == "translate":
        first = Path(args.files[0]).resolve()
        out = Path(args.out) if args.out else (first if first.is_dir() else first.parent) / "cpp_out"
        project = translate_project(inputs, out)
        for m in project.order:
            print(project.modules[m].report.to_text())
            print(f"wrote {out / (m + '.hpp')}")
        return 0 if not any(project.modules[m].report.rejected for m in project.requested) else 1
    report = verify_project(inputs, n_random=args.cases, seed=args.seed, rtol=args.rtol, atol=args.atol,
                            bench=args.bench, compiler=args.cxx, build_dir=args.build, all_modules=args.all,
                            profile=args.profile)
    print(report.to_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
