"""Command line:

    python -m qtrans translate FILE.py [FILE.py ...| FOLDER] [-o OUT_DIR]
    python -m qtrans verify FILE.py [...] [--cases N] [--seed S] [--bench] [--rtol X] [--atol X] [--all]
                                        [--profile strict|stress]
    python -m qtrans bind FILE.py [...] [-o OUT_DIR] [--check] [--cases N] [--bench] [--all]
    python -m qtrans explain FILE.py[:FUNCTION] [--html OUT.html] [--json]
    python -m qtrans inspect FILE.py [--json]
    python -m qtrans doctor [--bind] [--json]
    python -m qtrans live FILE.py [--port N] [--no-browser]

A file that imports other modules next to it (`from bl_core import nearest_psd`) is translated
together with them, as a project: one header per module. A folder means every module in it.
A file with code that runs (a script, or an `if __name__ == "__main__":` block) is a program:
it also becomes <module>.cpp, the whole thing with an int main() you can compile and run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .project import translate_project
from .verify import verify_project


_ASCII = {"✓": "ok", "✗": "X", "│": "|", "↳": "->", "─": "-", "–": "-", "—": "-", "…": "...", "→": "->",
          "±": "+/-", "≤": "<=", "≥": ">=", "×": "x", "·": "."}


def _print(text: str):
    """Print, falling back to ASCII for the marks the text views use when the console can't show
    them (a Windows console on cp1252, say), instead of failing with UnicodeEncodeError."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
    except (UnicodeEncodeError, LookupError):
        for k, v in _ASCII.items():
            text = text.replace(k, v)
        text = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(text)


def _inputs(files):
    return files[0] if len(files) == 1 and Path(files[0]).is_dir() else files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qtrans", description="Translate typed quant Python into readable C++.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="write <module>.hpp (one per module), <module>.cpp for a program, and a report")
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
    v.add_argument("--strict", action="store_true",
                   help="fail when your own C++ for a function was written for an earlier version of the Python")

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

    e = sub.add_parser("explain", help="Python next to its C++, line by line, with the reason for each choice")
    e.add_argument("target", help="FILE.py, or FILE.py:FUNCTION for one function")
    e.add_argument("--html", default=None, help="write a side-by-side page here (open it in a browser)")
    e.add_argument("--json", action="store_true", help="machine-readable output")
    e.add_argument("--width", type=int, default=0, help="terminal width for the text view")

    i = sub.add_parser("inspect", help="what qtrans will make of each function, without compiling")
    i.add_argument("file")
    i.add_argument("--json", action="store_true", help="machine-readable output")

    d = sub.add_parser("doctor", help="check the compiler, Eigen and packages qtrans needs, by using them")
    d.add_argument("--bind", action="store_true", help="also build and import Python bindings")
    d.add_argument("--json", action="store_true", help="machine-readable output")

    lv = sub.add_parser("live", help="edit Python in the browser and see its C++ update as you type")
    lv.add_argument("file", help="the Python file to edit (created if it doesn't exist)")
    lv.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: any free one)")
    lv.add_argument("--no-browser", action="store_true", help="don't open the browser, just print the address")

    ov = sub.add_parser("override", help="your own C++ for a function: start it, --accept it, --remove it, --list")
    ov.add_argument("file", help="the Python file")
    ov.add_argument("function", nargs="?", default=None, help="the function (omit with --list)")
    ov.add_argument("--accept", action="store_true", help="check it against the Python and record that it agrees")
    ov.add_argument("--remove", action="store_true", help="go back to qtrans's translation")
    ov.add_argument("--list", action="store_true", help="every override of the file, and whether it is up to date")

    args = ap.parse_args(argv)
    if args.cmd == "override":
        from .overrides import OverrideError, command
        try:
            return command(args.file, args.function, accept=args.accept, remove=args.remove, show=args.list,
                           out=_print)
        except OverrideError as e:
            _print(f"qtrans override: {e}")
            return 1
    if args.cmd == "live":
        from .live import serve
        return serve(args.file, port=args.port, open_browser=not args.no_browser)
    if args.cmd == "doctor":
        from .doctor import main as doctor
        return doctor(bind=args.bind, as_json=args.json, out=_print)
    if args.cmd == "inspect":
        from .inspect_ import inspect_file, to_json as ijson, to_text as itext
        infos = inspect_file(args.file)
        _print(ijson(args.file, infos) if args.json else itext(args.file, infos))
        return 0 if all(x.translated for x in infos) else 1
    if args.cmd == "explain":
        from .explain import explain_file, to_html, to_json as ejson, to_text as etext
        path, _, fn = args.target.partition(".py:")
        path = path + ".py" if fn or not path.endswith(".py") else path
        view = explain_file(path if Path(path).exists() else args.target, fn or None)
        if args.html:
            Path(args.html).write_text(to_html(view), encoding="utf-8")
            print(f"wrote {args.html}: open it in a browser")
        elif args.json:
            _print(ejson(view))
        else:
            _print(etext(view, args.width))
        return 0
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
            r = project.modules[m]
            print(r.report.to_text())
            if r.library or r.program is None:
                print(f"wrote {out / (m + '.hpp')}")
            if r.program is not None:
                print(f"wrote {out / (m + '.cpp')}: the program, with main(); build it with")
                print(f"  g++ -std=c++17 -O2 -I {out}{' -I <Eigen>' if r.uses_arrays else ''} {out / (m + '.cpp')} -o {m}")
        return 0 if not any(project.modules[m].report.rejected for m in project.requested) else 1
    report = verify_project(inputs, n_random=args.cases, seed=args.seed, rtol=args.rtol, atol=args.atol,
                            bench=args.bench, compiler=args.cxx, build_dir=args.build, all_modules=args.all,
                            profile=args.profile, strict=args.strict)
    print(report.to_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
