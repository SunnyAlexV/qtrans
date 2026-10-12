"""`qtrans doctor`: is everything qtrans needs on this machine, and does it work?

    python -m qtrans doctor [--bind] [--json]

Each check reports ok, warn (something optional is missing or old) or fail (qtrans can't do its
job), with the fix. The last checks don't just look for files: they translate a small module
with arrays, linear algebra and random numbers, compile it, and verify it against Python, which
is what `qtrans verify` does; --bind also builds and imports the Python bindings.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"
WINDOWS = os.name == "nt"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _version_tuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _package(name, minimum=None, needed_for="", required=False):
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return Check(name, FAIL if required else WARN, "not installed" + (f" (needed for {needed_for})" if needed_for else ""),
                     f"python -m pip install {name}" + (" (or conda install " + name + ")" if not WINDOWS else ""))
    v = getattr(mod, "__version__", "?")
    if minimum and _version_tuple(v) < _version_tuple(minimum):
        return Check(name, FAIL if required else WARN, f"{v}, older than {minimum}" + (f" (needed for {needed_for})" if needed_for else ""),
                     f"python -m pip install --upgrade {name}")
    return Check(name, OK, v + (f" ({needed_for})" if needed_for else ""))


def _eigen_version(folder: Path) -> str:
    macros = folder / "Eigen" / "src" / "Core" / "util" / "Macros.h"
    try:
        text = macros.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "?"
    nums = [re.search(rf"#define EIGEN_{k}_VERSION\s+(\d+)", text) for k in ("WORLD", "MAJOR", "MINOR")]
    return ".".join(m.group(1) for m in nums) if all(nums) else "?"


SAMPLE = textwrap.dedent('''
    import numpy as np
    from qtrans.typing import Mat, Vec


    def ridge(A: Mat, b: Vec, lam: float) -> Vec:
        n = A.shape[1]
        return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)


    def simulate(seed: int, n: int) -> float:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        return float(np.mean(np.maximum(x, 0.0)))


    def floor_div(a: int, b: int) -> int:
        return a // b


    QTRANS_CASES = {"floor_div": [(-7, 2), (7, -2)], "simulate": [(42, 1000)]}
''')


def run(bind: bool = False) -> list:
    checks = []
    py = sys.version_info
    checks.append(Check("Python", OK if py >= (3, 10) else FAIL,
                        f"{platform.python_version()} ({sys.executable})",
                        "" if py >= (3, 10) else "qtrans needs Python 3.10 or newer"))
    checks.append(_package("numpy", "1.24", "the translated code's reference", required=True))
    checks.append(_package("scipy", "1.10", "scipy.stats.norm and friends; scipy.linalg in tools"))
    checks.append(_package("pybind11", "2.10", "qtrans bind"))
    checks.append(_package("setuptools", None, "qtrans bind"))
    checks.append(_package("mpmath", None, "the tests' high-precision references"))

    from .verify import extra_flags, find_compiler, find_eigen
    cxx = None
    try:
        cxx = find_compiler()
        version = cxx.version()
        checks.append(Check("C++ compiler", OK, f"{cxx.path} ({version or cxx.kind})"))
    except RuntimeError as e:
        checks.append(Check("C++ compiler", FAIL, "none found", str(e)))
    eigen = None
    try:
        eigen = find_eigen()
        v = _eigen_version(eigen)
        old = v != "?" and _version_tuple(v) < (3, 3, 0)
        checks.append(Check("Eigen", WARN if old else OK, f"{v} in {eigen}",
                            "qtrans is tested with Eigen 3.4; upgrade it" if old else ""))
    except RuntimeError as e:
        checks.append(Check("Eigen", FAIL, "not found", str(e)))
    env = {k: os.environ[k] for k in ("CXX", "QTRANS_EIGEN", "QTRANS_CXXFLAGS") if os.environ.get(k)}
    if env:
        checks.append(Check("environment", OK, ", ".join(f"{k}={v}" for k, v in env.items())))
    if extra_flags():
        checks.append(Check("QTRANS_CXXFLAGS", WARN, " ".join(extra_flags()),
                            "added to every verification's compile command; unset it unless you mean it"))

    if cxx is not None:
        checks.append(_cxx17(cxx))
    if cxx is not None and eigen is not None:
        checks.append(_end_to_end())
    if bind:
        checks.append(_bind())
    elif cxx is not None:
        checks.append(Check("bindings", OK, "not checked (qtrans doctor --bind builds and imports a small module)"))
    return checks


def _cxx17(cxx) -> Check:
    src = "#include <optional>\n#include <tuple>\nint main() { auto [a, b] = std::tuple<int, int>{1, 2}; " \
          "std::optional<int> o = a + b; return *o == 3 ? 0 : 1; }\n"
    with tempfile.TemporaryDirectory(prefix="qtrans_doctor_") as d:
        f = Path(d) / "t.cpp"
        f.write_text(src, encoding="utf-8")
        exe = Path(d) / ("t.exe" if WINDOWS else "t")
        r = subprocess.run(cxx.command(f, exe, d), capture_output=True, text=True, cwd=d)
        if r.returncode != 0:
            return Check("C++17", FAIL, f"{cxx.path} couldn't compile a C++17 test:\n{(r.stderr or r.stdout)[-600:]}",
                         "install a newer compiler (g++ 9 or later, clang 10, MSVC 2019), or point CXX at one")
        rr = subprocess.run([str(exe)], capture_output=True)
        if rr.returncode != 0:
            return Check("C++17", FAIL, "the test program compiled but didn't run correctly",
                         "on Windows with MinGW, its bin folder must be on PATH (the C++ runtime DLLs)")
    return Check("C++17", OK, "compiled and ran a test program")


def _end_to_end() -> Check:
    from .verify import verify_file
    with tempfile.TemporaryDirectory(prefix="qtrans_doctor_") as d:
        f = Path(d) / "doctor_sample.py"
        f.write_text(SAMPLE, encoding="utf-8")
        try:
            rep = verify_file(f, n_random=20, build_dir=Path(d) / "build")
        except Exception as e:  # noqa: BLE001 - the point is to report whatever goes wrong
            return Check("translate, compile, verify", FAIL, f"{type(e).__name__}: {str(e)[-800:]}",
                         "see the message; `qtrans verify` would fail the same way")
        cases = sum(len(fc.cases) for fc in rep.functions)
        if not rep.ok:
            return Check("translate, compile, verify", FAIL, rep.to_text()[-800:],
                         "this is a bug or a platform difference: please report it with this output")
        warn = f", {len(rep.warnings)} compiler warnings" if rep.warnings else ""
        return Check("translate, compile, verify", WARN if rep.warnings else OK,
                     f"a sample with arrays, np.linalg.solve, NumPy's random generator and int // : "
                     f"{cases} cases, all agree with Python{warn}",
                     "the generated code should compile without warnings: please report these" if warn else "")


def _bind() -> Check:
    try:
        import pybind11  # noqa: F401
    except ImportError:
        return Check("bindings", FAIL, "pybind11 is not installed", "python -m pip install pybind11")
    from .bind import build_extension
    with tempfile.TemporaryDirectory(prefix="qtrans_doctor_") as d:
        f = Path(d) / "doctor_bind.py"
        f.write_text(SAMPLE, encoding="utf-8")
        try:
            ext = build_extension(f, out_dir=d, build_dir=Path(d) / "build")
            sys.path.insert(0, d)
            try:
                mod = importlib.import_module(ext.name)
                got = mod.floor_div(-7, 2)
            finally:
                sys.path.remove(d)
        except Exception as e:  # noqa: BLE001
            fix = "on Windows the bindings need MSVC (Visual Studio Build Tools, 'Desktop development with C++')" \
                if WINDOWS else "a compiler that can build Python extensions, and Python's development headers"
            return Check("bindings", FAIL, f"{type(e).__name__}: {str(e)[-800:]}", fix)
        if got != -4:
            return Check("bindings", FAIL, f"floor_div(-7, 2) gave {got}, not -4", "please report this")
    return Check("bindings", OK, "built, imported and called a compiled module")


def to_text(checks) -> str:
    mark = {OK: "✓", WARN: "!", FAIL: "✗"}
    out = ["qtrans doctor", ""]
    for c in checks:
        lines = c.detail.splitlines() or [""]
        out.append(f"  {mark[c.status]} {c.name:<28} {lines[0]}")
        out += [f"    {'':<28} {x}" for x in lines[1:]]
        if c.fix and c.status != OK:
            out.append(f"    {'':<28} fix: {c.fix}")
    bad = [c for c in checks if c.status == FAIL]
    warn = [c for c in checks if c.status == WARN]
    out += ["", "  " + ("everything qtrans needs works" if not bad else f"{len(bad)} problem(s) to fix")
            + (f" ({len(warn)} warning(s))" if warn else "")]
    return "\n".join(out)


def to_json(checks) -> str:
    return json.dumps([asdict(c) for c in checks], indent=2)


def main(bind=False, as_json=False, out=print) -> int:
    checks = run(bind)
    out(to_json(checks) if as_json else to_text(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0
