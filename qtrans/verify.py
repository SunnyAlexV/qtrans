"""Differential testing of the translation: run every translated function in Python and in
compiled C++ on the same inputs and compare. Agreement on every case run is strong evidence,
not a proof: no finite set of cases rules out a difference elsewhere.

How the comparison stays exact
  * Inputs go to C++ as hexadecimal floats (float.hex), outputs come back via printf("%a").
    Hex floats are exact, so no rounding happens in transit: any difference reported is a
    real difference in the computation.
  * The C++ is compiled with -ffp-contract=off so the compiler may not fuse a*b+c into one
    FMA instruction (which rounds once instead of twice, as Python does).

Outcome of each test case
  match     same value (within rtol/atol) or the same exception category
  ill-cond  values differ by more than rtol, but by no more than 100x the amount Python's
            own answer moves when the inputs are nudged by 4 ULPs. The problem itself is
            ill-conditioned there (e.g. inverting a nearly singular matrix), so last-bit
            differences between NumPy/LAPACK and Eigen are amplified: two backward-stable
            implementations can differ by that much (a higher-precision one could do better).
  known     Python raised one of the float errors IEEE-754 arithmetic doesn't raise
            (ZeroDivisionError, OverflowError, math domain error) and C++ returned a
            value. Listed in the README as a deliberate difference.
  mismatch  anything else. A correct translation has zero of these.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import dataclasses
import importlib.util
import inspect
import math
import random
import os
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .translator import (NPMATH_HEADERS, ARRAYS_HEADER, LIST_HEADER, UNICODE_HEADER, PYTHON_HEADER, PYRANDOM_HEADER, PRINT_HEADER, RANDOM_HEADER, RUNTIME_HEADER, TranslationResult,
                         library_tree, translate_file)
from .types import BOOL, FLOAT, INT, IVEC, MAT, RNG, STR, VEC, VOID, ArrayType, DictType, IntArrayType, ListType, OptionalType, StructType, TupleType

# Which Python exceptions a given C++ exception class stands for.
CPP_EXC_MATCHES = {
    "ValueError": {"ValueError"},
    "ArithmeticError": {"ArithmeticError", "ZeroDivisionError", "FloatingPointError"},
    "OverflowError": {"OverflowError"},
    "AssertionError": {"AssertionError"},
    "RuntimeError": {"RuntimeError", "NotImplementedError", "Exception"},
    "IndexError": {"IndexError"},
    "KeyError": {"KeyError"},  # a dict without the key (qt::py::KeyError)
    "TypeError": {"TypeError"},
    # an int result beyond 64 bits (Python returns a big int there; see classify)
    "IntOverflow": {"OverflowError"},
    # Eigen's internal shape / bounds checks can't tell NumPy's ValueError and IndexError apart
    "ShapeError": {"ValueError", "IndexError"},
}
# Python's float errors, by message. Python 3.14 words most math domain errors by what the
# function expected and the argument it got: math.sqrt(-1) -> "expected a nonnegative input,
# got -1.0", math.acos(2) -> "expected a number in range from -1 up to 1, got 2.0",
# math.sin(inf) -> "expected a finite input, got inf" (math.pow and math.fmod still say
# "math domain error").
_KNOWN_MESSAGES = ("math domain error", "math range error", "cannot convert float")
_MATH_314 = re.compile(r"^expected .+, got [-+]?(inf|nan|[0-9][0-9.e+-]*)$")


def _float_error_message(msg: str) -> bool:
    return any(m in msg for m in _KNOWN_MESSAGES) or bool(_MATH_314.match(msg))


# --- compiler ------------------------------------------------------------------------
def extra_flags() -> list:
    """QTRANS_CXXFLAGS: flags added to the verifier's compile command, after its own (so -O0 or
    -O3 replaces -O2), e.g. "-fsanitize=address,undefined" in CI. Not used for `qtrans bind`."""
    return shlex.split(os.environ.get("QTRANS_CXXFLAGS", ""), posix=os.name != "nt")


@dataclass
class Compiler:
    path: str
    kind: str  # "gnu" (g++/clang++) or "msvc"

    def command(self, source, exe, include_dir, system_includes=(), optimise=True, defines=()):
        if self.kind == "msvc":
            return ([self.path, "/nologo", "/std:c++17", "/EHsc", "/W4", "/fp:precise", "/bigobj",
                     "/O2" if optimise else "/Od", f"/I{include_dir}"] + [f"/D{d}" for d in defines]
                    + [f"/external:I{d}" for d in system_includes] + ["/external:W0"] + extra_flags()
                    + [str(source), f"/Fe{exe}"])
        # Eigen goes in as a *system* include so its own internal warnings don't count against
        # the generated code
        return ([self.path, "-std=c++17", "-O2" if optimise else "-O0", "-Wall", "-Wextra",
                 "-Wpedantic", "-Wshadow", "-Wconversion", "-Wdouble-promotion",
                 "-ffp-contract=off", f"-I{include_dir}"] + [f"-D{d}" for d in defines]
                + [f"-isystem{d}" for d in system_includes] + extra_flags() + [str(source), "-o", str(exe)])

    def version(self) -> str:
        """The compiler's own version line (g++ --version; cl prints it to stderr)."""
        try:
            r = subprocess.run([self.path] + ([] if self.kind == "msvc" else ["--version"]),
                               capture_output=True, text=True, timeout=30)
            out = (r.stdout if self.kind != "msvc" else r.stderr) or r.stdout or r.stderr
            return out.strip().splitlines()[0].strip() if out.strip() else Path(self.path).name
        except (OSError, subprocess.SubprocessError):
            return Path(self.path).name


def cpu_model() -> str:
    """The CPU's name: glibc, NumPy and OpenBLAS pick their exp / log / matrix-product kernels for
    it (FMA, AVX2, AVX-512), so the last bit of such results can change with it."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.startswith("model name"):
                    return " ".join(line.split(":", 1)[1].split())
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        return platform.processor()
    except Exception:  # noqa: BLE001 - only a label
        return ""


def environment(cxx: "Compiler | None" = None) -> str:
    """What produced a result: OS, CPU, Python, NumPy, SciPy, the compiler and extra flags.
    Results that depend on the platform's maths library (exp, pow, erf) hold only for it."""
    parts = [f"{platform.system()} {platform.release()} {platform.machine()}"
             + (f" ({cpu_model()})" if cpu_model() else ""),
             f"Python {platform.python_version()}", f"NumPy {np.__version__}"]
    try:
        import scipy
        parts.append(f"SciPy {scipy.__version__}")
    except ImportError:
        pass
    if cxx is not None:
        parts.append(cxx.version())
    if extra_flags():
        parts.append("QTRANS_CXXFLAGS=" + " ".join(extra_flags()))
    return "; ".join(parts)


def find_eigen(explicit: str | None = None) -> Path:
    """Eigen's include folder (the one containing Eigen/Dense)."""
    candidates = [explicit, os.environ.get("QTRANS_EIGEN"),
                  Path(sys.prefix) / "include" / "eigen3",             # conda on Linux / macOS
                  Path(sys.prefix) / "Library" / "include" / "eigen3",  # conda on Windows
                  "/usr/include/eigen3", "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3"]
    for c in candidates:
        if c and (Path(c) / "Eigen" / "Dense").exists():
            return Path(c)
    raise RuntimeError(
        "Array code needs the Eigen headers, which weren't found. Install them with "
        "`conda install -c conda-forge eigen` (or apt install libeigen3-dev / brew install eigen), "
        "or set QTRANS_EIGEN to the folder that contains Eigen/Dense.")


def _windows_gcc_candidates():
    """Where a MinGW-w64 g++ usually is on Windows when it isn't on the PATH: unzipped from
    winlibs.com, MSYS2, TDM-GCC, Strawberry Perl, or a conda environment."""
    home = Path(os.environ.get("USERPROFILE", "~")).expanduser()
    roots = [Path("C:/mingw64"), Path("C:/winlibs/mingw64"), Path("C:/msys64/ucrt64"), Path("C:/msys64/mingw64"),
             Path("C:/TDM-GCC-64"), Path("C:/Strawberry/c"), Path("C:/Program Files/mingw64"),
             home / "mingw64", Path(sys.prefix) / "Library" / "mingw-w64", Path(sys.prefix) / "Library"]
    roots += sorted(Path("C:/").glob("*/mingw64")) + sorted(home.glob("*/mingw64")) + sorted(home.glob("*/*/mingw64"))
    for r in roots:
        exe = r / "bin" / "g++.exe"
        try:
            if exe.is_file():
                yield exe
        except OSError:
            continue


def find_compiler(explicit: str | None = None) -> Compiler:
    candidates = [explicit, os.environ.get("CXX"), "g++", "clang++", "c++", "cl"]
    for c in candidates:
        if not c:
            continue
        path = shutil.which(c)
        if path:
            kind = "msvc" if Path(path).stem.lower() == "cl" else "gnu"
            return Compiler(path, kind)
    if os.name == "nt":
        for exe in _windows_gcc_candidates():
            # its folder goes on the PATH of what qtrans runs: a MinGW program needs the
            # compiler's own DLLs (libstdc++, libgcc) next to it or on the PATH to start
            os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")
            return Compiler(str(exe), "gnu")
    raise RuntimeError(NO_COMPILER)


NO_COMPILER = (
    "No C++ compiler found. On Windows the simplest is GCC from winlibs.com: under Release versions, "
    "UCRT runtime, take the newest GCC (with POSIX threads) and, on its Win64 line, the 'Zip archive' "
    "(not 7-Zip, not Win32). Extract it anywhere: C:\\, your Desktop or Downloads (Windows' Extract All "
    "adds a folder named after the zip; that's fine), and qtrans finds mingw64\\bin\\g++.exe there, "
    "no PATH changes needed. Or MSYS2's mingw-w64-ucrt-x86_64-gcc, or Visual Studio Build Tools. "
    "On Linux: apt install g++; on macOS: xcode-select --install. Or set the CXX environment variable "
    "to the compiler's path.")


def build_hint(cpp, exe_name: str, include_dir, arrays: bool = False) -> str:
    """The command that builds a translated program as verify built it: the compiler qtrans would
    use (its full path when it isn't on the PATH), C++17, and no fused multiply-add (the results
    are Python's only without it)."""
    path_before = os.environ.get("PATH", "")
    try:
        comp = find_compiler()
    except RuntimeError:
        comp = None
    eigen = None
    if arrays:
        try:
            eigen = str(find_eigen())
        except RuntimeError:
            eigen = "<the folder holding Eigen/Dense>"

    def q(x):
        x = str(x)
        return f'"{x}"' if " " in x else x
    windows = os.name == "nt"
    exe = exe_name + (".exe" if windows else "")
    if comp is not None and comp.kind == "msvc":
        return " ".join(["cl /nologo /std:c++17 /EHsc /O2 /fp:precise", f"/I {q(include_dir)}"]
                        + ([f"/I {q(eigen)}"] if eigen else []) + [q(cpp), f"/Fe:{exe}"])
    if comp is None:
        cxx = "g++"
    else:
        name = Path(comp.path).name
        on_path = shutil.which(name, path=path_before)
        cxx = name if on_path and Path(on_path).resolve() == Path(comp.path).resolve() else q(comp.path)
        if cxx.lower().endswith(".exe") and cxx == name:
            cxx = name[:-4]
    flags = ["-std=c++17", "-O2", "-ffp-contract=off"] + (["-static"] if windows else [])
    return " ".join([cxx] + flags + [f"-I {q(include_dir)}"] + ([f"-I {q(eigen)}"] if eigen else [])
                    + [q(cpp), "-o", exe])


# --- results ---------------------------------------------------------------------------
@dataclass
class Case:
    func: str
    args: tuple
    source: str  # "given" (QTRANS_CASES) or "random"
    py: tuple = None
    cpp: tuple = None
    outcome: str = ""
    rel_err: float = 0.0
    ulps: int = 0
    tiny: bool = False  # passed only because |difference| <= atol (values near zero)
    sensitivity: float = 0.0  # Python's own relative change under a 4-ULP input nudge
    order: bool = False  # 'known' because the input breaks two checks at once (see classify)
    audit: tuple | None = None  # (numpy.linalg operations audited, worst residual / bound, which)
    py_out: str = ""   # what the call printed, in Python
    cpp_out: str = ""  # and in C++
    printed_differs: bool = False

    @property
    def unstable(self) -> bool:
        """A numpy.linalg operation of this call left a residual above its stability bound."""
        return self.audit is not None and self.audit[0] > 0 and self.audit[1] > 1.0

    @property
    def certified(self) -> bool:
        """Every numpy.linalg operation of this call (at least one) was backward stable."""
        return self.audit is not None and self.audit[0] > 0 and self.audit[1] <= 1.0


@dataclass
class FunctionCheck:
    name: str
    cases: list = field(default_factory=list)
    py_ns: float | None = None
    cpp_ns: float | None = None

    def count(self, outcome):
        return sum(1 for c in self.cases if c.outcome == outcome)

    rtol: float = 0.0
    atol: float = 0.0
    key: str = ""      # how the driver names it ("module:function" in a project)
    module: str = ""
    py_name: str = ""  # the Python function (a version, "half(float)", names its function "half")
    shape_plans: list = field(default_factory=list)  # the non-square shapes it was tried on
    timeouts: int = 0  # cases left out because Python didn't finish one in time
    py_rounds: list = field(default_factory=list)   # bench=N: ns per call, each round (py_ns: their median)
    cpp_rounds: list = field(default_factory=list)
    bench_case: object = None  # the case timed
    py_reps: list = field(default_factory=list)    # bench=N: the calls timed in each round
    cpp_reps: list = field(default_factory=list)
    override: object = None    # your own C++ is checked here: its overrides.Override
    reference: object = None   # qtrans's own translation, on the very same inputs (a FunctionCheck)
    vs_reference: tuple = ()   # (cases where both returned, bit-identical among them, max ULP apart)

    @property
    def max_rel_err(self):
        return max((c.rel_err for c in self.cases if c.outcome == "match" and not c.tiny), default=0.0)

    @property
    def max_ulps(self):
        return max((c.ulps for c in self.cases if c.outcome == "match" and not c.tiny), default=0)

    @property
    def n_values(self):
        return sum(1 for c in self.cases if c.outcome == "match" and c.py[0] == "ok")


@dataclass
class VerifyReport:
    file: str
    translation: TranslationResult
    compiler: str
    compile_command: str
    warnings: list
    functions: list
    rtol: float
    atol: float
    environment: str = ""
    program: ProgramCheck | None = None  # a program (top-level code): its run against the compiled one
    strict: bool = False  # an override written for an earlier version of the Python fails the check

    @property
    def mismatches(self):
        return [c for f in self.functions for c in f.cases if c.outcome == "mismatch"]

    @property
    def stale_overrides(self):
        return [f.name for f in self.functions if f.override is not None and f.override.status != "up to date"]

    @property
    def ok(self):
        return not self.mismatches and (self.program is None or self.program.ok) and \
            not (self.strict and self.stale_overrides)

    def _override_rows(self, f, bench) -> list:
        """Under an overridden function: that it is your C++, its status, and how qtrans's own
        translation did on the same inputs (and how far apart the two C++ versions are)."""
        ov = f.override
        out = [f"    - your C++ ({ov.status}, {getattr(self.translation, 'overrides_file', None).name}, line {ov.line})"]
        r = f.reference
        if r is None:
            why = getattr(ov, "qtrans_could", None)
            out.append("      qtrans can't translate this function itself" + (f" ({why})" if why else "")
                       + ": only your C++ is checked")
            return out
        row = (f"    {'qtrans translation':<24}{len(r.cases):>6}{r.count('match'):>7}{r.count('illcond'):>9}"
               f"{r.count('known'):>7}{r.count('mismatch'):>9}{r.max_rel_err:>13.1e}"
               f"{r.max_ulps if r.max_ulps < 10**7 else float(r.max_ulps):>9{'' if r.max_ulps < 10**7 else '.1e'}}")
        if bench and r.cpp_ns and f.cpp_ns:
            row += f"{r.py_ns:>11.0f}{r.cpp_ns:>9.1f}{r.py_ns / r.cpp_ns:>9.0f}x"
        out.append(row)
        if f.vs_reference:
            both, same, ulps = f.vs_reference
            text = f"      your C++ against qtrans's, on the same inputs: {same} of {both} bit-identical"
            if same < both:
                text += f", at most {ulps} ULP apart"
            if bench and r.cpp_ns and f.cpp_ns:
                text += f"; yours takes {f.cpp_ns / r.cpp_ns:.2f}x the time"
            out.append(text)
        return out

    def to_text(self) -> str:
        t = self.translation.report
        lines = [f"qtrans verify: {self.file}",
                 f"  translation: {t.summary()}",
                 f"  compiler:    {self.compiler} ({len(self.warnings)} warnings in generated code)",
                 f"  tolerance:   rtol={self.rtol:g}, atol={self.atol:g}"]
        if self.environment:
            lines.append(f"  platform:    {self.environment}")
        lines.append("")
        bench = any(f.cpp_ns for f in self.functions)
        head = (f"  {'function':<26}{'cases':>6}{'match':>7}{'ill-cond':>9}{'known':>7}{'MISMATCH':>9}"
                f"{'max rel err':>13}{'max ULP':>9}")
        if bench:
            head += f"{'Python ns':>11}{'C++ ns':>9}{'speed-up':>10}"
        lines.append(head)
        for f in self.functions:
            row = (f"  {f.name:<26}{len(f.cases):>6}{f.count('match'):>7}{f.count('illcond'):>9}"
                   f"{f.count('known'):>7}{f.count('mismatch'):>9}{f.max_rel_err:>13.1e}"
                   f"{f.max_ulps if f.max_ulps < 10**7 else float(f.max_ulps):>9{'' if f.max_ulps < 10**7 else '.1e'}}")
            if bench and f.cpp_ns:
                row += f"{f.py_ns:>11.0f}{f.cpp_ns:>9.1f}{f.py_ns / f.cpp_ns:>9.0f}x"
            if f.rtol != self.rtol or f.atol != self.atol:
                row += f"   (QTRANS_TOLERANCE: rtol={f.rtol:g})"
            lines.append(row)
            if f.override is not None:
                lines += self._override_rows(f, bench)
        if self.mismatches:
            lines += ["", "  First mismatches:"]
            for c in self.mismatches[:10]:
                if c.unstable:
                    lines.append(f"    {c.func}{c.args}: {c.audit[2]} left a residual {c.audit[1]:.3g} times its "
                                 "stability bound (C++ " + str(c.cpp)[:200] + ")")
                elif c.printed_differs:
                    lines.append(f"    {c.func}{c.args}: printed {c.py_out[:200]!r} in Python, {c.cpp_out[:200]!r} in C++")
                else:
                    lines.append(f"    {c.func}{c.args}: Python {c.py}, C++ {c.cpp}")
        audited = [c for f in self.functions for c in f.cases if c.audit is not None and c.audit[0] > 0]
        if audited:
            worst = max(audited, key=lambda c: c.audit[1])
            ops = sum(c.audit[0] for c in audited)
            special = [c for c in audited if c.outcome in ("illcond", "edge")]
            lines += ["", f"  linalg audit: {len(audited)} cases ran {ops} numpy.linalg operations; the largest residual",
                      f"  was {worst.audit[1]:.3f} of its backward-stability bound ({worst.audit[2]}, in {worst.func}).",
                      "  At most 1: each result is the exact answer for a matrix within rounding of its input."]
            if special:
                ok = sum(1 for c in special if c.certified)
                lines.append(f"  The {len(special)} ill-cond / edge cases among them: {ok} with every operation within its bound.")
        lines += ["", "  max rel err / max ULP exclude results within atol of each other (values near zero)",
                  "  and ill-conditioned cases; for arrays, rel err is normwise and ULPs count elements",
                  "  >= 1% of the array's largest value."]
        for f in self.functions:
            if f.timeouts:
                lines += ["", f"  {f.name}: {f.timeouts} case(s) left out: Python didn't finish a call within "
                              f"{PYTHON_CASE_SECONDS:g} s (it hung), so there is nothing to compare; after a hang on",
                          "  inf / NaN / near-overflow inputs, its other such cases are skipped"]
        order = [c for f in self.functions for c in f.cases if c.order]
        if order:
            c = order[0]
            lines += ["", f"  {len(order)} 'known' case(s) break two checks in one expression, and the two sides "
                          "report different ones (Python",
                      "  raises the first, left to right; C++ leaves the order of a call's arguments unspecified),",
                      f"  e.g. {c.func}: Python {c.py[1]} ({c.py[2][:60]}), C++ {c.cpp[1]}"]
        edge = [c for f in self.functions for c in f.cases if c.outcome == "edge"]
        if edge:
            c = edge[0]
            lines += ["", f"  'edge': {len(edge)} case(s) of linear algebra on inf, NaN, subnormal or near-overflow inputs,",
                      "  where NumPy's BLAS/LAPACK and Eigen add in different orders (inf - inf -> NaN on one side,",
                      f"  an intermediate overflow on the other). Not counted as mismatches. First: {c.func}"]
        ill = [c for f in self.functions for c in f.cases if c.outcome == "illcond"]
        if ill:
            worst = max(ill, key=lambda c: c.sensitivity)
            lines += [f"  'ill-cond': {len(ill)} cases where a rounding-sized nudge of the inputs (4 ULPs of each",
                      f"  value; for functions using np.linalg, also of a matrix's largest value) moves Python's",
                      f"  own answer by up to {worst.sensitivity:.1e} (relative); C++ agreed within 100x that."]
        known = [c for f in self.functions for c in f.cases if c.outcome == "known"]
        ints = [c for c in known if c.cpp[0] == "exc" and c.cpp[1] == "IntOverflow"]
        floats = [c for c in known if c not in ints and not c.order]
        if floats:
            c = floats[0]
            lines += ["", f"  'known' = Python raised a float error where IEEE-754 C++ returns inf/NaN,",
                      f"  e.g. {c.func}{c.args}: Python {c.py[1]} ({c.py[2] if len(c.py) > 2 else ''}), "
                      f"C++ {c.cpp[1]}"]
        if ints:
            c = ints[0]
            py = f"{c.py[1]}" if c.py[0] == "ok" else f"{c.py[1]} ({c.py[2]})"
            lines += ["", f"  'known' also counts {len(ints)} case(s) where an int left 64 bits: Python's int grows,",
                      f"  C++ raises OverflowError, e.g. {c.func}{c.args}: Python {py}"]
        if self.program is not None:
            lines += ["", self.program.to_text().replace("{file}", Path(self.file).name)]
        if self.warnings:
            lines += ["", "  Compiler warnings in generated code:"] + [f"    {w}" for w in self.warnings[:10]]
        bad = len(self.mismatches) + (0 if self.program is None or self.program.ok else 1)
        result = "PASS (no mismatches)" if self.ok else f"FAIL ({bad} mismatches)"
        stale = self.stale_overrides
        if stale:
            names = ", ".join(f"{n}()" for n in stale)
            if self.strict and not bad:
                result = f"FAIL (--strict): your C++ for {names} was written for an earlier version of the Python"
            else:
                result += f"; your C++ for {names} was written for an earlier version of the Python"
            result += f" (accept it with `qtrans override --accept {Path(self.file).name} NAME`)"
        skipped = [f.name for f in t.rejected]
        if skipped and self.ok:
            checked = len(self.functions) + (self.program is not None)
            what = ", ".join("the program (main)" if n == "(program)" else f"{n}()" for n in skipped[:6])
            what += f" and {len(skipped) - 6} more" if len(skipped) > 6 else ""
            result = ("NOTHING CHECKED" if checked == 0 else "PASS for what was translated") + \
                f"; not translated, so not checked: {what}"
        lines += ["", "  RESULT: " + result]
        return "\n".join(lines)


# --- input generation -------------------------------------------------------------------
_STRINGS = ["", "x"]  # replaced per module by the string literals it contains
_TEXT_METHODS = {"upper", "lower", "capitalize", "title", "swapcase", "casefold", "istitle", "strip", "lstrip", "rstrip", "split", "replace", "find", "index",
                 "count", "startswith", "endswith", "isdigit", "isdecimal", "isnumeric", "isalpha", "isalnum",
                 "isspace", "isupper", "islower", "isprintable", "zfill", "ljust", "rjust", "center"}
_TEXT_PROBES = ["  padded  ", "a,b,,c", "Mixed Case", "UPPER", "lower", "12", "-3.5", " 1_000.25 ", "1e3", "inf",
                "-nan", "x1", "9223372036854775808", "é", "ÉCOLE", "straße", "ΟΔΟΣ ΣΑΣ", "\u00a0nbsp\u2003",
                "١٢٣", "tab\there", "line\nbreak", "ǅ", "ﬁ"]
_STRESS = False       # profile "stress": extreme floats and longer arrays (see random_value)
_EXTREME = [math.inf, -math.inf, math.nan, 1e308, -1e308, 5e-324, -5e-324, 2.2250738585072014e-308,
            0.0, -0.0, 1e-300, 1e300, 1e-16, 1.0 + 2.0 ** -52, 0.49999999999999994, 2.5, -2.5]
_CLASSES = {}         # name -> (ClassInfo, the Python class), per module
_INIT_PARAMS = {}     # plain class -> its __init__ parameters (how random objects are built)


class _CantBuild(Exception):
    """A random object its own __post_init__ refused: draw another."""


def _is_object(v) -> bool:
    """An instance of one of the module's translated classes (or any dataclass / NamedTuple)."""
    if isinstance(v, type):
        return False
    return (dataclasses.is_dataclass(v) or (isinstance(v, tuple) and hasattr(v, "_fields"))
            or type(v).__name__ in _CLASSES)


def random_value(rng, t, n=3, shape=None):
    """A random argument of type t. Arrays use size n (Vec of n, Mat of n x n) so that the
    arguments of one call have compatible shapes, unless `shape` is given (see shape_plans).
    Strings come from the module's own string literals (plus "" and one that matches none), so
    `if stance == "Long only"` branches run."""
    if t == STR:
        return str(rng.choice(_STRINGS))
    if isinstance(t, ListType):  # a list of 0 to 6 elements (empty lists are where sum() is the int 0)
        k = int(rng.integers(0, 7)) if rng.random() < 0.9 else 0
        return [random_value(rng, t.elem) for _ in range(k)]
    if isinstance(t, TupleType):  # an element of a list of tuples
        return tuple(random_value(rng, e, n) for e in t.elts)
    if isinstance(t, DictType):  # 0 to 5 entries; keys from a small set, so lookups hit and miss
        k = int(rng.integers(0, 6)) if rng.random() < 0.9 else 0
        out = {}
        for _ in range(k):
            key = str(rng.choice(_STRINGS)) if t.key == STR else int(rng.integers(-3, 8))
            out[key] = random_value(rng, t.val)
        return out
    if isinstance(t, StructType):
        info, cls = _CLASSES[t.name]
        if t.ref:  # a Payoff may be any class that is one: a Call, a Put, ...
            options = [c for c in [info] + info.descendants() if not (c.abc and c.abstract_left())]
            if not options:
                raise _CantBuild(t.name)
            info = options[int(rng.integers(0, len(options)))]
            cls = _CLASSES[info.name][1]
            t = info.type
        for _ in range(30):
            params = [p.type for p in _INIT_PARAMS[t.name]] if info.plain else [f.type for f in info.init_fields]
            args = [np.random.default_rng(random_value(rng, RNG)) if pt == RNG else random_value(rng, pt, n)
                    for pt in params]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(all="ignore"):
                        return cls(*args)
            except Exception:  # noqa: BLE001 - rejected by its own checks
                continue
        raise _CantBuild(t.name)
    if isinstance(t, OptionalType):
        return None if rng.random() < 0.25 else random_value(rng, t.inner, n, shape)
    if t == RNG:
        return int(rng.integers(0, 2 ** 31))  # a seed: each side builds its own generator from it
    if isinstance(t, ArrayType):
        shape = shape if shape is not None else (n,) if t.rank == 1 else (n, n)
        return np.array([random_value(rng, FLOAT) for _ in range(int(np.prod(shape)))],
                        dtype=np.float64).reshape(shape)
    if isinstance(t, IntArrayType):  # small ints of both signs (indexes, counts), sometimes huge ones
        k = shape[0] if shape is not None else n
        if rng.random() < 0.1:
            return rng.integers(-(2 ** 62), 2 ** 62, size=k, dtype=np.int64)
        return rng.integers(-3, 10, size=k, dtype=np.int64)
    if t == FLOAT:
        if _STRESS and rng.random() < 0.3:
            # infinities, NaN, the largest and smallest doubles, subnormals, signed zeros, the
            # doubles next to 1 and 0.5, and halves (where rounding ties live)
            return float(_EXTREME[int(rng.integers(len(_EXTREME)))])
        u = rng.random()
        if u < 0.70:
            return float(10 ** rng.uniform(-2, 2))      # positive, spread over 4 decades
        if u < 0.85:
            return float(rng.uniform(-2, 2))            # around zero, both signs
        if u < 0.95:
            return float(rng.choice([0.0, 0.5, 1.0, 2.0, 100.0]))
        return float(rng.choice([-1.0, 1e-8, 1e8, -0.0]))
    if t == INT:
        return int(rng.integers(1, 101)) if rng.random() < 0.8 else int(rng.integers(-10, 11))
    if t == BOOL:
        return bool(rng.random() < 0.5)
    raise TypeError(t)


# --- rectangular and empty arrays ---------------------------------------------------------------------
# "1": a dimension of length 1, which NumPy stretches against any other (broadcasting)
_SLOT_CHOICES = {1: ("n", "m", "1"), 2: ("nn", "nm", "mn", "mm", "1m", "n1")}


def _rank_of(t):
    inner = t.inner if isinstance(t, OptionalType) else t
    return inner.rank if isinstance(inner, ArrayType) else None


def shape_plans(fn, sig, rng, limit=12, max_combos=243):
    """Which non-square shapes the function accepts. Square cases (every Vec of n, every Mat
    n x n) can't catch a transposed dimension, so each array parameter is also tried as length
    n, m or 1 (a Vec) and n x n, n x m, m x n, m x m, 1 x m or n x 1 (a Mat), with n = 3, m = 4;
    an assignment is kept when Python runs the function on it. Python is the judge of which
    shapes make sense; the 1s are where NumPy broadcasts.
    -> [plan], each plan a tuple with an entry per parameter ("n", "nm", ... or None)"""
    import itertools
    slots = [i for i, p in enumerate(sig.params) if _rank_of(p.type)]
    if not slots:
        return []
    combos = list(itertools.product(*[_SLOT_CHOICES[_rank_of(sig.params[i].type)] for i in slots]))
    combos = [c for c in combos if not (set(c) <= {"n", "nn"} or set(c) <= {"m", "mm"})]  # not all square
    combos = [combos[i] for i in rng.permutation(len(combos))[:max_combos]]  # a spread, not the first few
    plans = []
    for combo in combos:
        plan = [None] * len(sig.params)
        for i, c in zip(slots, combo):
            plan[i] = c
        for _ in range(2):
            try:
                args = plan_args(rng, sig, tuple(plan), 3, 4, optional_none=False)
            except _CantBuild:
                break
            r = run_python(fn, args, sig)
            if is_timeout(r):
                return plans  # the function hangs on some inputs: don't keep probing it
            if r[0] == "ok":
                plans.append(tuple(plan))
                break
        if len(plans) >= limit:
            break
    return plans


def plan_args(rng, sig, plan, n, m, optional_none=True):
    dims = {"n": n, "m": m, "1": 1}
    out = []
    for p, c in zip(sig.params, plan):
        if c is None:
            out.append(random_value(rng, p.type, n))
            continue
        shape = tuple(dims[ch] for ch in c)
        if isinstance(p.type, OptionalType):
            out.append(None if optional_none and rng.random() < 0.25 else random_value(rng, p.type.inner, n, shape))
        else:
            out.append(random_value(rng, p.type, n, shape))
    return tuple(out)


def random_args(rng, sig, plans):
    """One random call: usually square (every array of size n), and in 40% of calls, when the
    function accepts non-square shapes, one of its rectangular plans with n != m, either of
    them possibly 0 (an empty matrix of 0 x m) or 1."""
    if plans and rng.random() < 0.4:
        plan = plans[int(rng.integers(len(plans)))]
        n, m = (int(x) for x in rng.choice(7, size=2, replace=False))
        return plan_args(rng, sig, plan, n, m)
    n = 0 if rng.random() < 0.05 else int(rng.integers(1, 7))
    if _STRESS and rng.random() < 0.15:
        n = int(rng.integers(7, 41))  # longer arrays: blocked sums, loops past small sizes
    return tuple(random_value(rng, p.type, n) for p in sig.params)


def _encode(v, t):
    if t == RNG and isinstance(v, np.random.Generator):  # a generator inside an object: its exact state
        full = v.bit_generator.state
        st = full["state"]
        m = (1 << 64) - 1
        return (f"{st['state'] >> 64} {st['state'] & m} {st['inc'] >> 64} {st['inc'] & m} "
                f"{int(full['has_uint32'])} {int(full['uinteger'])}")
    if isinstance(t, StructType):
        if t.ref:  # its class first: the C++ builds an object of that class
            info, _ = _CLASSES[type(v).__name__]
            return " ".join([f"@{info.name}"] + [_encode(getattr(v, f.name), f.type) for f in info.fields])
        info, _ = _CLASSES[t.name]
        return " ".join(_encode(getattr(v, f.name), f.type) for f in info.fields)
    if t == STR:
        return "S" + str(v).encode("utf-8").hex()
    if isinstance(t, ListType):
        return " ".join([str(len(v))] + [_encode(x, t.elem) for x in v])
    if isinstance(t, TupleType):  # its elements, in order
        return " ".join(_encode(x, e) for x, e in zip(v, t.elts))
    if isinstance(t, DictType):  # its length, then key and value, in order
        return " ".join([str(len(v))] + [f"{_encode(k, t.key)} {_encode(x, t.val)}" for k, x in v.items()])
    if isinstance(t, OptionalType):
        return "N" if v is None else _encode(v, t.inner)
    if t == RNG:
        return str(int(v))
    if isinstance(t, ArrayType):
        a = np.asarray(v, dtype=np.float64)
        head = f"{a.shape[0]}" if t.rank == 1 else f"{a.shape[0]} {a.shape[1]}"
        return " ".join([head] + [float(x).hex() for x in a.ravel()])
    if isinstance(t, IntArrayType):
        a = np.asarray(v)
        return " ".join([str(a.shape[0])] + [str(int(x)) for x in a.ravel()])
    if t == FLOAT:
        return float(v).hex()
    if t == INT:
        return str(int(v))
    return "1" if v else "0"


def _decode_float(s):
    s = s.strip()
    low = s.lower()
    if low.startswith("-nan") or low.startswith("nan"):
        return math.nan
    if low in ("inf", "+inf"):
        return math.inf
    if low == "-inf":
        return -math.inf
    return float.fromhex(s)


def _flatten(t):
    return list(t.elts) if isinstance(t, TupleType) else [t]


def _changes_shown(p) -> bool:
    """Is the argument compared after the call too? One the function modifies in place, and any
    object of a class hierarchy (or list of them): behind a shared_ptr, a change made inside
    reaches the caller's object, as in Python."""
    t = p.type.elem if isinstance(p.type, ListType) else p.type
    return getattr(p, "mode", "") == "ref" or (isinstance(t, StructType) and t.ref)


def random_seed(args, sig) -> int:
    """The seed both sides give the random module's generator before calling a function that
    uses it: from the function's name and the call's ints and sizes (not its floats, so the
    nudged calls that measure conditioning draw the same numbers)."""
    parts = [sig.name.split("(")[0]]

    def walk(v):
        if isinstance(v, (bool, int, np.integer)):
            parts.append(str(int(v)))
        elif isinstance(v, (list, tuple)):
            parts.append(f"[{len(v)}")
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            parts.append(f"{{{len(v)}")
        elif isinstance(v, np.ndarray):
            parts.append(str(v.shape))
        elif isinstance(v, str):
            parts.append(v)
    for a in args:
        walk(a)
    return zlib.crc32("|".join(parts).encode("utf-8"))


def encode_call(args, sig) -> str:
    """A call's arguments for the driver (CALL <function> <this>): the random module's seed
    first for a function that uses it."""
    enc = " ".join(_encode(v, p.type) for v, p in zip(args, sig.params))
    return f"{random_seed(args, sig)} {enc}" if getattr(sig, "py_random", False) else enc


def _compared_after(sig) -> bool:
    return any(_changes_shown(p) for p in sig.params) or getattr(sig, "py_random", False)


def output_type(sig):
    """What a call produces: the return value, plus the final state of every array the
    function modifies in place (in Python the caller sees those changes too)."""
    # a generator passed in is reported as its next raw output after the call: equal only if
    # both sides consumed exactly the same number of draws
    refs = [INT if p.type == RNG else p.type for p in sig.params if _changes_shown(p)]
    if getattr(sig, "py_random", False):  # the random module's generator: its next 32 bits, gauss()'s saved value
        refs += [INT, OptionalType(FLOAT)]
    if not refs:
        return sig.ret
    returned = [] if sig.ret.cpp() == "void" else _flatten(sig.ret)
    return TupleType(tuple(returned + refs))


# --- python side ------------------------------------------------------------------------
def _py_category(exc):
    """The Python class the C++ reports for the same error: the exception's own class (the C++
    classes have Python's names and family, your own classes included), except that NumPy's
    internal subclasses (UFuncTypeError, AxisError, ...) count as the built-in class they derive
    from; LinAlgError is NumPy's public one and keeps its name."""
    cls = type(exc)
    if cls.__name__ == "LinAlgError" or not (cls.__module__ or "").startswith("numpy"):
        return cls.__name__
    for base in cls.__mro__:
        if base.__module__ == "builtins":
            return base.__name__
    return cls.__name__


def _has_complex(v):
    if isinstance(v, (list, tuple)):
        return any(_has_complex(x) for x in v)
    return isinstance(v, (complex, np.complexfloating)) or (isinstance(v, np.ndarray) and np.iscomplexobj(v))


def _py_value(v, t):
    if isinstance(t, TupleType) and not isinstance(v, tuple):
        raise TypeError(f"expected a tuple, got {type(v).__name__}")
    if isinstance(t, TupleType):
        if len(v) != len(t.elts):
            raise TypeError(f"expected a tuple of {len(t.elts)}, got {len(v)}")
        return tuple(_py_value(x, e) for x, e in zip(v, t.elts))
    if isinstance(t, DictType):
        if not isinstance(v, dict):
            raise TypeError(f"expected a dict, got {type(v).__name__}")
        if t.val == FLOAT and any(type(x) is int for x in v.values()):
            raise TypeError("a dict of floats holding an int (Python prints it as one)")
        return {_py_value(k, t.key): _py_value(x, t.val) for k, x in v.items()}
    if isinstance(t, ListType):
        if not isinstance(v, list):
            raise TypeError(f"expected a list, got {type(v).__name__}")
        if t.elem == FLOAT and any(type(x) is int for x in v):
            raise TypeError("a list of floats holding an int (Python prints it as one)")
        return [_py_value(x, t.elem) for x in v]
    if isinstance(t, StructType):
        ok = {b.__name__ for b in type(v).__mro__} if t.ref else {type(v).__name__}  # a Call is a Payoff
        if not _is_object(v) or t.name not in ok:
            raise TypeError(f"expected a {t.name}, got {type(v).__name__}")
        return copy.deepcopy(v)
    if t == STR:
        if not isinstance(v, str):
            raise TypeError(f"expected a str, got {type(v).__name__}")
        return v
    if isinstance(t, OptionalType):
        return None if v is None else _py_value(v, t.inner)
    if t == RNG:
        if isinstance(v, np.random.Generator):
            raise TypeError("give generator arguments in QTRANS_CASES as integer seeds")
        return int(v)
    if isinstance(t, TupleType):
        return tuple(_py_value(x, et) for x, et in zip(v, t.elts))
    if _has_complex(v):
        # x ** y with x < 0 and fractional y made Python's value complex; float() and a
        # float64 copy would drop the imaginary part silently (a ComplexWarning)
        raise TypeError("Python returned a complex number")
    if isinstance(t, ArrayType):
        a = np.array(v, dtype=np.float64, copy=True)
        if a.ndim != t.rank:
            raise TypeError(f"expected a {t.rank}-D array, got {a.ndim}-D")
        return a
    if isinstance(t, IntArrayType):  # an int64 array (a float one would print and divide differently)
        if not (isinstance(v, np.ndarray) and v.dtype == np.int64 and v.ndim == 1):
            raise TypeError(f"expected a 1-D int64 array, got {type(v).__name__} "
                            f"{getattr(v, 'dtype', '')}".rstrip())
        return np.array(v, copy=True)
    if t == FLOAT:
        return float(v)
    if t == INT:
        return int(v)
    return bool(v)


def _call_args(args, sig):
    """Fresh copies of the arguments for one call: arrays copied (the call may modify them),
    generators built from their seeds."""
    out = []
    for a, p in zip(args, sig.params):
        if p.type == RNG:
            out.append(np.random.default_rng(int(a)) if int(a) >= 0 else int(a))
        elif isinstance(a, np.ndarray):
            out.append(np.array(a, dtype=np.float64, copy=True))
        elif _is_object(a):
            out.append(copy.deepcopy(a))  # the call may modify it
        else:
            out.append(a)
    return out


PYTHON_CASE_SECONDS = 10.0


def is_timeout(r) -> bool:
    return r[0] == "exc" and r[1] == "Timeout"


def run_python(fn, args, sig, limit=None, printed=False):
    """Python's result for one call: in the worker process with a time limit of `limit`
    seconds (PYTHON_CASE_SECONDS) while a verification runs (see PythonWorker), else here.
    printed=True: (result, what the call printed)."""
    limit = PYTHON_CASE_SECONDS if limit is None else limit
    if _WORKER is not None:
        r = _WORKER.call(fn, args, sig, limit, mode="printed" if printed else None)
        if r is not None:
            if printed and (r[0] != "printed"):
                return r, ""  # a timeout or a crash: nothing printed to compare
            return (r[1], r[2]) if printed else r
    return _run_python_printed(fn, args, sig) if printed else _run_python(fn, args, sig)


def _probe_singular(fn, args, sig):
    """("ok", True) when a matrix the call gave np.linalg.inv / solve was singular to working
    precision (smallest singular value <= n * eps * the largest), or one it gave
    np.linalg.cholesky was not positive definite to working precision (smallest eigenvalue
    <= n * eps * the largest |eigenvalue|). LU on such a matrix meets an exactly-zero pivot,
    or Cholesky a non-positive one, or not, as rounding decides: LAPACK and Eigen, adding in
    different orders, can decide differently, and neither is wrong."""
    import numpy.linalg as la
    found = []
    real = {name: getattr(la, name) for name in ("inv", "solve", "cholesky")}

    def edge(a, definite):
        m = np.asarray(a, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.size == 0 or not np.isfinite(m).all():
            return False
        tol = m.shape[0] * np.finfo(np.float64).eps
        if definite:
            w = np.linalg.eigvalsh(m)
            return bool(w[0] <= tol * max(np.max(np.abs(w)), np.finfo(np.float64).tiny))
        sv = np.linalg.svd(m, compute_uv=False)
        return bool(sv[-1] <= tol * sv[0])

    def wrap(name):
        def probe(a, *rest, **kw):
            try:
                found.append(edge(a, name == "cholesky"))
            except Exception:  # noqa: BLE001 - the probe never changes what the call does
                pass
            return real[name](a, *rest, **kw)
        return probe

    for name in real:
        setattr(la, name, wrap(name))
    try:
        _run_python(fn, args, sig)
    finally:
        for name, f in real.items():
            setattr(la, name, f)
    return ("ok", any(found))


class _PowProbe(ast.NodeTransformer):
    """a ** b -> __qtrans_pow__(a, b), and x **= b -> x = __qtrans_pow__(x, b) for a name x."""

    def visit_BinOp(self, n):
        self.generic_visit(n)
        if isinstance(n.op, ast.Pow):
            return ast.copy_location(ast.Call(ast.Name("__qtrans_pow__", ast.Load()), [n.left, n.right], []), n)
        return n

    def visit_AugAssign(self, n):
        self.generic_visit(n)
        if isinstance(n.op, ast.Pow) and isinstance(n.target, ast.Name):
            call = ast.Call(ast.Name("__qtrans_pow__", ast.Load()), [ast.Name(n.target.id, ast.Load()), n.value], [])
            return ast.copy_location(ast.Assign([n.target], call), n)
        return n


_POW_MODULES: dict = {}


def _probe_complex(fn, args, sig):
    """("ok", True) when a ** b anywhere in the call (fn's module, its other functions included)
    gave a complex number: a Python float raised to a non-integer power with a negative base.
    From there Python computes with complex numbers and C++, like np.float64, with NaN (the
    documented complex-power difference), even when no complex value reaches the result
    (`v > x ** 0.5` compares by the real part)."""
    try:
        path = inspect.getsourcefile(fn)
        src = Path(path).read_text(encoding="utf-8-sig")
    except (TypeError, OSError):
        return ("ok", False)
    went = []

    def pow_(a, b):
        r = a ** b
        if isinstance(r, (complex, np.complexfloating)) or (isinstance(r, np.ndarray) and np.iscomplexobj(r)):
            went.append(True)
        return r
    key = (path, hash(src))  # the shrinker rewrites one file over and over
    if key not in _POW_MODULES:
        tree = ast.fix_missing_locations(_PowProbe().visit(ast.parse(src, path)))
        ns = {"__name__": f"_qtrans_pow_probe_{uuid.uuid4().hex[:8]}", "__file__": path}
        ns["__qtrans_pow__"] = lambda a, b: ns["__qtrans_pow_impl__"](a, b)
        try:
            exec(compile(tree, path, "exec"), ns)  # noqa: S102 - the user's own module, as load_module does
        except Exception:  # noqa: BLE001 - the probe never changes a verdict when it can't run
            return ("ok", False)
        _POW_MODULES[key] = ns
    ns = _POW_MODULES[key]
    ns["__qtrans_pow_impl__"] = pow_
    try:
        probe = _resolve(_Namespace(ns), fn.__qualname__)
    except AttributeError:
        return ("ok", False)
    _run_python(probe, args, sig)
    return ("ok", bool(went))


class _Namespace:
    def __init__(self, d):
        self.__dict__.update(d)


def python_went_complex(fn, args, sig) -> bool:
    """Did a ** b go complex during this call? See _probe_complex (run in the worker, with the
    time limit, when there is one)."""
    r = _WORKER.call(fn, args, sig, PYTHON_CASE_SECONDS, mode="complex") if _WORKER is not None else None
    if r is None:
        r = _probe_complex(fn, args, sig)
    return r[0] == "ok" and r[1] is True


def _singular_verdict(case) -> bool:
    """The side that raised said "singular" or "not positive definite" (the C++ driver reports
    shape errors as ShapeError, so its ValueError is a LinAlgError)."""
    if case.py[0] == "exc":
        return "ingular" in case.py[2] or "positive definite" in case.py[2]
    return case.cpp[0] == "exc" and case.cpp[1] in ("LinAlgError", "ValueError")


def python_near_singular(fn, case, sig) -> bool:
    """Was a matrix this call factored singular (or, for Cholesky, semi-definite) to working
    precision? See _probe_singular; run in the worker, with the time limit, when there is one."""
    r = _WORKER.call(fn, case.args, sig, PYTHON_CASE_SECONDS, mode="singular") if _WORKER is not None else None
    if r is None:
        r = _probe_singular(fn, case.args, sig)
    return r[0] == "ok" and r[1] is True


def _run_python(fn, args, sig):
    return _run_python_printed(fn, args, sig)[0]


def _run_python_printed(fn, args, sig):
    """(result, what the call printed)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = _call_python(fn, args, sig)
    return r, buf.getvalue()


def _call_python(fn, args, sig):
    rnd = getattr(sig, "py_random", False)
    if rnd:
        random.seed(random_seed(args, sig))
    args = _call_args(args, sig)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                v = fn(*args)
    except Exception as e:  # noqa: BLE001 - every exception is a test outcome
        return ("exc", _py_category(e), str(e))
    if sig.ret == VOID and not _compared_after(sig):
        return ("ok", None)  # a function that only prints
    try:
        refs = [rng_fingerprint(args[i]) if p.type == RNG else args[i]
                for i, p in enumerate(sig.params) if _changes_shown(p)]
        if rnd:
            after = random.Random()
            after.setstate(random.getstate())
            refs += [after.getrandbits(32), random._inst.gauss_next]
        if refs:
            flat = list(v) if isinstance(sig.ret, TupleType) else [] if sig.ret.cpp() == "void" else [v]
            return ("ok", _py_value(tuple(flat + refs), output_type(sig)))
        return ("ok", _py_value(v, sig.ret))
    except (TypeError, ValueError) as e:
        if "complex" in str(e) or isinstance(v, complex):
            return ("exc", "TypeError", "Python returned a complex number")
        return ("bad", f"{v!r} ({e})")


def load_module(path, name=None):
    """Import a file under a fresh name (so two files of one name don't collide), registered
    in sys.modules so its objects can be pickled to the Python worker (see PythonWorker)."""
    path = Path(path).resolve()
    name = name or f"_qtrans_verify_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(path.parent))
    try:
        # a program's top-level code (what becomes int main()) is left out: importing the file
        # to call its functions must not run the program (it is run on its own, see run_program)
        tree = library_tree(ast.parse(path.read_text(encoding="utf-8-sig"), str(path)))
        exec(compile(tree, str(path), "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path.remove(str(path.parent))
    return mod


# --- running the Python side in a worker process ------------------------------------------------
def _worker_client(address, authkey_hex):
    """Entry point of the worker process (started with `python -c`, so the caller's own script
    is never re-imported, as multiprocessing's spawn would): connect back, get the setup."""
    from multiprocessing.connection import Client
    conn = Client(address, authkey=bytes.fromhex(authkey_hex))
    specs, path, class_names = conn.recv()
    _worker_main(conn, specs, path, class_names)


def _worker_main(conn, specs, path, class_names):
    """The worker: imports the modules as the verifier did, then runs one call per message."""
    sys.path[:0] = path
    mods = {}
    for key, (kind, where, name) in specs.items():
        if kind == "file":
            mods[key] = load_module(where, name)
        else:
            sys.path.insert(0, where)
            mods[key] = importlib.import_module(name)
    global _CLASSES
    _CLASSES = {n: (None, None) for n in class_names}  # enough for _is_object
    while True:
        msg = conn.recv()
        if msg is None:
            return
        key, fname, args, sig, mode = msg
        fn = _resolve(mods[key], fname)
        if mode == "printed":
            r = ("printed",) + _run_python_printed(fn, args, sig)
        else:
            with contextlib.redirect_stdout(io.StringIO()):  # a function that prints: not here
                r = (_probe_singular(fn, args, sig) if mode == "singular" else
                     _probe_complex(fn, args, sig) if mode == "complex" else _run_python(fn, args, sig))
        try:
            conn.send(r)
        except Exception as e:  # noqa: BLE001 - a result that can't be pickled
            conn.send(("bad", f"result could not be sent back from the worker: {e}"))


def _resolve(mod, dotted):
    obj = mod
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class PythonWorker:
    """Runs the Python side of verification in a child process, each call with a time limit.
    A call can hang (NumPy's SVD on some matrices with inf in them never returns, sometimes
    holding the interpreter lock), and a hang inside compiled code can't be interrupted from
    within the process: the worker is then killed and restarted, and the call reported as a
    timeout. Calls whose arguments can't be pickled run in this process instead."""

    def __init__(self, specs, class_names):
        self.specs, self.class_names = specs, list(class_names)
        self.funcs = {}  # id(fn) -> (module key, name)
        self.proc = self.conn = None
        self.restarts = 0

    def register(self, key, mod):
        for name in dir(mod):
            obj = getattr(mod, name)
            if callable(obj):
                self.funcs[id(obj)] = (key, name)

    def _start(self):
        import secrets
        from multiprocessing.connection import Listener
        key = secrets.token_bytes(16)
        with Listener(authkey=key) as listener:  # a local socket / named pipe, authenticated
            here = str(Path(__file__).resolve().parent.parent)
            code = (f"import sys; sys.path.insert(0, {here!r}); from qtrans.verify import _worker_client; "
                    f"_worker_client({listener.address!r}, {key.hex()!r})")
            self.proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL)
            self.conn = listener.accept()
        self.conn.send((self.specs, list(sys.path), self.class_names))

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def call(self, fn, args, sig, limit, mode=None):
        where = self.funcs.get(id(fn))
        if where is None:
            return None
        if not self._alive():
            self._start()
        try:
            self.conn.send((where[0], where[1], args, sig, mode))
        except Exception:  # noqa: BLE001 - arguments that can't be pickled: run here
            return None
        if self.conn.poll(limit + (30.0 if self.restarts == 0 and not getattr(self, "_warm", False) else 0.0)):
            self._warm = True
            try:
                return self.conn.recv()
            except EOFError:  # the worker died (a crash in compiled code)
                self.proc = None
                return ("exc", "Crash", "Python crashed in compiled code")
        self.stop(kill=True)
        self.restarts += 1
        self._warm = False
        return ("exc", "Timeout", f"Python did not finish within {limit:g} s")

    def stop(self, kill=False):
        if self.proc is None:
            return
        try:
            if kill:
                self.proc.kill()
            else:
                self.conn.send(None)
            self.proc.wait(5)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None


_WORKER = None  # the PythonWorker of the verification in progress


# --- C++ side ---------------------------------------------------------------------------------
_PARSE = {FLOAT: "F", INT: "I", BOOL: "B", VEC: "V", MAT: "M", STR: "Str", IVEC: "IV"}
_CTYPE = {FLOAT: "double", INT: "long long", BOOL: "bool"}

_DRIVER_PRELUDE = r"""// Test driver generated by qtrans verify. Reads one call per line:
//   CALL <function> <args...>    -> OK <results...> | EXC <category>
//   BENCH <function> <args...>   -> BENCH <nanoseconds per call> <checksum> <calls timed>
// Numbers travel as hex floats (exact); arrays as their shape followed by the values.
#include "@@HEADER@@"
#include "qtrans_runtime.hpp"  // qt::type_error, even when the module itself needs nothing from it
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

struct Reader {
    std::vector<std::string> tokens;
    std::size_t pos = 0;
    const std::string& next() {
        if (pos >= tokens.size()) throw std::runtime_error("driver: missing input");
        return tokens[pos++];
    }
};
static double F(Reader& r) { return std::strtod(r.next().c_str(), nullptr); }  // reads hex floats exactly
static long long I(Reader& r) { return std::strtoll(r.next().c_str(), nullptr, 10); }
static bool B(Reader& r) { return r.next() == "1"; }
static std::uint64_t U(Reader& r) { return std::strtoull(r.next().c_str(), nullptr, 10); }
// strings travel as S + hex of their UTF-8 bytes, so spaces and any character survive
static std::string Str(Reader& r) {
    const std::string& t = r.next();
    std::string s;
    for (std::size_t i = 1; i + 1 < t.size(); i += 2) s += static_cast<char>(std::stoi(t.substr(i, 2), nullptr, 16));
    return s;
}
static void out(const std::string& s) {
    std::printf(" S");
    for (unsigned char c : s) std::printf("%02x", c);
}
static double num(const std::string& s) { return static_cast<double>(s.size()); }
// a list arrives as its length, then its elements
template <class T, class Fn>
static std::vector<T> qtrans_list(Reader& r, Fn f) {
    const long long n = std::strtoll(r.next().c_str(), nullptr, 10);
    std::vector<T> v;
    for (long long i = 0; i < n; ++i) v.push_back(f(r));
    return v;
}
#ifdef QTRANS_LIST_HPP
// a dict arrives as its length, then each key and value
template <class K, class V, class FK, class FV>
static qt::py::dict<K, V> qtrans_dict(Reader& r, FK fk, FV fv) {
    const long long n = std::strtoll(r.next().c_str(), nullptr, 10);
    qt::py::dict<K, V> d;
    for (long long i = 0; i < n; ++i) {
        K k = fk(r);
        d[k] = fv(r);
    }
    return d;
}
#endif
// an argument that may be None arrives as N
template <class T, class Fn>
static std::optional<T> qtrans_optional(Reader& r, Fn f) {
    if (r.pos < r.tokens.size() && r.tokens[r.pos] == "N") { ++r.pos; return std::nullopt; }
    return f(r);
}

static void out(double x) { std::printf(" %a", x); }  // hex float: exact
static void out(long long x) { std::printf(" %lld", x); }
static void out(bool x) { std::printf(" %d", x ? 1 : 0); }
static double num(double x) { return x; }
static double num(long long x) { return static_cast<double>(x); }
static double num(bool x) { return x ? 1.0 : 0.0; }
@@ARRAY_IO@@
// declared first: a list's elements may be tuples or optionals, written by the overloads below
template <class... T>
static void out(const std::tuple<T...>& t);
template <class T>
static void out(const std::optional<T>& x);
template <class... T>
static double num(const std::tuple<T...>& t);
template <class T>
static double num(const std::optional<T>& x);
template <class T>
static void out(const std::vector<T>& v) {
    std::printf(" %lld", static_cast<long long>(v.size()));
    for (const T& x : v) out(x);
}
template <class T>
static double num(const std::vector<T>& v) { return v.empty() ? 0.0 : num(v[0]); }
#ifdef QTRANS_LIST_HPP
template <class K, class V>
static void out(const qt::py::dict<K, V>& d) {
    std::printf(" %lld", d.size());
    for (const auto& [k, v] : d) { out(k); out(v); }
}
template <class K, class V>
static double num(const qt::py::dict<K, V>& d) { return d.empty() ? 0.0 : num(d.begin()->second); }
#endif
template <class T>
static void out(const std::optional<T>& x) {
    if (x) out(*x); else std::printf(" N");
}
template <class T>
static double num(const std::optional<T>& x) { return x ? num(*x) : 0.0; }
template <class... T>
static void out(const std::tuple<T...>& t) {
    std::apply([](const auto&... v) { (out(v), ...); }, t);
}
template <class... T>
static double num(const std::tuple<T...>& t) {
    // every element, so none of them can be left uncomputed when timed
    return std::apply([](const auto&... v) { return (0.0 + ... + num(v)); }, t);
}

// Make the compiler assume x may have changed, so a call on unchanged arguments can't be
// computed once and reused across benchmark repetitions (the usual DoNotOptimize technique).
template <class T>
static inline void qtrans_opaque(T& x) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(&x) : "memory");
#else
    static T* volatile sink;
    sink = &x;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

template <class Fn>
static void bench(Fn&& f) {
    // Repeat until the timing covers at least 0.1 s, so clock resolution doesn't matter.
    double sink = num(f());  // one warm-up call, not timed
    long long reps = 1;
    double seconds = 0.0;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        for (long long k = 0; k < reps; ++k) sink += num(f());
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (seconds > 0.1 || reps > (1LL << 40)) break;
        reps *= 4;
    }
    std::printf("BENCH %.6g %a %lld\n", seconds * 1e9 / static_cast<double>(reps), sink, reps);
}

int main() {
    // what the functions print goes here, and is reported after each call (PRINTED)
    std::ostringstream printed;
    struct Restore {  // std::cout must have its own buffer back before `printed` goes away
        std::streambuf* saved;
        ~Restore() { std::cout.rdbuf(saved); }
    } restore{std::cout.rdbuf(printed.rdbuf())};
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string mode, fn, tok;
        in >> mode >> fn;
        Reader r;
        while (in >> tok) r.tokens.push_back(tok);
#ifdef QTRANS_HAVE_AUDIT
        qt::audit::reset();
        qt::audit::state().enabled = mode == "CALL";  // never while timing
#endif
        try {
@@BRANCHES@@
            { std::printf("EXC UnknownFunction\n"); }
        }
        catch (const qt::int_overflow&) { std::printf("EXC IntOverflow\n"); }@@ARRAY_CATCH@@
        catch (const qt::Exception& e) { std::printf("EXC %s\n", e.type_name()); }  // Python's class, by name
        // a standard exception (from your own C++ for a function): the Python class it stands for
        catch (const std::invalid_argument&) { std::printf("EXC ValueError\n"); }
        catch (const std::domain_error&) { std::printf("EXC ArithmeticError\n"); }
        catch (const std::out_of_range&) { std::printf("EXC IndexError\n"); }
        catch (const std::logic_error&) { std::printf("EXC AssertionError\n"); }
        catch (const std::overflow_error&) { std::printf("EXC OverflowError\n"); }
        catch (const std::runtime_error&) { std::printf("EXC RuntimeError\n"); }
        catch (const std::exception&) { std::printf("EXC Exception\n"); }
#ifdef QTRANS_HAVE_AUDIT
        // the linear-algebra audit of this call (see qtrans_arrays.hpp): operations audited,
        // the worst residual / bound, and the operation it came from
        if (mode == "CALL")
            std::printf("AUDIT %ld %a %s\n", qt::audit::state().count, qt::audit::state().worst,
                        qt::audit::state().op[0] ? qt::audit::state().op : "-");
#endif
        if (mode == "CALL" && !printed.str().empty()) {
            std::printf("PRINTED");
            out(printed.str());
            std::printf("\n");
        }
        printed.str("");
        printed.clear();
    }
    return 0;
}
"""

_ARRAY_IO = r"""static qt::Vec V(Reader& r) {
    const long long n = I(r);
    qt::Vec v(n);
    for (long long i = 0; i < n; ++i) v(i) = F(r);
    return v;
}
static qt::Mat M(Reader& r) {
    const long long rows = I(r), cols = I(r);
    qt::Mat m(rows, cols);
    for (long long i = 0; i < rows; ++i)
        for (long long j = 0; j < cols; ++j) m(i, j) = F(r);
    return m;
}
static void out(const qt::Vec& v) {
    std::printf(" %lld", static_cast<long long>(v.size()));
    for (qt::Index i = 0; i < v.size(); ++i) out(v(i));
}
static void out(const qt::Mat& m) {
    std::printf(" %lld %lld", static_cast<long long>(m.rows()), static_cast<long long>(m.cols()));
    for (qt::Index i = 0; i < m.rows(); ++i)
        for (qt::Index j = 0; j < m.cols(); ++j) out(m(i, j));
}
static double num(const qt::Vec& v) { return v.size() ? v(0) : 0.0; }
static qt::IVec IV(Reader& r) {
    const long long n = I(r);
    qt::IVec v(n);
    for (long long i = 0; i < n; ++i) v(i) = I(r);
    return v;
}
static void out(const qt::IVec& v) {
    std::printf(" %lld", static_cast<long long>(v.size()));
    for (qt::Index i = 0; i < v.size(); ++i) out(static_cast<long long>(v(i)));
}
static double num(const qt::IVec& v) { return v.size() ? static_cast<double>(v(0)) : 0.0; }
static double num(const qt::Mat& m) { return m.size() ? m(0, 0) : 0.0; }"""


_RNG_IO = r"""static void out(qt::Rng& g) {  // how far the generator is: its next raw output, and the half
    const bool has = g.has_uint32();                  // word integers() keeps (see rng_fingerprint)
    const std::uint64_t half = g.uinteger();
    std::uint64_t v = g.next_uint64();
    if (has) v ^= half * 0x9E3779B97F4A7C15ULL + 1;
    std::printf(" %llu", static_cast<unsigned long long>(v));
}"""


def rng_fingerprint(g) -> int:
    """How far a generator is, as one number: its next raw output, mixed with the half word
    integers() keeps for its next 32-bit draw (the driver's out(qt::Rng&) computes the same)."""
    st = g.bit_generator.state
    v = int(g.bit_generator.random_raw())
    if st["has_uint32"]:
        v ^= (int(st["uinteger"]) * 0x9E3779B97F4A7C15 + 1) & ((1 << 64) - 1)
    return v


def driver_source(result, header_name=None, namespace=None, targets=None) -> str:
    """The test driver: one program for a module, or for a whole project (result a
    ProjectResult: every header included, the functions of `targets` callable as
    "module:function", every class of the project readable)."""
    if isinstance(result, TranslationResult):
        modules = {result.module: (result, namespace or result.namespace)}
        headers = [header_name or f"{result.module}.hpp"]
        classes = [(info, namespace or result.namespace) for info in (result.classes or {}).values()]
        single = True
    else:
        modules = {m: (r, r.namespace) for m, r in result.modules.items() if targets is None or m in targets}
        headers = [f"{m}.hpp" for m in result.modules]
        classes = [(info, result.modules[m].namespace) for _, (info, m) in result.classes().items()]
        single = False
    branches = []
    for module, (res, namespace) in modules.items():
        for name, sig in res.signatures.items():
            if sig.owner:
                continue  # methods are exercised through the functions that call them
            if sig.ret.cpp() == "void" and not _compared_after(sig) and name not in res.prints:
                continue
            branches.append(_branch(name if single else f"{module}:{name}", sig, namespace))
    uses_arrays = any(r.uses_arrays for r, _ in modules.values()) if single else result.uses_arrays
    uses_random = any(r.uses_random for r, _ in modules.values()) if single else result.uses_random
    io = (_ARRAY_IO if uses_arrays else "") + ("\n" + _RNG_IO if uses_random else "")
    io += _struct_io(classes)
    return (_DRIVER_PRELUDE.replace('#include "@@HEADER@@"', "\n".join(f'#include "{h}"' for h in headers))
            .replace("@@ARRAY_IO@@", io)
            .replace("@@ARRAY_CATCH@@", '\n        catch (const qt::shape_error&) { std::printf("EXC ShapeError\\n"); }'
                     if uses_arrays else "")
            .replace("@@BRANCHES@@", "\n".join(branches)))


def _branch(key, sig, namespace) -> str:
    # read arguments into named locals first: C++ doesn't fix the order in which a
    # call's arguments are evaluated, and the reader consumes tokens in sequence
    def read(i, p, volatile):
        if p.type == RNG:
            return f"qt::Rng p{i}(I(r));"
        if isinstance(p.type, (OptionalType, StructType, ListType, DictType)):
            return f"auto p{i} = {_reader(p.type, namespace)};"
        if volatile and p.type in _CTYPE:
            return f"volatile {_CTYPE[p.type]} p{i} = {_PARSE[p.type]}(r);"
        return f"auto p{i} = {_PARSE[p.type]}(r);"
    reads = " ".join(read(i, p, False) for i, p in enumerate(sig.params))
    vreads = " ".join(read(i, p, True) for i, p in enumerate(sig.params))
    args = ", ".join(f"p{i}" for i in range(len(sig.params)))
    opaque = "".join(f"qtrans_opaque(p{i}); " for i, p in enumerate(sig.params) if p.type not in _CTYPE)
    refs = "".join(f" out(p{i});" for i, p in enumerate(sig.params) if _changes_shown(p))
    if getattr(sig, "py_random", False):  # seeded first, compared after
        reads = "qt::py::random_module().seed(I(r)); " + reads
        vreads = "qt::py::random_module().seed(I(r)); " + vreads
        refs += (" out(static_cast<long long>(qt::py::random_module().peek32()));"
                 " out(qt::py::random_module().gauss_next());")
    call = f"{namespace}::{sig.cpp}({args})"
    if sig.ret.cpp() == "void":
        do_call = f'{call}; std::printf("OK");{refs} std::printf("\\n");'
        do_bench = f"bench([&] {{ {opaque}{call}; return 0.0; }});"
    else:
        do_call = f'auto res = {call}; std::printf("OK"); out(res);{refs} std::printf("\\n");'
        do_bench = f"bench([&] {{ {opaque}return {call}; }});"
    return (f'            if (fn == "{key}") {{\n'
            f'                if (mode == "CALL") {{ {reads} {do_call} }}\n'
            f'                else {{ {vreads} {do_bench} }}\n'
            f'            }} else')


def _reader(t, ns, r="r") -> str:
    """C++ that reads one value of type t from Reader r."""
    if t == RNG:  # inside an object: the generator's exact state (read in order: see RngState)
        return f"RngState({r})"
    if isinstance(t, StructType):
        return f"read_{t.name}({r})"  # class names are unique across a project
    if isinstance(t, OptionalType):
        inner = t.inner
        return (f"qtrans_optional<{inner.cpp()}>({r}, [](Reader& rr) {{ return {_reader(inner, ns, 'rr')}; }})")
    if isinstance(t, ListType):
        et = f"decltype(read_{t.elem.name}(std::declval<Reader&>()))" if isinstance(t.elem, StructType) else t.elem.cpp()
        return f"qtrans_list<{et}>({r}, [](Reader& rr) {{ return {_reader(t.elem, ns, 'rr')}; }})"
    if isinstance(t, TupleType):  # a braced list: its elements are read in order (a call's arguments aren't)
        return f"{t.cpp()}{{{', '.join(_reader(e, ns, r) for e in t.elts)}}}"
    if isinstance(t, DictType):
        return (f"qtrans_dict<{t.key.cpp()}, {t.val.cpp()}>({r}, [](Reader& rr) {{ return {_reader(t.key, ns, 'rr')}; }}, "
                f"[](Reader& rr) {{ return {_reader(t.val, ns, 'rr')}; }})")
    return f"{_PARSE[t]}({r})"


def _struct_io(classes) -> str:
    """read_X / out / num for every class (classes: [(ClassInfo, its namespace)]): the fields in
    declaration order, the object rebuilt through its constructor (so __post_init__ runs, as when
    Python built it), then any init=False fields set to Python's values."""
    if not classes:
        return ""
    lines = ["", "// objects: fields in declaration order"]
    if any(f.type == RNG for info, _ in classes for f in info.fields):
        # four separate statements: C++ doesn't fix the order of a call's arguments, so
        # from_state(U(r), U(r), ...) could read them in any order (GCC goes right to left)
        lines += ["static qt::Rng RngState(Reader& r) {",
                  "    const std::uint64_t sh = U(r);", "    const std::uint64_t sl = U(r);",
                  "    const std::uint64_t ih = U(r);", "    const std::uint64_t il = U(r);",
                  "    const std::uint64_t has = U(r);", "    const std::uint64_t half = U(r);",
                  "    qt::Rng g = qt::Rng::from_state(sh, sl, ih, il);",
                  "    g.set_uint32(has != 0, static_cast<std::uint32_t>(half));", "    return g;", "}"]
    for info, ns in classes:
        c = f"{ns}::{info.cpp}"
        if info.ref:
            lines.append(f"static std::shared_ptr<{c}> read_{info.name}(Reader& r);")
            if not (info.abc and info.abstract_left()):
                lines.append(f"static std::shared_ptr<{c}> read_{info.name}_exactly(Reader& r);")
            lines.append(f"static void out(const std::shared_ptr<{c}>& o);")
        else:
            lines.append(f"static {c} read_{info.name}(Reader& r);")
    for info, ns in classes:
        c = f"{ns}::{info.cpp}"
        if info.ref:
            lines += _hierarchy_io(info, ns)
            continue
        body = [f"static {c} read_{info.name}(Reader& r) {{"]
        for k, f in enumerate(info.fields):
            body.append(f"    auto f{k} = {_reader(f.type, ns)};")
        if info.plain:  # Python already ran __init__: set the fields it produced
            body.append(f"    {c} o{{}};")
            for k, f in enumerate(info.fields):
                body.append(f"    o.{f.cpp} = f{k};")
        else:
            init_args = ", ".join(f"f{k}" for k, f in enumerate(info.fields) if f.init)
            body.append(f"    {c} o({init_args});" if init_args else f"    {c} o{{}};")
            for k, f in enumerate(info.fields):
                if not f.init:
                    body.append(f"    o.{f.cpp} = f{k};")
        body += ["    return o;", "}"]
        lines += body
        outs = " ".join(f"{{ qt::Rng g = o.{f.cpp}; out(g); }}" if f.type == RNG else f"out(o.{f.cpp});"
                        for f in info.fields)
        lines.append(f"static void out(const {c}& o) {{ {outs} }}")
        every = " + ".join(f"num(o.{f.cpp})" for f in info.fields if f.type != RNG) or "0.0"
        lines.append(f"static double num(const {c}& o) {{ return {every}; }}")  # every field: none left uncomputed
    return "\n".join(lines) + "\n"


def _hierarchy_io(info, ns) -> list:
    """read_X / out / num for a class of a hierarchy: the object's class travels first (@Call),
    then that class's fields; reading builds an object of exactly that class."""
    c = f"{ns}::{info.cpp}"
    concrete = [k for k in [info] + info.descendants() if not (k.abc and k.abstract_left())]
    lines = []
    if info in concrete:
        body = [f"static std::shared_ptr<{c}> read_{info.name}_exactly(Reader& r) {{"]
        for k, f in enumerate(info.fields):
            body.append(f"    auto f{k} = {_reader(f.type, ns)};")
        if info.plain:  # Python already ran __init__: set the fields it produced
            body.append(f"    auto o = std::make_shared<{c}>();")
            set_ = list(enumerate(info.fields))
        else:
            args = ", ".join(f"f{k}" for k, f in enumerate(info.fields) if f.init)
            body.append(f"    auto o = std::make_shared<{c}>({args});")
            set_ = [(k, f) for k, f in enumerate(info.fields) if not f.init]
        body += [f"    o->{f.cpp} = f{k};" for k, f in set_]
        body += ["    return o;", "}"]
        lines += body
    lines.append(f"static std::shared_ptr<{c}> read_{info.name}(Reader& r) {{")
    lines.append("    const std::string tag = r.next();")
    for k in concrete:
        lines.append(f'    if (tag == "@{k.name}") return read_{k.name}_exactly(r);')
    lines += ['    throw std::runtime_error("driver: unknown class " + tag);', "}"]
    lines.append(f"static void out(const std::shared_ptr<{c}>& o) {{")
    for k in concrete:
        kc = f"{ns}::{k.cpp}"
        outs = " ".join(f"{{ qt::Rng g = q.{f.cpp}; out(g); }}" if f.type == RNG else f"out(q.{f.cpp});"
                        for f in k.fields)
        lines.append(f"    if (typeid(*o) == typeid({kc})) {{ const auto& q = static_cast<const {kc}&>(*o); "
                     f'std::printf(" @{k.name}"); {outs} return; }}')
    lines += ['    throw std::runtime_error("driver: an object of a class it doesn\'t know");', "}"]
    every = " + ".join(f"num(o->{f.cpp})" for f in info.fields if f.type != RNG) or "0.0"
    lines.append(f"static double num(const std::shared_ptr<{c}>& o) {{ return {every}; }}")
    return lines


class _Toks:
    def __init__(self, parts):
        self.parts, self.i = parts, 0

    def peek(self):
        return self.parts[self.i] if self.i < len(self.parts) else None

    def __next__(self):
        self.i += 1
        return self.parts[self.i - 1]


def _parse_audit(line):
    """AUDIT <operations> <worst ratio, hex float> <operation>: (count, worst, operation)."""
    parts = line.split(" ", 3)
    try:
        return (int(parts[1]), float.fromhex(parts[2]), parts[3] if len(parts) > 3 else "-")
    except (IndexError, ValueError):
        return None


def _parse_cpp_line(line, out_t):
    parts = line.split()
    if not parts:
        return ("exc", "NoOutput")
    if parts[0] == "EXC":
        return ("exc", parts[1] if len(parts) > 1 else "Exception")
    if out_t == VOID:
        return ("ok", None)
    toks = _Toks(parts[1:])
    values = [_parse_value(t, toks) for t in _flatten(out_t)]
    return ("ok", tuple(values) if isinstance(out_t, TupleType) else values[0])


def _parse_value(t, toks):
    if isinstance(t, TupleType):
        return tuple(_parse_value(e, toks) for e in t.elts)
    if isinstance(t, DictType):
        n = int(next(toks))
        out = {}
        for _ in range(n):
            k = _parse_value(t.key, toks)
            out[k] = _parse_value(t.val, toks)
        return out
    if isinstance(t, ListType):
        n = int(next(toks))
        return [_parse_value(t.elem, toks) for _ in range(n)]
    if isinstance(t, StructType):
        if t.ref:
            info, _ = _CLASSES[next(toks)[1:]]
            return {"__class__": info.name, **{f.name: _parse_value(f.type, toks) for f in info.fields}}
        info, _ = _CLASSES[t.name]
        return {f.name: _parse_value(f.type, toks) for f in info.fields}
    if isinstance(t, OptionalType):
        if toks.peek() == "N":
            next(toks)
            return None
        t = t.inner
    if t == STR:
        return bytes.fromhex(next(toks)[1:]).decode("utf-8", errors="replace")
    if isinstance(t, ArrayType):
        shape = (int(next(toks)),) if t.rank == 1 else (int(next(toks)), int(next(toks)))
        n = int(np.prod(shape))
        return np.array([_decode_float(next(toks)) for _ in range(n)], dtype=np.float64).reshape(shape)
    if isinstance(t, IntArrayType):
        n = int(next(toks))
        return np.array([int(next(toks)) for _ in range(n)], dtype=np.int64)
    if t == FLOAT:
        return _decode_float(next(toks))
    if t in (INT, RNG):
        return int(next(toks))  # Python ints: also holds unsigned 64-bit raw outputs
    return next(toks) == "1"


# --- what a program prints ------------------------------------------------------------------------
_NUMBER = re.compile(r"[-+]?(?:\d[\d_,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?|[-+]?(?:inf|nan)\b")


def _number(tok):
    try:
        return float(tok.replace(",", "").replace("_", "").rstrip("%"))
    except ValueError:
        return None


def printed_numbers_differ(py: str, cpp: str, rtol: float, atol: float = 1e-13):
    """Where two printed texts differ, are the differences only in numbers that agree to rounding?
    -> (None if they do; else the first differing line), and how many numbers differed.

    A computation that agrees to the last bit or two (np.linalg.solve and Eigen add in different
    orders) prints a shortest repr that differs in its last digits: those numbers must agree
    within rtol (or atol, near zero), as verify's other checks do. Everything that isn't a number
    must be identical."""
    a_lines, b_lines = py.splitlines(), cpp.splitlines()
    if len(a_lines) != len(b_lines):
        k = next((i for i, (x, y) in enumerate(zip(a_lines, b_lines)) if x != y), min(len(a_lines), len(b_lines)))
        return k + 1, 0
    differing = 0
    for k, (x, y) in enumerate(zip(a_lines, b_lines)):
        if x == y:
            continue
        xs, ys = _NUMBER.split(x), _NUMBER.split(y)
        xn, yn = _NUMBER.findall(x), _NUMBER.findall(y)
        if xs != ys or len(xn) != len(yn):
            # NumPy pads array elements to one width: a different number of digits shifts the
            # spaces; compare without the padding
            if [t.strip() for t in xs] != [t.strip() for t in ys] or len(xn) != len(yn):
                return k + 1, differing
        for p, c in zip(xn, yn):
            if p == c:
                continue
            pv, cv = _number(p), _number(c)
            if pv is None or cv is None:
                return k + 1, differing
            if math.isnan(pv) and math.isnan(cv):
                continue
            if pv == cv:
                continue
            if not (math.isfinite(pv) and math.isfinite(cv)) or abs(pv - cv) > max(rtol * max(abs(pv), abs(cv)), atol):
                return k + 1, differing
            differing += 1
    return None, differing


def printed_agrees(py: str, cpp: str, rtol: float) -> bool:
    return printed_numbers_differ(py, cpp, rtol)[0] is None


@dataclass
class ProgramCheck:
    """`python file.py` against the compiled program: what each printed, and its exit status."""
    py_out: str = ""
    cpp_out: str = ""
    py_code: int = 0
    cpp_code: int = 0
    py_err: str = ""
    cpp_err: str = ""
    seconds_py: float = 0.0
    seconds_cpp: float = 0.0
    line: int | None = None    # the first line of output that differs
    rounding: int = 0          # numbers that differ only by rounding
    error: str = ""            # it couldn't be run (a timeout)

    @property
    def ok(self):
        return not self.error and self.line is None and self.py_code == self.cpp_code

    @property
    def identical(self):
        return self.ok and self.py_out == self.cpp_out

    def to_text(self) -> str:
        n = len(self.py_out.splitlines())
        head = (f"  program: `python {{file}}` and the compiled program, run the same way: "
                f"{n} line(s) of output, exit status {self.py_code}")
        if self.error:
            return head + f"\n    {self.error}"
        if self.identical:
            return head + f"\n    output identical, character for character; exit status the same" + \
                f" ({self.seconds_py:.2f} s in Python, {self.seconds_cpp:.2f} s in C++)"
        out = [head]
        if self.line is None and self.rounding:
            out.append(f"    output the same but for {self.rounding} number(s) that differ in their last digits "
                       "(agreeing within rtol: the last bit or two of a computation, as above)")
        elif self.line is not None:
            a = self.py_out.splitlines()
            b = self.cpp_out.splitlines()
            out.append(f"    OUTPUT DIFFERS at line {self.line}:")
            out.append(f"      Python: {a[self.line - 1] if self.line <= len(a) else '(no line)'}")
            out.append(f"      C++:    {b[self.line - 1] if self.line <= len(b) else '(no line)'}")
        if self.py_code != self.cpp_code:
            out.append(f"    EXIT STATUS DIFFERS: Python {self.py_code}, C++ {self.cpp_code}")
        for name, err in (("Python", self.py_err), ("C++", self.cpp_err)):
            if err.strip():
                out.append(f"    {name} wrote to stderr: " + err.strip().splitlines()[-1][:200])
        return "\n".join(out)


PROGRAM_SECONDS = 120.0


def run_program(py_file, exe, rtol) -> ProgramCheck:
    """Run the Python file and the compiled program, each in the file's folder, and compare."""
    folder = Path(py_file).resolve().parent
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    here = str(Path(__file__).resolve().parent.parent)  # this qtrans, for `from qtrans.typing import Vec`
    env["PYTHONPATH"] = os.pathsep.join(x for x in (env.get("PYTHONPATH"), here) if x)
    chk = ProgramCheck()
    try:
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, str(Path(py_file).resolve())], capture_output=True, cwd=folder, env=env,
                           timeout=PROGRAM_SECONDS)
        chk.seconds_py = time.perf_counter() - t0
        t0 = time.perf_counter()
        c = subprocess.run([str(exe)], capture_output=True, cwd=folder, timeout=PROGRAM_SECONDS)
        chk.seconds_cpp = time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        chk.error = f"it didn't finish within {PROGRAM_SECONDS:g} s ({Path(str(e.cmd[0])).name})"
        return chk

    def text(b):
        return b.decode("utf-8", errors="replace").replace("\r\n", "\n")
    chk.py_out, chk.cpp_out = text(p.stdout), text(c.stdout)
    chk.py_err, chk.cpp_err = text(p.stderr), text(c.stderr)
    chk.py_code, chk.cpp_code = p.returncode, c.returncode
    if chk.py_out != chk.cpp_out:
        chk.line, chk.rounding = printed_numbers_differ(chk.py_out, chk.cpp_out, rtol)
    return chk


# --- comparison ---------------------------------------------------------------------------------
def ulp_distance(a: float, b: float) -> int:
    def key(x):
        i = struct.unpack("<q", struct.pack("<d", x))[0]
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
    return abs(key(a) - key(b))


def _compare_values(pv, cv, t, rtol, atol):
    """-> (equal, rel_err, ulps)"""
    if t == VOID:
        return (True, 0.0, 0)
    if isinstance(t, DictType):  # the same keys in the same order (it shows when printed), values close
        if list(pv) != list(cv):
            return (False, math.inf, 0)
        return _compare_values(list(pv.values()), list(cv.values()), ListType(t.val), rtol, atol)
    if isinstance(t, ListType):
        if len(pv) != len(cv):
            return (False, math.inf, 0)
        worst = (True, 0.0, 0)
        for a, b in zip(pv, cv):
            eq, r, u = _compare_values(a, b, t.elem, rtol, atol)
            worst = (worst[0] and eq, max(worst[1], r), max(worst[2], u))
        return worst
    if isinstance(t, TupleType):
        worst = (True, 0.0, 0)
        for a, b, et in zip(pv, cv, t.elts):
            eq, r, u = _compare_values(a, b, et, rtol, atol)
            worst = (worst[0] and eq, max(worst[1], r), max(worst[2], u))
        return worst
    if isinstance(t, StructType):
        info, _ = _CLASSES[t.name]
        if t.ref:  # the same class on both sides, then that class's fields
            pc = type(pv).__name__
            cc = cv["__class__"] if isinstance(cv, dict) else type(cv).__name__
            if pc != cc:
                return (False, math.inf, 0)
            info, _ = _CLASSES[pc]
        worst = (True, 0.0, 0)
        get = (lambda o, k: o[k]) if isinstance(cv, dict) else getattr  # (a nudged Python object on both sides)
        for f in info.fields:
            eq, r, u = _compare_values(getattr(pv, f.name), get(cv, f.name), f.type, rtol, atol)
            worst = (worst[0] and eq, max(worst[1], r), max(worst[2], u))
        return worst
    if t == RNG:  # a generator inside an object: equal if its next raw output is
        if isinstance(pv, np.random.Generator):
            pv = rng_fingerprint(copy.deepcopy(pv))
        if isinstance(cv, np.random.Generator):  # (a compiled object's generator field, via the bindings)
            cv = rng_fingerprint(copy.deepcopy(cv))
        return (pv == cv, 0.0 if pv == cv else math.inf, 0)
    if isinstance(t, OptionalType):
        if pv is None or cv is None:
            return (pv is None and cv is None, 0.0 if pv is None and cv is None else math.inf, 0)
        return _compare_values(pv, cv, t.inner, rtol, atol)
    if isinstance(t, ArrayType):
        return _compare_arrays(pv, cv, rtol, atol)
    if isinstance(t, IntArrayType):  # ints: exactly equal, or not
        eq = pv.shape == cv.shape and bool(np.array_equal(pv, cv))
        return (eq, 0.0 if eq else math.inf, 0)
    if t != FLOAT:
        return (pv == cv, 0.0 if pv == cv else math.inf, 0)
    if math.isnan(pv) or math.isnan(cv):
        return (math.isnan(pv) and math.isnan(cv), 0.0, 0)
    if math.isinf(pv) or math.isinf(cv):
        return (pv == cv, 0.0 if pv == cv else math.inf, 0)
    diff = abs(pv - cv)
    scale = max(abs(pv), abs(cv))
    rel = diff / scale if scale > 0 else 0.0
    return (diff <= atol or rel <= rtol, rel, ulp_distance(pv, cv))


def _compare_arrays(p, c, rtol, atol):
    """Normwise: every element within atol + rtol * (largest |value| in the array). Elementwise
    relative error is meaningless for an element that is tiny because of cancellation, e.g. a
    zero-cost weight in a matrix product; the array's scale is the right yardstick, as in
    numerical linear algebra."""
    if p.shape != c.shape:
        return (False, math.inf, 0)
    if p.size == 0:
        return (True, 0.0, 0)
    pn, cn = np.isnan(p), np.isnan(c)
    if not np.array_equal(pn, cn):
        return (False, math.inf, 0)
    pi, ci = np.isinf(p), np.isinf(c)
    if not np.array_equal(pi, ci) or not np.array_equal(p[pi], c[ci]):
        return (False, math.inf, 0)
    fin = ~(pn | pi)
    if not fin.any():
        return (True, 0.0, 0)
    d = np.abs(p[fin] - c[fin])
    scale = float(np.max(np.abs(p[fin])))
    rel = float(d.max()) / scale if scale > 0 else 0.0
    eq = bool(np.all(d <= atol + rtol * scale))
    # ULPs only over elements that matter at the array's scale (>= 1% of its largest
    # value): a near-zero element of a large-valued array has a meaningless ULP count
    big = np.abs(p[fin]) >= 0.01 * scale
    ulps = max((ulp_distance(float(a), float(b)) for a, b in zip(p[fin][big], c[fin][big])), default=0)
    return (eq, rel, ulps)


def _nonfinite_ties(p, c, rtol, atol) -> bool:
    """p and c agree except at elements that are non-finite on both sides (inf vs NaN, or
    inf vs -inf), of which there is at least one."""
    if isinstance(p, tuple) or isinstance(c, tuple):
        if not (isinstance(p, tuple) and isinstance(c, tuple) and len(p) == len(c)):
            return False
        ties = [_nonfinite_ties(a, b, rtol, atol) for a, b in zip(p, c)]
        return any(ties) and all(t or _compare_values_loose(a, b, rtol, atol) for t, a, b in zip(ties, p, c))
    try:
        pa, ca = np.asarray(p, dtype=np.float64), np.asarray(c, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if pa.shape != ca.shape:
        return False
    both = ~np.isfinite(pa) & ~np.isfinite(ca)
    differ = both & ~((pa == ca) | (np.isnan(pa) & np.isnan(ca)))
    if not differ.any():
        return False
    return _compare_arrays(np.where(both, 0.0, pa).reshape(-1), np.where(both, 0.0, ca).reshape(-1), rtol, atol)[0]


def _compare_values_loose(a, b, rtol, atol) -> bool:
    try:
        pa, ca = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError):
        return a == b
    return pa.shape == ca.shape and _compare_arrays(pa.reshape(-1), ca.reshape(-1), rtol, atol)[0]


def classify(case: Case, ret, rtol, atol, exact_floats=False):
    """exact_floats: the function raises Python's float errors itself (Signature.float_errors,
    since 1.0), so where Python raises one the C++ must raise it too: that's a match, and
    anything else a mismatch, not a known difference."""
    if case.unstable:
        case.outcome = "mismatch"
        return
    py, cpp = case.py, case.cpp
    if py[0] == "ok" and cpp[0] == "ok":
        eq, rel, ulps = _compare_values(py[1], cpp[1], ret, rtol, atol)
        case.rel_err, case.ulps = rel, ulps
        case.outcome = "match" if eq else "mismatch"
        case.tiny = eq and not _compare_values(py[1], cpp[1], ret, rtol, 0.0)[0]
    elif py[0] == "exc" and cpp[0] == "exc" and (py[1] == cpp[1] or py[1] in CPP_EXC_MATCHES.get(cpp[1], set())):
        case.outcome = "match"
    elif cpp[0] == "exc" and cpp[1] == "IntOverflow":
        # Python's int grew past 64 bits (or a float became one), where C++ raises
        # OverflowError: the documented 64-bit int difference, never a silently wrong number
        case.outcome = "known"
    elif py[0] == "exc" and (
            (not exact_floats and py[1] in ("ZeroDivisionError", "OverflowError"))
            # (-1e308) ** 2.5: complex, and too big for one: OverflowError in Python, NaN in C (as above)
            or (py[1] == "OverflowError" and "complex" in py[2])
            or (not exact_floats and py[1] == "ValueError" and _float_error_message(py[2]))
            # x ** y with x < 0 and fractional y: a Python float goes complex (and the next
            # real-only operation raises TypeError); C, like np.float64, gives NaN
            or (py[1] == "TypeError" and "complex" in py[2])):
        # Python raised a float error where IEEE-754 C++ carries on with inf/NaN. What C++
        # does next (returns NaN, or a later check throws) follows from that, not from a
        # translation error.
        case.outcome = "known"
    elif py[0] == "exc" and cpp[0] == "exc" and py[1] in _ARG_ERRORS and cpp[1] in _ARG_ERRORS:
        # both reject the input, with different errors: it breaks two checks in one expression
        # (shapes that don't broadcast, and an index out of range or a division by zero). Python raises
        # the first in left-to-right order; C++ leaves the order of a call's arguments
        # unspecified (GCC and MSVC usually go right to left, Clang left to right).
        case.outcome = "known"
        case.order = True
    else:
        case.outcome = "mismatch"


# The errors that reject an input (a shape, an index, a singular matrix, a division by zero),
# as opposed to a wrong value: when an input breaks two of these checks, which one is raised
# first may differ (see classify).
# OverflowError: int(x) of an inf the C++ carried on with (math.log(0.0) is -inf in C++, a
# ValueError in Python), raised before Python's own first error in another argument
_ARG_ERRORS = {"ValueError", "IndexError", "ShapeError", "LinAlgError", "ZeroDivisionError", "ArithmeticError",
               "OverflowError"}


def _ulps(v, k):
    """v moved by k ULPs (k may be an array of +-4): whole steps of the spacing between doubles,
    so a subnormal moves too (multiplying it by 1 + 4 eps rounds back to itself). Zeros, inf
    and NaN stay as they are."""
    move = np.isfinite(v) & (v != 0)
    return np.where(move, v + k * np.spacing(np.where(move, v, 1.0)), v)


def _nudge(v, rng, normwise=False):
    """Move every float in v by 4 ULPs in a random direction: of itself, or (normwise=True,
    arrays only) of the array's largest element. The second is the size of error a backward-
    stable linear-algebra routine (LAPACK's, Eigen's) is allowed to make; two such routines can
    differ by as much as Python's answer moves under it, even on a badly scaled matrix whose
    answer barely moves under elementwise nudges."""
    eps = np.finfo(np.float64).eps
    if isinstance(v, np.ndarray):
        if normwise and v.size:
            finite = v[np.isfinite(v)]
            scale = float(np.max(np.abs(finite))) if finite.size else 0.0
            return v + rng.choice([-4.0, 4.0], size=v.shape) * eps * scale
        return _ulps(v, rng.choice([-4.0, 4.0], size=v.shape))
    if isinstance(v, float):
        return float(_ulps(np.float64(v), float(rng.choice([-4.0, 4.0]))))
    if _is_object(v):
        if isinstance(v, tuple):  # NamedTuple
            return type(v)(*(_nudge(x, rng, normwise) for x in v))
        c = copy.deepcopy(v)
        names = [f.name for f in dataclasses.fields(c)] if dataclasses.is_dataclass(c) else list(vars(c))
        for name in names:
            object.__setattr__(c, name, _nudge(getattr(c, name), rng, normwise))
        return c
    return v


_LAPACK = {"solve", "inv", "pinv", "eigh", "eigvalsh", "cholesky", "det", "lstsq"}
# BLAS-backed products: with inf/NaN or values near overflow, the order of their additions
# decides inf vs NaN and whether an intermediate overflows
_BLAS = {"dot", "matmul", "cov", "einsum", "outer", "inner", "vdot", "tensordot"}


def linalg_functions(paths) -> set:
    """Functions that use matrix products or numpy.linalg, directly or through other functions
    (of any module given): where 'edge' applies (see edge_inputs)."""
    direct, calls = set(), {}
    for p in paths:
        tree = ast.parse(Path(p).read_text(encoding="utf-8-sig"))
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        defs += [m for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body if isinstance(m, ast.FunctionDef)]
        for f in defs:
            if any((isinstance(n, ast.Attribute) and n.attr in _LAPACK | _BLAS) or
                   (isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult)) for n in ast.walk(f)):
                direct.add(f.name)
            calls.setdefault(f.name, set()).update(
                n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute)))
    out, changed = set(direct), True
    while changed:
        changed = False
        for name, cs in calls.items():
            if name not in out and cs & out:
                out.add(name)
                changed = True
    return out


def edge_inputs(args) -> bool:
    """Does any input number lie where floating point stops being ordinary: inf, NaN, a
    subnormal, or beyond 1e150 (a product of two such overflows)?"""
    def bad(x):
        x = np.asarray(x, dtype=np.float64)
        a = np.abs(x[np.isfinite(x)])
        return bool((~np.isfinite(x)).any() or (a > 1e150).any() or ((a > 0) & (a < 2.2250738585072014e-308)).any())
    for v in args:
        if isinstance(v, (float, np.ndarray)) and not isinstance(v, bool):
            if np.size(v) and bad(v):
                return True
        elif _is_object(v):
            items = v if isinstance(v, tuple) else (vars(v).values() if hasattr(v, "__dict__") else ())
            if edge_inputs(list(items)):
                return True
    return False


def lapack_functions(path) -> set:
    """Functions that call numpy.linalg's LAPACK-backed routines, directly or through other
    functions of the module. Only for these is normwise error the right yardstick."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    direct = {name for name, f in defs.items() if any(
        isinstance(n, ast.Attribute) and n.attr in _LAPACK for n in ast.walk(f))}
    calls = {name: {n.func.id for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
             for name, f in defs.items()}
    out, changed = set(direct), True
    while changed:
        changed = False
        for name, cs in calls.items():
            if name not in out and cs & out:
                out.add(name)
                changed = True
    return out


def python_sensitivity(fn, case, sig, out_t, rng, trials=3, normwise_too=False):
    """How far Python's own answer moves (relative) when the inputs move by rounding-sized
    amounts: 4 ULPs of each value, and (for functions using LAPACK-style linear algebra, which
    is accurate relative to a matrix's norm, not to each element) 4 ULPs of the largest value."""
    worst = 0.0
    has_array = any(isinstance(a, np.ndarray) for a in case.args)
    for normwise in ((False, True) if has_array and normwise_too else (False,)):
        for _ in range(trials):
            nudged = run_python(fn, tuple(_nudge(a, rng, normwise) for a in case.args), sig)
            if is_timeout(nudged):
                return worst  # stop: each hung call costs PYTHON_CASE_SECONDS and a busy thread
            if nudged[0] != "ok":
                return math.inf  # a nudge changes the outcome: the problem sits on a knife edge
            worst = max(worst, _compare_values(case.py[1], nudged[1], out_t, 0.0, 0.0)[1])
    return worst


def python_flips(fn, case, sig, rng, trials=3, normwise_too=False) -> bool:
    """Does a rounding-sized nudge of the inputs change whether Python raises? For linear
    algebra on a numerically singular matrix (condition number near 1/eps), whether LU meets an
    exactly-zero pivot is decided by rounding, so "Singular matrix" on one side and a result on
    the other is a knife edge, not a wrong translation."""
    has_array = any(isinstance(a, np.ndarray) for a in case.args)
    for normwise in ((False, True) if has_array and normwise_too else (False,)):
        for _ in range(trials):
            nudged = run_python(fn, tuple(_nudge(a, rng, normwise) for a in case.args), sig)
            if is_timeout(nudged):
                return False
            if nudged[0] != case.py[0]:
                return True
    return False


# --- main entry point ----------------------------------------------------------------------------
def verify_file(path, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                compiler=None, build_dir=None, eigen=None, profile="strict", strict=False) -> VerifyReport:
    """Translate, compile and check one module. The project modules it imports are translated
    and compiled with it (their own functions are checked by verifying them)."""
    rep = verify_project([path], n_random, seed, rtol, atol, bench, compiler, build_dir, eigen, profile=profile,
                         strict=strict)
    return rep.modules[Path(path).resolve().stem]


@dataclass
class ProjectVerifyReport:
    modules: dict  # module -> VerifyReport

    @property
    def ok(self):
        return all(r.ok for r in self.modules.values())

    @property
    def mismatches(self):
        return [c for r in self.modules.values() for c in r.mismatches]

    def to_text(self) -> str:
        return "\n\n".join(r.to_text() for r in self.modules.values())


def load_project(project):
    """The Python modules of a project, imported the way they import each other (by name, from
    their folder), so a class is the same class wherever it is used. -> {module: module object}"""
    if len(project.modules) == 1:
        (name, path), = project.paths.items()
        return {name: load_module(path)}
    def package_of(p):  # a folder with __init__.py is a package: its modules import as pkg.mod
        folder = Path(p).parent
        return folder.name if (folder / "__init__.py").exists() else None
    folders = set()
    for name, p in project.paths.items():
        pkg = package_of(p)
        folders.add(str(Path(p).parent.parent if pkg else Path(p).parent))
        for key in [k for k in sys.modules if k == name or (pkg and (k == pkg or k.startswith(pkg + ".")))]:
            sys.modules.pop(key, None)  # a module of that name from another folder (or an older run)
    sys.path[:0] = sorted(folders)
    try:
        return {name: importlib.import_module(f"{package_of(p)}.{name}" if package_of(p) else name)
                for name, p in ((n, project.paths[n]) for n in project.order)}
    finally:
        for f in folders:
            sys.path.remove(f)


def setup_inputs(project, pymods):
    """Point the input generators at this project: its string literals, its classes."""
    global _STRINGS, _CLASSES, _INIT_PARAMS
    lits, text = set(), False
    for p in project.paths.values():
        for n in ast.walk(ast.parse(Path(p).read_text(encoding="utf-8-sig"))):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 80:
                lits.add(n.value)
            elif isinstance(n, ast.Call) and (
                    (isinstance(n.func, ast.Attribute) and n.func.attr in _TEXT_METHODS)
                    or (isinstance(n.func, ast.Name) and n.func.id in ("float", "int") and n.args
                        and not isinstance(n.args[0], ast.Constant))):
                text = True
    # code that takes text apart also gets text made to test it: spaces, case, digits, Unicode
    _STRINGS = sorted(lits) + ["", "no such option"] + (_TEXT_PROBES if text else [])
    _CLASSES = {n: (info, getattr(pymods[m], n)) for n, (info, m) in project.classes().items()}
    _INIT_PARAMS = {}
    for n, (info, m) in project.classes().items():
        if info.plain:  # its own __init__, or the one it inherits (none: an interface)
            init = info.find_method("__init__")
            _INIT_PARAMS[n] = project.modules[m].signatures[init.key].params if init is not None else []


def module_cases(project, module, pymod) -> tuple[dict, dict]:
    """The hand-picked cases and tolerances for a module: QTRANS_CASES / QTRANS_TOLERANCE in the
    module itself, plus those for it in a qtrans_cases.py next to it."""
    from .project import project_cases
    given = dict(getattr(pymod, "QTRANS_CASES", {}) or {})
    tolerances = dict(getattr(pymod, "QTRANS_TOLERANCE", {}) or {})
    extra_cases, extra_tol = project_cases(Path(project.paths[module]).parent)
    for name, cases in (extra_cases.get(module) or {}).items():
        given[name] = list(given.get(name, [])) + list(cases)
    tolerances.update(extra_tol.get(module) or {})
    return given, tolerances


def project_lapack_functions(project) -> set:
    """lapack_functions across modules (a function calling an imported one that uses LAPACK)."""
    direct, calls = set(), {}
    for p in project.paths.values():
        tree = ast.parse(Path(p).read_text(encoding="utf-8-sig"))
        for f in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
            if any(isinstance(n, ast.Attribute) and n.attr in _LAPACK for n in ast.walk(f)):
                direct.add(f.name)
            calls.setdefault(f.name, set()).update(
                n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute)))
    out, changed = set(direct), True
    while changed:
        changed = False
        for name, cs in calls.items():
            if name not in out and cs & out:
                out.add(name)
                changed = True
    return out


def verify_project(paths, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                   compiler=None, build_dir=None, eigen=None, all_modules=False,
                   profile="strict", strict=False) -> ProjectVerifyReport:
    """See _verify_once. Where a function is your own C++ (<module>_overrides.hpp), it is what
    is checked against the Python; qtrans's own translation of it (when it has one) is then
    checked too, on exactly the same inputs, and the two C++ versions compared with each other.
    strict=True: an override written for an earlier version of the Python fails the check."""
    global _REPLAY
    rep = _verify_once(paths, n_random, seed, rtol, atol, bench, compiler, build_dir, eigen, all_modules, profile)
    replay = {}
    for m, r in rep.modules.items():
        r.strict = strict
        for f in r.functions:
            ov = getattr(r.translation, "overrides", {}).get(f.name)
            if ov is not None:
                f.override = ov
                if getattr(ov, "qtrans_could", None) is None:
                    replay[(m, f.name)] = [(c.args, c.source) for c in f.cases]
    if not replay:
        return rep
    from .translator import overrides_off
    base = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="qtrans_"))
    _REPLAY = replay
    try:
        with overrides_off():
            ref = _verify_once(paths, n_random, seed, rtol, atol, bench, compiler, base / "qtrans_translation",
                               eigen, all_modules, profile)
    finally:
        _REPLAY = None
    for m, r in rep.modules.items():
        theirs = {f.name: f for f in ref.modules[m].functions} if m in ref.modules else {}
        for f in r.functions:
            rf = theirs.get(f.name)
            if f.override is None or rf is None:
                continue
            f.reference = rf
            sig = r.translation.signatures[f.name]
            both = same = ulps = 0
            for a, b in zip(f.cases, rf.cases):
                if not (isinstance(a.cpp, tuple) and isinstance(b.cpp, tuple) and a.cpp[:1] == b.cpp[:1] == ("ok",)):
                    continue
                both += 1
                eq, _, u = _compare_values(b.cpp[1], a.cpp[1], sig.ret, 0.0, 0.0)
                same += bool(eq)
                ulps = max(ulps, u)
            f.vs_reference = (both, same, ulps)
    return rep


def _verify_once(paths, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                 compiler=None, build_dir=None, eigen=None, all_modules=False,
                 profile="strict") -> ProjectVerifyReport:
    """Translate the given files (and the project modules they import), compile one driver for
    all of them, and check the functions of the given files (all_modules=True: of every module).

    profile "strict" (the default): random inputs spread over typical magnitudes, square and
    rectangular arrays, plus your hand-picked cases. "stress": also infinities, NaN, the largest
    and smallest doubles, subnormals, signed zeros and rounding ties in 30% of the numbers, and
    longer arrays. (Integers stay moderate: a loop over range(2**62) would run as long in Python.)"""
    if profile not in ("strict", "stress"):
        raise ValueError(f"unknown profile {profile!r}: use 'strict' or 'stress'")
    global _STRESS
    _STRESS = profile == "stress"
    from .project import translate_project
    build = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="qtrans_"))
    build.mkdir(parents=True, exist_ok=True)
    project = translate_project(paths, build)
    targets = list(project.order) if all_modules else project.requested
    pymods = load_project(project)
    setup_inputs(project, pymods)
    global _WORKER
    _WORKER = PythonWorker(_worker_specs(project, pymods), _CLASSES)
    for key, mod in pymods.items():
        _WORKER.register(key, mod)
    try:
        return _verify_loaded(project, pymods, targets, build, n_random, seed, rtol, atol, bench, compiler, eigen)
    finally:
        _WORKER.stop()
        _WORKER = None


def _worker_specs(project, pymods):
    """How the worker imports each module: by file under the same fresh name, or by name."""
    specs = {}
    for key, mod in pymods.items():
        name = mod.__name__
        if name.startswith("_qtrans_verify_"):
            specs[key] = ("file", str(project.paths[key]), name)
        else:
            root = Path(project.paths[key]).parent
            if "." in name:  # a package: import from the folder above it
                root = root.parent
            specs[key] = ("import", str(root), name)
    return specs


_REPLAY = None  # (module, function) -> [(args, source)]: check only these, on exactly these inputs


def _verify_loaded(project, pymods, targets, build, n_random, seed, rtol, atol, bench, compiler, eigen):
    single = len(project.modules) == 1
    rng = np.random.default_rng(seed)

    checks = {m: [] for m in targets}
    for module in targets:
        result, pymod = project.modules[module], pymods[module]
        given, tolerances = module_cases(project, module, pymod)
        for name, sig in result.signatures.items():
            if sig.owner:
                continue  # methods are checked through the functions that use them
            if sig.ret.cpp() == "void" and not _compared_after(sig) and name not in result.prints:
                continue  # nothing to compare
            tol = tolerances.get(name, rtol)
            f_rtol, f_atol = (tol, atol) if isinstance(tol, (int, float)) else tuple(tol)
            if _REPLAY is not None and (module, name) not in _REPLAY:
                continue
            fc = FunctionCheck(name, rtol=f_rtol, atol=f_atol)
            fc.key = name if single else f"{module}:{name}"
            fc.module = module
            fc.py_name = sig.name
            fn = getattr(pymod, sig.name)
            if _REPLAY is not None:
                fc.cases = [Case(name, args, src) for args, src in _REPLAY[(module, name)]]
                fc.shape_plans = []

            defaults = [p.default for p in inspect.signature(fn).parameters.values()]
            for args in (given.get(sig.name, []) if _REPLAY is None else []):
                args = list(args)
                if getattr(sig, "overloaded", False) and any(
                        isinstance(v, bool) != (p.type == BOOL) or (isinstance(v, int) and not isinstance(v, bool)) != (p.type == INT)
                        for v, p in zip(args, sig.params) if p.type in (INT, FLOAT, BOOL)):
                    continue  # a hand-picked case for another version (an int case, a float version)
                # cases may leave out trailing arguments that have defaults
                for i in range(len(args), len(sig.params)):
                    if defaults[i] is inspect.Parameter.empty:
                        raise ValueError(f"QTRANS_CASES['{name}'] has a case with {len(args)} arguments "
                                         f"but '{sig.params[i].name}' has no default")
                    args.append(defaults[i])
                if len(args) != len(sig.params):
                    raise ValueError(f"QTRANS_CASES['{name}'] has a case with {len(args)} arguments; "
                                     f"{name}() takes {len(sig.params)}")
                args = tuple(_py_value(a, p.type) for a, p in zip(args, sig.params))
                fc.cases.append(Case(name, args, "given"))
            plans = shape_plans(fn, sig, rng) if _REPLAY is None else []
            fc.shape_plans = plans
            for _ in range(n_random if _REPLAY is None else 0):
                try:
                    args = random_args(rng, sig, plans)
                except _CantBuild:
                    continue  # an object its own checks refuse for every draw tried
                fc.cases.append(Case(name, args, "random"))
            kept, hung_on_edge = [], False
            for k, c in enumerate(fc.cases):
                if hung_on_edge and edge_inputs(c.args):
                    fc.timeouts += 1  # it hung on such inputs before: don't wait for it again
                    continue
                c.py, c.py_out = run_python(fn, c.args, sig, printed=True)
                if is_timeout(c.py):
                    fc.timeouts += 1
                    if edge_inputs(c.args) and not hung_on_edge:
                        hung_on_edge = True  # skip its other inf / NaN / near-overflow cases
                        continue
                    fc.timeouts += len(fc.cases) - k - 1  # it hangs on ordinary inputs: stop
                    break
                kept.append(c)
            fc.cases = kept
            checks[module].append(fc)

    # compile one driver for everything
    cxx = find_compiler(compiler)
    driver = build / "qtrans_driver.cpp"
    if single:
        (module, result), = project.modules.items()
        src = driver_source(result, f"{module}.hpp", result.namespace)
    else:
        src = driver_source(project, targets=targets)
    driver.write_text(src, encoding="utf-8")
    if not (build / "qtrans_runtime.hpp").exists():
        shutil.copy(RUNTIME_HEADER, build / "qtrans_runtime.hpp")
    shutil.copy(PYTHON_HEADER, build / "qtrans_python.hpp")
    shutil.copy(PYRANDOM_HEADER, build / "qtrans_pyrandom.hpp")
    shutil.copy(PRINT_HEADER, build / "qtrans_print.hpp")
    shutil.copy(LIST_HEADER, build / "qtrans_list.hpp")
    shutil.copy(UNICODE_HEADER, build / "qtrans_unicode.hpp")
    for h in NPMATH_HEADERS:  # np.exp and friends, as NumPy computes them (scalar code uses them too)
        shutil.copy(h, build / h.name)
    for m, r in project.modules.items():  # the driver includes each module's header, programs' too
        if not (build / f"{m}.hpp").exists():
            (build / f"{m}.hpp").write_text(r.header, encoding="utf-8")
    system = []
    if project.uses_arrays:
        shutil.copy(ARRAYS_HEADER, build / "qtrans_arrays.hpp")
        system.append(find_eigen(eigen))
    if project.uses_random:
        shutil.copy(RANDOM_HEADER, build / "qtrans_random.hpp")
    exe = build / ("qtrans_driver.exe" if os.name == "nt" else "qtrans_driver")
    # the linear-algebra audit (qtrans_arrays.hpp) is compiled in: every numpy.linalg operation a
    # call makes measures its own backward error; it only reads, so the results are the same
    cmd = cxx.command(driver, exe, build, system, defines=("QTRANS_LINALG_AUDIT",) if project.uses_arrays else ())
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=build)
    if proc.returncode != 0:
        raise RuntimeError(f"C++ compilation failed:\n{' '.join(cmd)}\n{proc.stderr}{proc.stdout}")
    diagnostics = (proc.stderr + proc.stdout).splitlines()

    # run every case in one process
    lines, index = [], []
    for module in targets:
        sigs = project.modules[module].signatures
        for fc in checks[module]:
            for c in fc.cases:
                enc = encode_call(c.args, sigs[fc.name])
                lines.append(f"CALL {fc.key} {enc}")
                index.append((fc, c))
    run = subprocess.run([str(exe)], input="".join(x + "\n" for x in lines), capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=600)
    outputs, audits, printed = [], [], []
    for out_line in run.stdout.splitlines():
        if out_line.startswith("AUDIT ") and outputs:
            audits[-1] = _parse_audit(out_line)
        elif out_line.startswith("PRINTED ") and outputs:
            printed[-1] = bytes.fromhex(out_line.split()[1][1:]).decode("utf-8", errors="replace")
        else:
            outputs.append(out_line)
            audits.append(None)
            printed.append("")
    if len(outputs) != len(lines):
        raise RuntimeError(f"C++ driver produced {len(outputs)} results for {len(lines)} cases "
                           f"(exit code {run.returncode}): {run.stderr[-500:]}")
    nudge_rng = np.random.default_rng(seed + 1)
    normwise = lapack_functions(project.paths[targets[0]]) if single else project_lapack_functions(project)
    linalg = linalg_functions(project.paths.values())
    for (fc, c), outline, audit, out_text in zip(index, outputs, audits, printed):
        sig = project.modules[fc.module].signatures[fc.name]
        c.cpp = _parse_cpp_line(outline, output_type(sig))
        c.audit = audit
        c.cpp_out = out_text
        classify(c, output_type(sig), fc.rtol, fc.atol, exact_floats=getattr(sig, "float_errors", False))
        if c.outcome == "match" and c.py_out != c.cpp_out and not printed_agrees(c.py_out, c.cpp_out, fc.rtol):
            c.outcome = "mismatch"  # same result, but it printed something else
            c.printed_differs = True
        if c.unstable:
            continue  # a linalg operation broke its stability bound: a failure, whatever the values
        if c.outcome == "mismatch" and python_went_complex(getattr(pymods[fc.module], fc.py_name or fc.name), c.args, sig):
            # x ** y with x < 0 and fractional y went complex in Python and NaN in C++ before
            # this result: the documented difference, wherever it shows (or doesn't) in the value
            c.outcome = "known"
        if c.outcome == "mismatch" and c.py[0] == "ok" and c.cpp[0] == "ok":
            c.sensitivity = python_sensitivity(getattr(pymods[fc.module], fc.py_name or fc.name), c, sig, output_type(sig),
                                               nudge_rng, normwise_too=(fc.py_name or fc.name) in normwise)
            if c.rel_err <= 100 * c.sensitivity:
                c.outcome = "illcond"
        elif c.outcome == "mismatch" and (fc.py_name or fc.name) in linalg and (c.py[0] == "ok") != (c.cpp[0] == "ok") \
                and (c.py[1] if c.py[0] == "exc" else c.cpp[1]) in ("LinAlgError", "ValueError"):
            # one side found the matrix singular (LinAlgError is a ValueError): ill-conditioned
            # when Python's own verdict changes under a rounding-sized nudge
            fn = getattr(pymods[fc.module], fc.py_name or fc.name)
            if python_flips(fn, c, sig, nudge_rng, normwise_too=(fc.py_name or fc.name) in normwise) \
                    or (_singular_verdict(c) and python_near_singular(fn, c, sig)):
                c.sensitivity = math.inf
                c.outcome = "illcond"
        if c.outcome == "mismatch" and (fc.py_name or fc.name) in linalg and c.py[0] == "ok" and c.cpp[0] == "ok" \
                and _nonfinite_ties(c.py[1], c.cpp[1], fc.rtol, fc.atol):
            # differing elements are inf / -inf / NaN on both sides, all else agrees: a value that
            # is zero up to LAPACK's or Eigen's rounding (1e-16 vs exactly 0) was divided by
            # zero, so its sign decided the result. Rounding decides it: ill-conditioned.
            c.sensitivity = math.inf
            c.outcome = "illcond"
        if c.outcome == "mismatch" and (fc.py_name or fc.name) in linalg and edge_inputs(c.args):
            # matrix products / LAPACK on inf, NaN, subnormals or values near overflow: NumPy's
            # BLAS and Eigen add in different orders, so inf - inf -> NaN, or an intermediate
            # overflow, can happen on one side only. Both are correct; listed, never hidden.
            c.outcome = "edge"

    if bench:
        for module in targets:
            sigs = project.modules[module].signatures
            for fc in checks[module]:
                # time a representative call: a hand-picked case if there is one, else the random
                # case with the most data (the first random case can be an empty array)
                ok = [c for c in fc.cases if c.py[0] == "ok" and c.cpp[0] == "ok"]
                pool = [c for c in ok if c.source == "given"] or ok
                case = max(pool, key=lambda c: sum(np.size(a) for a in c.args if isinstance(a, np.ndarray)),
                           default=None)
                if case is None:
                    continue
                fn = getattr(pymods[module], fc.py_name or fc.name)
                enc = encode_call(case.args, sigs[fc.name])
                fc.bench_case = case
                for _ in range(max(1, int(bench))):  # bench=N: N rounds, Python and C++ taking turns
                    ns, reps = _bench_python_reps(fn, _call_args(case.args, sigs[fc.name]))
                    fc.py_rounds.append(ns)
                    fc.py_reps.append(reps)
                    r = subprocess.run([str(exe)], input=f"BENCH {fc.key} {enc}\n", capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
                    words = r.stdout.split()
                    fc.cpp_rounds.append(float(words[1]))
                    fc.cpp_reps.append(int(words[3]))
                fc.py_ns, fc.cpp_ns = statistics.median(fc.py_rounds), statistics.median(fc.cpp_rounds)

    reports = {}
    env = environment(cxx)
    for module in targets:
        warns = [w.strip() for w in diagnostics if "warning" in w.lower() and f"{module}.hpp" in w]
        reports[module] = VerifyReport(str(project.paths[module]), project.modules[module], f"{Path(cxx.path).name}",
                                       " ".join(cmd), warns, checks[module], rtol, atol, env)
        result = project.modules[module]
        if result.program is not None:  # a program: run it both ways and compare what it prints
            src = build / f"{module}.cpp"
            src.write_text(result.program, encoding="utf-8")
            for h in (RUNTIME_HEADER, PYTHON_HEADER, PYRANDOM_HEADER, PRINT_HEADER, LIST_HEADER, UNICODE_HEADER,
                      *NPMATH_HEADERS):
                shutil.copy(h, build / h.name)
            prog_system = list(system)
            if "qtrans_arrays.hpp" in result.program or "qtrans_random.hpp" in result.program:
                shutil.copy(ARRAYS_HEADER, build / "qtrans_arrays.hpp")
                shutil.copy(RANDOM_HEADER, build / "qtrans_random.hpp")
                for h in NPMATH_HEADERS:
                    shutil.copy(h, build / h.name)
                if not prog_system:
                    prog_system.append(find_eigen(eigen))
            prog = build / (f"{module}_program.exe" if os.name == "nt" else f"{module}_program")
            pcmd = cxx.command(src, prog, build, prog_system)
            pp = subprocess.run(pcmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=build)
            if pp.returncode != 0:
                raise RuntimeError(f"C++ compilation of the program failed:\n{' '.join(pcmd)}\n{pp.stderr}{pp.stdout}")
            reports[module].warnings += [w.strip() for w in (pp.stderr + pp.stdout).splitlines()
                                         if "warning" in w.lower() and f"{module}.cpp" in w]
            reports[module].program = run_program(project.paths[module], prog, rtol)
    return ProjectVerifyReport(reports)


def _bench_python(fn, args):
    return _bench_python_reps(fn, args)[0]


def _bench_python_reps(fn, args):
    """ns per call and the calls timed: one warm-up call, then runs of 1, 4, 16, ... calls until
    one lasts at least 0.1 s; only that last run is timed (the shorter ones warm up too)."""
    fn(*args)
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed > 0.1:
            return elapsed * 1e9 / reps, reps
        reps *= 4
