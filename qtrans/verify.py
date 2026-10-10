"""Proof that the translation is right: run every translated function in Python and in
compiled C++ on the same inputs and compare.

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
            differences between NumPy/LAPACK and Eigen are amplified; agreement to the
            problem's own precision is all any two correct implementations can reach.
  known     Python raised one of the float errors IEEE-754 arithmetic doesn't raise
            (ZeroDivisionError, OverflowError, math domain error) and C++ returned a
            value. Listed in the README as a deliberate difference.
  mismatch  anything else. A correct translation has zero of these.
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .translator import ARRAYS_HEADER, RANDOM_HEADER, RUNTIME_HEADER, TranslationResult, translate_file
from .types import BOOL, FLOAT, INT, MAT, RNG, VEC, ArrayType, TupleType

# Which Python exceptions a given C++ exception class stands for.
CPP_EXC_MATCHES = {
    "ValueError": {"ValueError"},
    "ArithmeticError": {"ArithmeticError", "ZeroDivisionError", "FloatingPointError"},
    "OverflowError": {"OverflowError"},
    "AssertionError": {"AssertionError"},
    "RuntimeError": {"RuntimeError", "NotImplementedError", "Exception"},
    "IndexError": {"IndexError"},
    # Eigen's internal shape / bounds checks can't tell NumPy's ValueError and IndexError apart
    "ShapeError": {"ValueError", "IndexError"},
}
_KNOWN_MESSAGES = ("math domain error", "math range error", "cannot convert float")


# --- compiler ------------------------------------------------------------------------
@dataclass
class Compiler:
    path: str
    kind: str  # "gnu" (g++/clang++) or "msvc"

    def command(self, source, exe, include_dir, system_includes=(), optimise=True):
        if self.kind == "msvc":
            return ([self.path, "/nologo", "/std:c++17", "/EHsc", "/W4", "/fp:precise", "/bigobj",
                     "/O2" if optimise else "/Od", f"/I{include_dir}"]
                    + [f"/external:I{d}" for d in system_includes] + ["/external:W0", str(source), f"/Fe{exe}"])
        # Eigen goes in as a *system* include so its own internal warnings don't count against
        # the generated code
        return ([self.path, "-std=c++17", "-O2" if optimise else "-O0", "-Wall", "-Wextra",
                 "-Wpedantic", "-Wshadow", "-Wconversion", "-Wdouble-promotion",
                 "-ffp-contract=off", f"-I{include_dir}"]
                + [f"-isystem{d}" for d in system_includes] + [str(source), "-o", str(exe)])


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


def find_compiler(explicit: str | None = None) -> Compiler:
    candidates = [explicit, os.environ.get("CXX"), "g++", "clang++", "c++", "cl"]
    for c in candidates:
        if not c:
            continue
        path = shutil.which(c)
        if path:
            kind = "msvc" if Path(path).stem.lower() == "cl" else "gnu"
            return Compiler(path, kind)
    raise RuntimeError(
        "No C++ compiler found. Install one of: g++ (Linux: apt install g++; Windows with "
        "Anaconda: conda install -c conda-forge m2w64-toolchain), clang++, or MSVC (Visual "
        "Studio Build Tools). Or set the CXX environment variable to its path.")


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

    @property
    def mismatches(self):
        return [c for f in self.functions for c in f.cases if c.outcome == "mismatch"]

    @property
    def ok(self):
        return not self.mismatches

    def to_text(self) -> str:
        t = self.translation.report
        lines = [f"qtrans verify: {self.file}",
                 f"  translation: {t.summary()}",
                 f"  compiler:    {self.compiler} ({len(self.warnings)} warnings in generated code)",
                 f"  tolerance:   rtol={self.rtol:g}, atol={self.atol:g}", ""]
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
        if self.mismatches:
            lines += ["", "  First mismatches:"]
            for c in self.mismatches[:10]:
                lines.append(f"    {c.func}{c.args}: Python {c.py}, C++ {c.cpp}")
        lines += ["", "  max rel err / max ULP exclude results within atol of each other (values near zero)",
                  "  and ill-conditioned cases; for arrays, rel err is normwise and ULPs count elements",
                  "  >= 1% of the array's largest value."]
        ill = [c for f in self.functions for c in f.cases if c.outcome == "illcond"]
        if ill:
            worst = max(ill, key=lambda c: c.sensitivity)
            lines += [f"  'ill-cond': {len(ill)} cases where a 4-ULP nudge of the inputs moves Python's own answer",
                      f"  by up to {worst.sensitivity:.1e} (relative); C++ agreed within 100x that."]
        known = [c for f in self.functions for c in f.cases if c.outcome == "known"]
        if known:
            c = known[0]
            lines += ["", f"  'known' = Python raised a float error where IEEE-754 C++ returns inf/NaN,",
                      f"  e.g. {c.func}{c.args}: Python {c.py[1]} ({c.py[2]}), C++ {c.cpp[1]}"]
        if self.warnings:
            lines += ["", "  Compiler warnings in generated code:"] + [f"    {w}" for w in self.warnings[:10]]
        lines += ["", "  RESULT: " + ("PASS (no mismatches)" if self.ok else f"FAIL ({len(self.mismatches)} mismatches)")]
        return "\n".join(lines)


# --- input generation -------------------------------------------------------------------
def random_value(rng, t, n=3):
    """A random argument of type t. Arrays use size n (Vec of n, Mat of n x n) so that the
    arguments of one call have compatible shapes."""
    if t == RNG:
        return int(rng.integers(0, 2 ** 31))  # a seed: each side builds its own generator from it
    if isinstance(t, ArrayType):
        shape = (n,) if t.rank == 1 else (n, n)
        return np.array([random_value(rng, FLOAT) for _ in range(int(np.prod(shape)))],
                        dtype=np.float64).reshape(shape)
    if t == FLOAT:
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


def _encode(v, t):
    if t == RNG:
        return str(int(v))
    if isinstance(t, ArrayType):
        a = np.asarray(v, dtype=np.float64)
        head = f"{a.shape[0]}" if t.rank == 1 else f"{a.shape[0]} {a.shape[1]}"
        return " ".join([head] + [float(x).hex() for x in a.ravel()])
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


def output_type(sig):
    """What a call produces: the return value, plus the final state of every array the
    function modifies in place (in Python the caller sees those changes too)."""
    # a generator passed in is reported as its next raw output after the call: equal only if
    # both sides consumed exactly the same number of draws
    refs = [INT if p.type == RNG else p.type for p in sig.params if getattr(p, "mode", "") == "ref"]
    if not refs:
        return sig.ret
    returned = [] if sig.ret.cpp() == "void" else _flatten(sig.ret)
    return TupleType(tuple(returned + refs))


# --- python side ------------------------------------------------------------------------
def _py_category(exc):
    for cls, name in ((ZeroDivisionError, "ZeroDivisionError"), (OverflowError, "OverflowError"),
                      (FloatingPointError, "FloatingPointError"), (ArithmeticError, "ArithmeticError"),
                      (ValueError, "ValueError"), (IndexError, "IndexError"),
                      (AssertionError, "AssertionError"),
                      (NotImplementedError, "NotImplementedError"), (RuntimeError, "RuntimeError")):
        if isinstance(exc, cls):
            return name
    return type(exc).__name__


def _py_value(v, t):
    if t == RNG:
        if isinstance(v, np.random.Generator):
            raise TypeError("give generator arguments in QTRANS_CASES as integer seeds")
        return int(v)
    if isinstance(t, TupleType):
        return tuple(_py_value(x, et) for x, et in zip(v, t.elts))
    if isinstance(t, ArrayType):
        a = np.array(v, dtype=np.float64, copy=True)
        if a.ndim != t.rank:
            raise TypeError(f"expected a {t.rank}-D array, got {a.ndim}-D")
        return a
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
        else:
            out.append(a)
    return out


def run_python(fn, args, sig):
    args = _call_args(args, sig)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                v = fn(*args)
    except Exception as e:  # noqa: BLE001 - every exception is a test outcome
        return ("exc", _py_category(e), str(e))
    try:
        refs = [int(args[i].bit_generator.random_raw()) if p.type == RNG else args[i]
                for i, p in enumerate(sig.params) if getattr(p, "mode", "") == "ref"]
        if refs:
            flat = list(v) if isinstance(sig.ret, TupleType) else [] if sig.ret.cpp() == "void" else [v]
            return ("ok", _py_value(tuple(flat + refs), output_type(sig)))
        return ("ok", _py_value(v, sig.ret))
    except (TypeError, ValueError) as e:
        if "complex" in str(e) or isinstance(v, complex):
            return ("exc", "TypeError", "Python returned a complex number")
        return ("bad", f"{v!r} ({e})")


def load_module(path):
    path = Path(path).resolve()
    name = f"_qtrans_verify_{path.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


# --- C++ side ---------------------------------------------------------------------------------
_PARSE = {FLOAT: "F", INT: "I", BOOL: "B", VEC: "V", MAT: "M"}
_CTYPE = {FLOAT: "double", INT: "long long", BOOL: "bool"}

_DRIVER_PRELUDE = r"""// Test driver generated by qtrans verify. Reads one call per line:
//   CALL <function> <args...>    -> OK <results...> | EXC <category>
//   BENCH <function> <args...>   -> BENCH <nanoseconds per call> <checksum>
// Numbers travel as hex floats (exact); arrays as their shape followed by the values.
#include "@@HEADER@@"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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

static void out(double x) { std::printf(" %a", x); }  // hex float: exact
static void out(long long x) { std::printf(" %lld", x); }
static void out(bool x) { std::printf(" %d", x ? 1 : 0); }
static double num(double x) { return x; }
static double num(long long x) { return static_cast<double>(x); }
static double num(bool x) { return x ? 1.0 : 0.0; }
@@ARRAY_IO@@
template <class... T>
static void out(const std::tuple<T...>& t) {
    std::apply([](const auto&... v) { (out(v), ...); }, t);
}
template <class... T>
static double num(const std::tuple<T...>& t) { return num(std::get<0>(t)); }

template <class Fn>
static void bench(Fn&& f) {
    // Repeat until the timing covers at least 0.1 s, so clock resolution doesn't matter.
    double sink = 0.0;
    long long reps = 1;
    double seconds = 0.0;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        for (long long k = 0; k < reps; ++k) sink += num(f());
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (seconds > 0.1 || reps > (1LL << 40)) break;
        reps *= 4;
    }
    std::printf("BENCH %.6g %a\n", seconds * 1e9 / static_cast<double>(reps), sink);
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string mode, fn, tok;
        in >> mode >> fn;
        Reader r;
        while (in >> tok) r.tokens.push_back(tok);
        try {
@@BRANCHES@@
            { std::printf("EXC UnknownFunction\n"); }
        }@@ARRAY_CATCH@@
        catch (const std::invalid_argument&) { std::printf("EXC ValueError\n"); }
        catch (const std::domain_error&) { std::printf("EXC ArithmeticError\n"); }
        catch (const std::out_of_range&) { std::printf("EXC IndexError\n"); }
        catch (const std::logic_error&) { std::printf("EXC AssertionError\n"); }
        catch (const std::overflow_error&) { std::printf("EXC OverflowError\n"); }
        catch (const std::runtime_error&) { std::printf("EXC RuntimeError\n"); }
        catch (const std::exception&) { std::printf("EXC Exception\n"); }
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
static double num(const qt::Mat& m) { return m.size() ? m(0, 0) : 0.0; }"""


_RNG_IO = r"""static void out(qt::Rng& g) {  // the generator's next raw output: shows how many draws were used
    std::printf(" %llu", static_cast<unsigned long long>(g.next_uint64()));
}"""


def driver_source(result: TranslationResult, header_name: str, namespace: str) -> str:
    branches = []
    for name, sig in result.signatures.items():
        if sig.ret.cpp() == "void" and not any(p.mode == "ref" for p in sig.params):
            continue
        # read arguments into named locals first: C++ doesn't fix the order in which a
        # call's arguments are evaluated, and the reader consumes tokens in sequence
        def read(i, p, volatile):
            if p.type == RNG:
                return f"qt::Rng p{i}(I(r));"
            if volatile and p.type in _CTYPE:
                return f"volatile {_CTYPE[p.type]} p{i} = {_PARSE[p.type]}(r);"
            return f"auto p{i} = {_PARSE[p.type]}(r);"
        reads = " ".join(read(i, p, False) for i, p in enumerate(sig.params))
        vreads = " ".join(read(i, p, True) for i, p in enumerate(sig.params))
        args = ", ".join(f"p{i}" for i in range(len(sig.params)))
        refs = "".join(f" out(p{i});" for i, p in enumerate(sig.params) if p.mode == "ref")
        call = f"{namespace}::{sig.cpp}({args})"
        if sig.ret.cpp() == "void":
            do_call = f'{call}; std::printf("OK");{refs} std::printf("\\n");'
            do_bench = f"bench([&] {{ {call}; return 0.0; }});"
        else:
            do_call = f'auto res = {call}; std::printf("OK"); out(res);{refs} std::printf("\\n");'
            do_bench = f"bench([&] {{ return {call}; }});"
        branches.append(
            f'            if (fn == "{name}") {{\n'
            f'                if (mode == "CALL") {{ {reads} {do_call} }}\n'
            f'                else {{ {vreads} {do_bench} }}\n'
            f'            }} else')
    arrays = result.uses_arrays
    io = (_ARRAY_IO if arrays else "") + ("\n" + _RNG_IO if result.uses_random else "")
    return (_DRIVER_PRELUDE.replace("@@HEADER@@", header_name)
            .replace("@@ARRAY_IO@@", io)
            .replace("@@ARRAY_CATCH@@", '\n        catch (const qt::shape_error&) { std::printf("EXC ShapeError\\n"); }'
                     if arrays else "")
            .replace("@@BRANCHES@@", "\n".join(branches)))


def _parse_cpp_line(line, out_t):
    parts = line.split()
    if not parts:
        return ("exc", "NoOutput")
    if parts[0] == "EXC":
        return ("exc", parts[1] if len(parts) > 1 else "Exception")
    toks = iter(parts[1:])
    values = []
    for t in _flatten(out_t):
        if isinstance(t, ArrayType):
            shape = (int(next(toks)),) if t.rank == 1 else (int(next(toks)), int(next(toks)))
            n = int(np.prod(shape))
            values.append(np.array([_decode_float(next(toks)) for _ in range(n)], dtype=np.float64).reshape(shape))
        elif t == FLOAT:
            values.append(_decode_float(next(toks)))
        elif t == INT:
            values.append(int(next(toks)))  # Python ints: also holds unsigned 64-bit raw outputs
        else:
            values.append(next(toks) == "1")
    return ("ok", tuple(values) if isinstance(out_t, TupleType) else values[0])


# --- comparison ---------------------------------------------------------------------------------
def ulp_distance(a: float, b: float) -> int:
    def key(x):
        i = struct.unpack("<q", struct.pack("<d", x))[0]
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
    return abs(key(a) - key(b))


def _compare_values(pv, cv, t, rtol, atol):
    """-> (equal, rel_err, ulps)"""
    if isinstance(t, TupleType):
        worst = (True, 0.0, 0)
        for a, b, et in zip(pv, cv, t.elts):
            eq, r, u = _compare_values(a, b, et, rtol, atol)
            worst = (worst[0] and eq, max(worst[1], r), max(worst[2], u))
        return worst
    if isinstance(t, ArrayType):
        return _compare_arrays(pv, cv, rtol, atol)
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


def classify(case: Case, ret, rtol, atol):
    py, cpp = case.py, case.cpp
    if py[0] == "ok" and cpp[0] == "ok":
        eq, rel, ulps = _compare_values(py[1], cpp[1], ret, rtol, atol)
        case.rel_err, case.ulps = rel, ulps
        case.outcome = "match" if eq else "mismatch"
        case.tiny = eq and not _compare_values(py[1], cpp[1], ret, rtol, 0.0)[0]
    elif py[0] == "exc" and cpp[0] == "exc" and py[1] in CPP_EXC_MATCHES.get(cpp[1], set()):
        case.outcome = "match"
    elif py[0] == "exc" and (
            py[1] in ("ZeroDivisionError", "OverflowError")
            or (py[1] == "ValueError" and any(m in py[2] for m in _KNOWN_MESSAGES))
            # x ** y with x < 0 and fractional y: a Python float goes complex (and the next
            # real-only operation raises TypeError); C, like np.float64, gives NaN
            or (py[1] == "TypeError" and "complex" in py[2])):
        # Python raised a float error where IEEE-754 C++ carries on with inf/NaN. What C++
        # does next (returns NaN, or a later check throws) follows from that, not from a
        # translation error.
        case.outcome = "known"
    else:
        case.outcome = "mismatch"


def _nudge(v, rng):
    """Move every float in v by up to 4 ULPs, in a random direction."""
    eps = np.finfo(np.float64).eps
    if isinstance(v, np.ndarray):
        return v * (1.0 + rng.choice([-4.0, 4.0], size=v.shape) * eps)
    if isinstance(v, float):
        return v * (1.0 + float(rng.choice([-4.0, 4.0])) * eps)
    return v


def python_sensitivity(fn, case, sig, out_t, rng, trials=3):
    """How far Python's own answer moves (relative) when the inputs move by 4 ULPs."""
    worst = 0.0
    for _ in range(trials):
        nudged = run_python(fn, tuple(_nudge(a, rng) for a in case.args), sig)
        if nudged[0] != "ok":
            return math.inf  # a nudge changes the outcome: the problem sits on a knife edge
        worst = max(worst, _compare_values(case.py[1], nudged[1], out_t, 0.0, 0.0)[1])
    return worst


# --- main entry point ----------------------------------------------------------------------------
def verify_file(path, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                compiler=None, build_dir=None, eigen=None) -> VerifyReport:
    path = Path(path)
    build = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="qtrans_"))
    build.mkdir(parents=True, exist_ok=True)
    result = translate_file(path, build)
    module = load_module(path)
    given = getattr(module, "QTRANS_CASES", {}) or {}
    tolerances = getattr(module, "QTRANS_TOLERANCE", {}) or {}
    rng = np.random.default_rng(seed)

    checks = []
    for name, sig in result.signatures.items():
        if sig.ret.cpp() == "void" and not any(p.mode == "ref" for p in sig.params):
            continue
        tol = tolerances.get(name, rtol)
        f_rtol, f_atol = (tol, atol) if isinstance(tol, (int, float)) else tuple(tol)
        fc = FunctionCheck(name, rtol=f_rtol, atol=f_atol)
        fn = getattr(module, name)
        defaults = [p.default for p in inspect.signature(fn).parameters.values()]
        for args in given.get(name, []):
            args = list(args)
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
        for _ in range(n_random):
            n = 0 if rng.random() < 0.05 else int(rng.integers(1, 7))
            fc.cases.append(Case(name, tuple(random_value(rng, p.type, n) for p in sig.params), "random"))
        for c in fc.cases:
            c.py = run_python(fn, c.args, sig)
        checks.append(fc)

    # compile the driver
    cxx = find_compiler(compiler)
    ns = result.header.split("namespace ", 1)[1].split(" ", 1)[0]
    driver = build / "qtrans_driver.cpp"
    driver.write_text(driver_source(result, f"{path.stem}.hpp", ns), encoding="utf-8")
    if not (build / "qtrans_runtime.hpp").exists():
        shutil.copy(RUNTIME_HEADER, build / "qtrans_runtime.hpp")
    system = []
    if result.uses_arrays:
        shutil.copy(ARRAYS_HEADER, build / "qtrans_arrays.hpp")
        system.append(find_eigen(eigen))
    if result.uses_random:
        shutil.copy(RANDOM_HEADER, build / "qtrans_random.hpp")
    exe = build / ("qtrans_driver.exe" if os.name == "nt" else "qtrans_driver")
    cmd = cxx.command(driver, exe, build, system)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=build)
    if proc.returncode != 0:
        raise RuntimeError(f"C++ compilation failed:\n{' '.join(cmd)}\n{proc.stderr}{proc.stdout}")
    diagnostics = (proc.stderr + proc.stdout).splitlines()
    warns = [w.strip() for w in diagnostics if "warning" in w.lower() and f"{path.stem}.hpp" in w]

    # run every case in one process
    lines, index = [], []
    for fc in checks:
        sig = result.signatures[fc.name]
        for c in fc.cases:
            enc = " ".join(_encode(v, p.type) for v, p in zip(c.args, sig.params))
            lines.append(f"CALL {fc.name} {enc}")
            index.append((fc, c))
    run = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    outputs = run.stdout.splitlines()
    if len(outputs) != len(lines):
        raise RuntimeError(f"C++ driver produced {len(outputs)} results for {len(lines)} cases "
                           f"(exit code {run.returncode}): {run.stderr[-500:]}")
    nudge_rng = np.random.default_rng(seed + 1)
    for (fc, c), outline in zip(index, outputs):
        sig = result.signatures[fc.name]
        c.cpp = _parse_cpp_line(outline, output_type(sig))
        classify(c, output_type(sig), fc.rtol, fc.atol)
        if c.outcome == "mismatch" and c.py[0] == "ok" and c.cpp[0] == "ok":
            c.sensitivity = python_sensitivity(getattr(module, fc.name), c, sig, output_type(sig), nudge_rng)
            if c.rel_err <= 100 * c.sensitivity:
                c.outcome = "illcond"

    if bench:
        for fc in checks:
            case = next((c for c in fc.cases if c.py[0] == "ok" and c.cpp[0] == "ok"), None)
            if case is None:
                continue
            fn = getattr(module, fc.name)
            fc.py_ns = _bench_python(fn, _call_args(case.args, result.signatures[fc.name]))
            sig = result.signatures[fc.name]
            enc = " ".join(_encode(v, p.type) for v, p in zip(case.args, sig.params))
            r = subprocess.run([str(exe)], input=f"BENCH {fc.name} {enc}\n", capture_output=True, text=True)
            fc.cpp_ns = float(r.stdout.split()[1])

    return VerifyReport(str(path), result, f"{Path(cxx.path).name}", " ".join(cmd), warns, checks, rtol, atol)


def _bench_python(fn, args):
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed > 0.1:
            return elapsed * 1e9 / reps
        reps *= 4
