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
import copy
import dataclasses
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
from .types import BOOL, FLOAT, INT, MAT, RNG, STR, VEC, ArrayType, OptionalType, StructType, TupleType

# Which Python exceptions a given C++ exception class stands for.
CPP_EXC_MATCHES = {
    "ValueError": {"ValueError"},
    "ArithmeticError": {"ArithmeticError", "ZeroDivisionError", "FloatingPointError"},
    "OverflowError": {"OverflowError"},
    "AssertionError": {"AssertionError"},
    "RuntimeError": {"RuntimeError", "NotImplementedError", "Exception"},
    "IndexError": {"IndexError"},
    "TypeError": {"TypeError"},
    # an int result beyond 64 bits (Python returns a big int there; see classify)
    "IntOverflow": {"OverflowError"},
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
    key: str = ""      # how the driver names it ("module:function" in a project)
    module: str = ""
    shape_plans: list = field(default_factory=list)  # the non-square shapes it was tried on
    timeouts: int = 0  # cases left out because Python didn't finish one in time

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
        for f in self.functions:
            if f.timeouts:
                lines += ["", f"  {f.name}: {f.timeouts} case(s) left out: Python didn't finish a call within "
                              f"{PYTHON_CASE_SECONDS:g} s (it hung), so there is nothing to compare; after a hang on",
                          "  inf / NaN / near-overflow inputs, its other such cases are skipped"]
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
        floats = [c for c in known if c not in ints]
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
        if self.warnings:
            lines += ["", "  Compiler warnings in generated code:"] + [f"    {w}" for w in self.warnings[:10]]
        lines += ["", "  RESULT: " + ("PASS (no mismatches)" if self.ok else f"FAIL ({len(self.mismatches)} mismatches)")]
        return "\n".join(lines)


# --- input generation -------------------------------------------------------------------
_STRINGS = ["", "x"]  # replaced per module by the string literals it contains
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
    if isinstance(t, StructType):
        info, cls = _CLASSES[t.name]
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
_SLOT_CHOICES = {1: ("n", "m"), 2: ("nn", "nm", "mn", "mm")}


def _rank_of(t):
    inner = t.inner if isinstance(t, OptionalType) else t
    return inner.rank if isinstance(inner, ArrayType) else None


def shape_plans(fn, sig, rng, limit=12, max_combos=243):
    """Which non-square shapes the function accepts. Square cases (every Vec of n, every Mat
    n x n) can't catch a transposed dimension, so each array parameter is also tried as length
    n or m (a Vec) and n x n, n x m or m x n (a Mat), with n = 3, m = 4; an assignment is kept
    when Python runs the function on it. Python is the judge of which shapes make sense.
    -> [plan], each plan a tuple with an entry per parameter ("n", "nm", ... or None)"""
    import itertools
    slots = [i for i, p in enumerate(sig.params) if _rank_of(p.type)]
    if not slots:
        return []
    combos = list(itertools.product(*[_SLOT_CHOICES[_rank_of(sig.params[i].type)] for i in slots]))
    combos = [c for c in combos if any(x in ("m", "nm", "mn") for x in c)]  # not square (all n, or all m)
    if len(combos) > max_combos:
        idx = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[i] for i in sorted(idx)]
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
    dims = {"n": n, "m": m}
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
        st = v.bit_generator.state["state"]
        m = (1 << 64) - 1
        return f"{st['state'] >> 64} {st['state'] & m} {st['inc'] >> 64} {st['inc'] & m}"
    if isinstance(t, StructType):
        info, _ = _CLASSES[t.name]
        return " ".join(_encode(getattr(v, f.name), f.type) for f in info.fields)
    if t == STR:
        return "S" + str(v).encode("utf-8").hex()
    if isinstance(t, OptionalType):
        return "N" if v is None else _encode(v, t.inner)
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
    if isinstance(t, StructType):
        if not _is_object(v) or type(v).__name__ != t.name:
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
        elif _is_object(a):
            out.append(copy.deepcopy(a))  # the call may modify it
        else:
            out.append(a)
    return out


PYTHON_CASE_SECONDS = 10.0


def is_timeout(r) -> bool:
    return r[0] == "exc" and r[1] == "Timeout"


def run_python(fn, args, sig, limit=None):
    """Python's result for one call: in the worker process with a time limit of `limit`
    seconds (PYTHON_CASE_SECONDS) while a verification runs (see PythonWorker), else here."""
    limit = PYTHON_CASE_SECONDS if limit is None else limit
    if _WORKER is not None:
        r = _WORKER.call(fn, args, sig, limit)
        if r is not None:
            return r
    return _run_python(fn, args, sig)


def _run_python(fn, args, sig):
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
        spec.loader.exec_module(mod)
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
        key, fname, args, sig = msg
        r = _run_python(_resolve(mods[key], fname), args, sig)
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

    def call(self, fn, args, sig, limit):
        where = self.funcs.get(id(fn))
        if where is None:
            return None
        if not self._alive():
            self._start()
        try:
            self.conn.send((where[0], where[1], args, sig))
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
_PARSE = {FLOAT: "F", INT: "I", BOOL: "B", VEC: "V", MAT: "M", STR: "Str"}
_CTYPE = {FLOAT: "double", INT: "long long", BOOL: "bool"}

_DRIVER_PRELUDE = r"""// Test driver generated by qtrans verify. Reads one call per line:
//   CALL <function> <args...>    -> OK <results...> | EXC <category>
//   BENCH <function> <args...>   -> BENCH <nanoseconds per call> <checksum>
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
static double num(const std::tuple<T...>& t) { return num(std::get<0>(t)); }

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
        }
        catch (const qt::type_error&) { std::printf("EXC TypeError\n"); }
        catch (const qt::int_overflow&) { std::printf("EXC IntOverflow\n"); }@@ARRAY_CATCH@@
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
            if sig.ret.cpp() == "void" and not any(p.mode == "ref" for p in sig.params):
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
        if isinstance(p.type, (OptionalType, StructType)):
            return f"auto p{i} = {_reader(p.type, namespace)};"
        if volatile and p.type in _CTYPE:
            return f"volatile {_CTYPE[p.type]} p{i} = {_PARSE[p.type]}(r);"
        return f"auto p{i} = {_PARSE[p.type]}(r);"
    reads = " ".join(read(i, p, False) for i, p in enumerate(sig.params))
    vreads = " ".join(read(i, p, True) for i, p in enumerate(sig.params))
    args = ", ".join(f"p{i}" for i in range(len(sig.params)))
    opaque = "".join(f"qtrans_opaque(p{i}); " for i, p in enumerate(sig.params) if p.type not in _CTYPE)
    refs = "".join(f" out(p{i});" for i, p in enumerate(sig.params) if p.mode == "ref")
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
                  "    return qt::Rng::from_state(sh, sl, ih, il);", "}"]
    for info, ns in classes:
        lines.append(f"static {ns}::{info.cpp} read_{info.name}(Reader& r);")
    for info, ns in classes:
        c = f"{ns}::{info.cpp}"
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
        first = f"num(o.{info.fields[0].cpp})" if info.fields else "0.0"
        lines.append(f"static double num(const {c}& o) {{ return {first}; }}")
    return "\n".join(lines) + "\n"


class _Toks:
    def __init__(self, parts):
        self.parts, self.i = parts, 0

    def peek(self):
        return self.parts[self.i] if self.i < len(self.parts) else None

    def __next__(self):
        self.i += 1
        return self.parts[self.i - 1]


def _parse_cpp_line(line, out_t):
    parts = line.split()
    if not parts:
        return ("exc", "NoOutput")
    if parts[0] == "EXC":
        return ("exc", parts[1] if len(parts) > 1 else "Exception")
    toks = _Toks(parts[1:])
    values = [_parse_value(t, toks) for t in _flatten(out_t)]
    return ("ok", tuple(values) if isinstance(out_t, TupleType) else values[0])


def _parse_value(t, toks):
    if isinstance(t, StructType):
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
    if t == FLOAT:
        return _decode_float(next(toks))
    if t in (INT, RNG):
        return int(next(toks))  # Python ints: also holds unsigned 64-bit raw outputs
    return next(toks) == "1"


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
    if isinstance(t, StructType):
        info, _ = _CLASSES[t.name]
        worst = (True, 0.0, 0)
        get = (lambda o, k: o[k]) if isinstance(cv, dict) else getattr  # (a nudged Python object on both sides)
        for f in info.fields:
            eq, r, u = _compare_values(getattr(pv, f.name), get(cv, f.name), f.type, rtol, atol)
            worst = (worst[0] and eq, max(worst[1], r), max(worst[2], u))
        return worst
    if t == RNG:  # a generator inside an object: equal if its next raw output is
        if isinstance(pv, np.random.Generator):
            pv = int(copy.deepcopy(pv).bit_generator.random_raw())
        if isinstance(cv, np.random.Generator):  # (a compiled object's generator field, via the bindings)
            cv = int(copy.deepcopy(cv).bit_generator.random_raw())
        return (pv == cv, 0.0 if pv == cv else math.inf, 0)
    if isinstance(t, OptionalType):
        if pv is None or cv is None:
            return (pv is None and cv is None, 0.0 if pv is None and cv is None else math.inf, 0)
        return _compare_values(pv, cv, t.inner, rtol, atol)
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
    elif cpp[0] == "exc" and cpp[1] == "IntOverflow":
        # Python's int grew past 64 bits (or a float became one), where C++ raises
        # OverflowError: the documented 64-bit int difference, never a silently wrong number
        case.outcome = "known"
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
        tree = ast.parse(Path(p).read_text(encoding="utf-8"))
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
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
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


# --- main entry point ----------------------------------------------------------------------------
def verify_file(path, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                compiler=None, build_dir=None, eigen=None, profile="strict") -> VerifyReport:
    """Translate, compile and check one module. The project modules it imports are translated
    and compiled with it (their own functions are checked by verifying them)."""
    rep = verify_project([path], n_random, seed, rtol, atol, bench, compiler, build_dir, eigen, profile=profile)
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
    lits = set()
    for p in project.paths.values():
        lits |= {n.value for n in ast.walk(ast.parse(Path(p).read_text(encoding="utf-8")))
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 80}
    _STRINGS = sorted(lits) + ["", "no such option"]
    _CLASSES = {n: (info, getattr(pymods[m], n)) for n, (info, m) in project.classes().items()}
    _INIT_PARAMS = {n: project.modules[m].signatures[f"{n}.__init__"].params
                    for n, (info, m) in project.classes().items() if info.plain}


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
        tree = ast.parse(Path(p).read_text(encoding="utf-8"))
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
            if sig.ret.cpp() == "void" and not any(p.mode == "ref" for p in sig.params):
                continue
            tol = tolerances.get(name, rtol)
            f_rtol, f_atol = (tol, atol) if isinstance(tol, (int, float)) else tuple(tol)
            fc = FunctionCheck(name, rtol=f_rtol, atol=f_atol)
            fc.key = name if single else f"{module}:{name}"
            fc.module = module
            fn = getattr(pymod, name)
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
            plans = shape_plans(fn, sig, rng)
            fc.shape_plans = plans
            for _ in range(n_random):
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
                c.py = run_python(fn, c.args, sig)
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
    system = []
    if project.uses_arrays:
        shutil.copy(ARRAYS_HEADER, build / "qtrans_arrays.hpp")
        system.append(find_eigen(eigen))
    if project.uses_random:
        shutil.copy(RANDOM_HEADER, build / "qtrans_random.hpp")
    exe = build / ("qtrans_driver.exe" if os.name == "nt" else "qtrans_driver")
    cmd = cxx.command(driver, exe, build, system)
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
                enc = " ".join(_encode(v, p.type) for v, p in zip(c.args, sigs[fc.name].params))
                lines.append(f"CALL {fc.key} {enc}")
                index.append((fc, c))
    run = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=600)
    outputs = run.stdout.splitlines()
    if len(outputs) != len(lines):
        raise RuntimeError(f"C++ driver produced {len(outputs)} results for {len(lines)} cases "
                           f"(exit code {run.returncode}): {run.stderr[-500:]}")
    nudge_rng = np.random.default_rng(seed + 1)
    normwise = lapack_functions(project.paths[targets[0]]) if single else project_lapack_functions(project)
    linalg = linalg_functions(project.paths.values())
    for (fc, c), outline in zip(index, outputs):
        sig = project.modules[fc.module].signatures[fc.name]
        c.cpp = _parse_cpp_line(outline, output_type(sig))
        classify(c, output_type(sig), fc.rtol, fc.atol)
        if c.outcome == "mismatch" and c.py[0] == "ok" and c.cpp[0] == "ok":
            c.sensitivity = python_sensitivity(getattr(pymods[fc.module], fc.name), c, sig, output_type(sig),
                                               nudge_rng, normwise_too=fc.name in normwise)
            if c.rel_err <= 100 * c.sensitivity:
                c.outcome = "illcond"
        if c.outcome == "mismatch" and fc.name in linalg and edge_inputs(c.args):
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
                fn = getattr(pymods[module], fc.name)
                fc.py_ns = _bench_python(fn, _call_args(case.args, sigs[fc.name]))
                enc = " ".join(_encode(v, p.type) for v, p in zip(case.args, sigs[fc.name].params))
                r = subprocess.run([str(exe)], input=f"BENCH {fc.key} {enc}\n", capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                fc.cpp_ns = float(r.stdout.split()[1])

    reports = {}
    for module in targets:
        warns = [w.strip() for w in diagnostics if "warning" in w.lower() and f"{module}.hpp" in w]
        reports[module] = VerifyReport(str(project.paths[module]), project.modules[module], f"{Path(cxx.path).name}",
                                       " ".join(cmd), warns, checks[module], rtol, atol)
    return ProjectVerifyReport(reports)


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
