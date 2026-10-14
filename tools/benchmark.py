"""The benchmark protocol: writes docs/BENCHMARKS.md. Run it to reproduce every number there:

    python tools/benchmark.py [--rounds 7] [--out docs/BENCHMARKS.md] [--no-pin]

What is measured, and how:
  - every function of the 11 example modules, on the input qtrans verify times (a hand-picked
    case, else the random case with the most data), and the workloads of tools/bench_workloads.py
    (a book of 10,000 options priced by a Python loop and NumPy-vectorised, a 100,000-path Monte
    Carlo, 1,000 bonds solved by Newton's method);
  - each function verified first (qtrans verify), its counts reported next to its timings;
  - only already-built code is timed: translating, compiling, building the bindings and importing
    them are one-off costs, measured separately and reported on their own;
  - the arguments are built once, before timing, on both sides (NumPy arrays for Python, C++
    values for C++); each call then runs on them as a caller's would;
  - Python: one warm-up call, then runs of 1, 4, 16, ... calls until one lasts at least 0.1 s;
    that run is timed and divided by its calls. C++ the same, in a driver compiled once (-O2, no
    fast-math, contraction off). The call is in the same translation unit, so the compiler may
    inline it, as it would in a C++ program; the arguments are hidden from the optimiser on every
    call (volatile scalars, an empty asm barrier on the rest) and every part of the result is
    summed into a checksum that is printed, so no call can be hoisted or removed;
  - `rounds` rounds, Python and C++ taking turns; the medians are reported, with the interquartile
    range over the median of the noisier side as the spread;
  - on Linux the process is pinned to one CPU (the C++ driver inherits it), so neither side
    migrates between CPUs mid-run; NumPy's BLAS threads share that CPU, as the C++ is
    single-threaded. The CPU governor is reported when the OS exposes it, and not changed;
  - for the workloads, also the compiled module called from Python (qtrans bind): arguments
    converted in and the result converted back, the cost a Python caller pays.
"""
import argparse
import os
import platform
import shlex
import subprocess
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qtrans.translator import translate_file  # noqa: E402
from qtrans.verify import environment, find_compiler, verify_file  # noqa: E402

EXAMPLES = ["black_scholes", "implied_vol", "futures_options", "bonds", "exotics", "portfolio",
            "lattice_pde", "risk", "monte_carlo", "black_litterman_app", "instruments"]


def shape_of(args):
    """What a timed input is: 'Vec(10000)', 'Mat(50x50)', or the number of scalars."""
    parts, scalars = [], 0
    for a in args:
        if isinstance(a, np.ndarray):
            parts.append(("Vec" if a.ndim == 1 else "Mat") + "(" + "x".join(str(n) for n in a.shape) + ")")
        elif isinstance(a, tuple) and hasattr(a, "_fields"):  # a NamedTuple: its class
            parts.append(type(a).__name__)
        elif isinstance(a, (list, tuple)):
            parts.append(f"list({len(a)})")
        elif isinstance(a, (int, float, bool)):
            scalars += 1
        else:
            parts.append(type(a).__name__)
    if scalars:
        parts.append(f"{scalars} number{'s' if scalars > 1 else ''}")
    return ", ".join(parts)


def _flags(cmd):
    """The compile command without its temporary paths."""
    keep = [w for w in cmd.split(" -o ")[0].split() if not (w.startswith("-I/") or w.endswith(".cpp"))]
    return " ".join(keep)


def spread(rounds):
    """Interquartile range over the median: robust to one slow round, unlike slowest / fastest."""
    if len(rounds) < 2 or statistics.median(rounds) <= 0:
        return 0.0
    q1, _, q3 = statistics.quantiles(rounds, n=4, method="inclusive")
    return (q3 - q1) / statistics.median(rounds)


def pin_cpu():
    """Pin this process (and so the C++ drivers it starts) to one CPU, on Linux."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpu = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    return cpu


def machine(pinned):
    """What the timings ran on, beyond environment(): CPU identity, SIMD, virtualisation, governor."""
    out = []
    info = {}
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if ":" in line:
                k, v = (x.strip() for x in line.split(":", 1))
                info.setdefault(k, v)
    except OSError:
        pass
    if info:
        flags = set(info.get("flags", "").split())
        simd = [f for f in ("avx2", "fma", "avx512f") if f in flags]
        out.append(f"CPU: {info.get('model name', '?')}, family {info.get('cpu family', '?')} model "
                   f"{info.get('model', '?')} stepping {info.get('stepping', '?')}; "
                   f"{os.cpu_count()} logical CPUs; SIMD: {', '.join(simd) or 'none of AVX2 / FMA / AVX-512'}"
                   + ("; a virtual machine (the CPU reports a hypervisor), so the name is what the host exposes"
                      if "hypervisor" in flags else ""))
    else:
        out.append(f"CPU: {platform.processor() or '?'}; {os.cpu_count()} logical CPUs")
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    out.append("CPU governor: " + (gov.read_text().strip() if gov.exists() else "not exposed by the OS")
               + "; frequency scaling not controlled")
    out.append(f"pinned to CPU {pinned}" if pinned is not None else "not pinned to a CPU")
    return out


def one_off_costs(path, wrep, work):
    """What is paid once, before any call: translate, compile the driver, build and import the module."""
    t0 = time.perf_counter()
    translate_file(path, work / "translate_once")
    t_translate = time.perf_counter() - t0
    cmd = shlex.split(wrep.compile_command)
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, cwd=work / path.stem, check=True)
    t_compile = time.perf_counter() - t0
    return t_translate, t_compile


def fmt_ns(ns):
    if ns >= 1e6:
        return f"{ns / 1e6:,.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:,.1f} µs"
    return f"{ns:,.0f} ns"


def counts(fc):
    return "/".join(str(fc.count(k)) for k in ("match", "illcond", "known", "mismatch"))


def bench_module(path, rounds, n_random, work):
    rep = verify_file(path, n_random=n_random, bench=rounds, build_dir=work / path.stem)
    return rep


def binding_times(rounds, work):
    """The workloads through the compiled module, from Python: conversions included."""
    from qtrans.bind import build_extension
    import importlib.util
    path = ROOT / "tools" / "bench_workloads.py"
    ext = build_extension(path, out_dir=work / "ext", build_dir=work / "ext_build")
    t0 = time.perf_counter()
    m = ext.load()
    t_import = time.perf_counter() - t0
    spec = importlib.util.spec_from_file_location("bench_workloads_py", path)
    py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(py)
    out = {}
    for name, cases in py.QTRANS_CASES.items():
        args = cases[0]
        f = getattr(m, name)
        times = []
        for _ in range(rounds):
            f(*args)  # warm-up
            reps = 1
            while True:
                t0 = time.perf_counter()
                for _ in range(reps):
                    f(*args)
                el = time.perf_counter() - t0
                if el > 0.1:
                    break
                reps *= 4
            times.append(el * 1e9 / reps)
        out[name] = statistics.median(times)
    return out, ext, t_import


def marker(fc):
    """A row whose cases are not all plain matches is marked, so no reader takes it for one."""
    return ("†" if fc.count("known") else "") + ("‡" if fc.count("illcond") else "")


def fmt_s(sec):
    return f"{sec * 1e3:,.0f} ms" if sec < 1 else f"{sec:,.1f} s"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "docs" / "BENCHMARKS.md"))
    ap.add_argument("--no-pin", action="store_true", help="don't pin the process to one CPU")
    a = ap.parse_args(argv)
    pinned = None if a.no_pin else pin_cpu()
    work = Path(tempfile.mkdtemp(prefix="qtrans_bench_"))
    cxx = find_compiler()

    import importlib.util
    wpath = ROOT / "tools" / "bench_workloads.py"
    spec = importlib.util.spec_from_file_location("bench_workloads_meta", wpath)
    meta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meta)
    wrep = bench_module(wpath, a.rounds, a.n_random, work)
    btimes, ext, t_import = binding_times(a.rounds, work)
    t_translate, t_compile = one_off_costs(wpath, wrep, work)
    reps = [(m, bench_module(ROOT / "examples" / f"{m}.py", a.rounds, a.n_random, work)) for m in EXAMPLES]
    every = list(wrep.functions) + [fc for _, r in reps for fc in r.functions]
    timed = [fc for fc in every if fc.py_ns and fc.cpp_ns]
    allreps = [n for fc in timed for n in fc.py_reps + fc.cpp_reps]

    L = ["# Benchmarks", "",
         "Generated by `tools/benchmark.py`: run it to reproduce every number here. **All benchmarks measure "
         "already-built code; every function is verified before timing, and the reported verification counts "
         "are shown beside its result.** A speed-up is never quoted for a translation that differs from Python.", "",
         "@HEADLINE@", "",
         "## What is timed", "",
         "- **Per-call time of already-built code only.** Translating, compiling, building the bindings and "
         "importing them are not in any timing; they are paid once and are listed under *One-off costs*.",
         "- **Inputs built beforehand.** The arguments are made once before timing (NumPy arrays for Python, C++ "
         "values for C++); the timed calls run on them.",
         "- **Warm-up.** One untimed call, then runs of 1, 4, 16, ... calls until one lasts at least 0.1 s; only "
         "that run is timed, and divided by its number of calls. The shorter runs before it warm up too. "
         f"Calls timed per round, over every function here: {min(allreps):,} to {max(allreps):,}.",
         "- **No call can be optimised away.** The C++ call is in the same translation unit, so the compiler may "
         "inline it, as in any C++ program that includes the header. The arguments are hidden from the optimiser on "
         "every call (volatile scalars, an empty asm barrier on arrays and objects) and every part of the result "
         "goes into a checksum that is printed. A C++ time of a few nanoseconds is real: what is left once "
         "Python's call is gone is a handful of instructions.",
         f"- **Rounds.** {a.rounds} rounds per function, Python and C++ taking turns; the **median** is reported. "
         "*Spread* is the interquartile range over the median, of the noisier side: robust to one slow round. "
         "This runs on a shared virtual machine, and a spread of tens of percent is the machine, not the code; "
         "it is why medians of several rounds are reported.", "",
         "**Verified (m/i/k/x)** = matches / ill-conditioned (Python's own answer moves by as much under a "
         "4-ULP nudge of the inputs, so the case can't tell) / known differences (documented in the README, each "
         "where it can apply) / mismatches, over the function's hand-picked and random cases. Rows marked **†** "
         "have known differences, rows marked **‡** ill-conditioned cases: their timings stand, but not every "
         "case of theirs is a plain match.", "",
         "## Machine and toolchain", "",
         f"- {environment(cxx)}",
         *[f"- {x}" for x in machine(pinned)],
         f"- compiled as: `{_flags(wrep.compile_command)}` (`QTRANS_LINALG_AUDIT`: the backward-stability "
         "checks of linear algebra stay on while timed, as in every verification)", "",
         "## One-off costs", "",
         "Paid once, before the first call, for `tools/bench_workloads.py` (8 functions); not in any timing below.", "",
         "| Step | Time |", "|---|---:|",
         f"| translate (Python to C++) | {fmt_s(t_translate)} |",
         f"| compile the C++ (the verify driver, with the header) | {fmt_s(t_compile)} |",
         f"| build the Python module (`qtrans bind`) | {fmt_s(ext.seconds)} |",
         f"| import the built module in Python | {fmt_s(t_import)} |", "",
         "## Workloads", "",
         "The same book priced three ways answers whether the Python baseline already uses NumPy. "
         "\"Compiled, from Python\" is the `qtrans bind` module called from Python on the same "
         "arguments: NumPy arrays converted in, the result converted back.", "",
         "| Function | Workload | Python | C++ | Speed-up | Compiled, from Python | Spread | Calls timed (Py / C++) | Verified (m/i/k/x) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for fc in wrep.functions:
        sp = max(spread(fc.py_rounds), spread(fc.cpp_rounds))
        cr = f"{int(statistics.median(fc.py_reps)):,} / {int(statistics.median(fc.cpp_reps)):,}"
        L.append(f"| `{fc.name}`{marker(fc)} | {meta.WORKLOADS.get(fc.name, '')} | {fmt_ns(fc.py_ns)} | "
                 f"{fmt_ns(fc.cpp_ns)} | {fc.py_ns / fc.cpp_ns:,.1f}× | {fmt_ns(btimes[fc.name])} | {sp:.0%} | "
                 f"{cr} | {counts(fc)} |")
    by = {fc.name: fc for fc in wrep.functions}
    L += ["", "What this shows:", ""]
    L.append(f"- Against a plain Python loop the C++ is {by['book_loop'].py_ns / by['book_loop'].cpp_ns:.0f}× faster "
             f"(`book_loop`) and {by['ytm_book'].py_ns / by['ytm_book'].cpp_ns:.0f}× (`ytm_book`). Against NumPy-"
             f"vectorised Python the gain is {by['book_numpy'].py_ns / by['book_numpy'].cpp_ns:.1f}× (`book_numpy`): "
             "NumPy already runs its loops in C. Replacing a Python loop with compiled code is not the same as "
             "beating well-vectorised NumPy by that much; the compiled loop is "
             f"{by['book_numpy'].py_ns / by['book_loop'].cpp_ns:.1f}× faster than the vectorised Python.")
    L.append(f"- `mc_call` is {'slower' if by['mc_call'].cpp_ns > by['mc_call'].py_ns else 'faster'} in C++ "
             f"({fmt_ns(by['mc_call'].cpp_ns)} against {fmt_ns(by['mc_call'].py_ns)}). Taken apart, the draws "
             f"are on a par ({fmt_ns(by['mc_draws'].cpp_ns)} against {fmt_ns(by['mc_draws'].py_ns)}), and `np.exp` "
             f"is not ({fmt_ns(by['mc_exp'].cpp_ns)} against {fmt_ns(by['mc_exp'].py_ns)}): NumPy has its own "
             "SIMD exp (AVX-512 on this CPU), and the C++ calls the C library's exp per element, so it agrees "
             "with Python's math.exp and only to rounding with NumPy's. Porting NumPy's exp would make it both "
             "faster and bit-identical on such CPUs; it is on the roadmap.")
    bs = by["bs_call"]
    big = ("book_loop", "book_numpy", "ytm_book")
    worst = max(abs(btimes[n] / by[n].cpp_ns - 1.0) for n in big)
    L.append(f"- Called from Python through `qtrans bind`, one scalar `bs_call` takes {fmt_ns(btimes['bs_call'])} "
             f"against {fmt_ns(bs.cpp_ns)} in C++: the difference is Python's call and the conversions, and it is "
             f"still {bs.py_ns / btimes['bs_call']:.0f}× faster than the Python function. On the books of options "
             f"and bonds the bound module is within {worst:.0%} of the C++ alone: converting 10,000-element arrays "
             "is small next to the work. The bindings cost most, in proportion, on the shortest calls.")
    L += ["", "## Every example function", "",
          "One call on the input qtrans verify times. Scalar functions gain the most, because Python's per-call "
          "overhead dominates them (and for `scipy.stats.norm`, SciPy's own per-call overhead); functions that "
          "spend their time inside NumPy gain far less, and some are slower.", "",
          "| Module | Function | Input | Python | C++ | Speed-up | Spread | Verified (m/i/k/x) |",
          "|---|---|---|---:|---:|---:|---:|---|"]
    ups = []
    for m, rep in reps:
        for fc in rep.functions:
            if not fc.py_ns or not fc.cpp_ns:
                continue
            up = fc.py_ns / fc.cpp_ns
            ups.append(up)
            sp = max(spread(fc.py_rounds), spread(fc.cpp_rounds))
            inp = shape_of(fc.bench_case.args) if fc.bench_case is not None else ""
            L.append(f"| {m} | `{fc.name}`{marker(fc)} | {inp} | {fmt_ns(fc.py_ns)} | {fmt_ns(fc.cpp_ns)} | "
                     f"{up:,.1f}× | {sp:.0%} | {counts(fc)} |")
    ups.sort()
    slower = sum(1 for u in ups if u < 1.0)
    L += ["", "† known differences in some cases; ‡ ill-conditioned cases (see *Verified* above).", "",
          f"{len(ups)} functions: speed-up median {statistics.median(ups):.0f}×, from {ups[0]:.1f}× to "
          f"{ups[-1]:,.0f}×; {slower} {'was' if slower == 1 else 'were'} slower in C++ than in Python.", ""]
    head = (f"**Across the {len(ups)} verified example functions, on the machine below, standalone generated C++ had "
            f"a median speed-up of {statistics.median(ups):.0f}× over the Python; results ranged from {ups[0]:.1f}× "
            f"to {ups[-1]:,.0f}×, depending on the workload and the baseline.** Quote it that way: qtrans is not "
            f"\"{statistics.median(ups):.0f}× faster\" in itself. The speed-up depends on the function, the input "
            "size, the libraries it calls, the hardware, and whether the Python baseline is a plain loop or "
            "vectorised NumPy (see *Workloads*).")
    L[L.index("@HEADLINE@")] = head
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
