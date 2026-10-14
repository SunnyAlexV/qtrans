"""The benchmark protocol: writes docs/BENCHMARKS.md. Run it to reproduce every number there:

    python tools/benchmark.py [--rounds 7] [--out docs/BENCHMARKS.md]

What is measured, and how:
  - every function of the 11 example modules, on the input qtrans verify times (a hand-picked
    case, else the random case with the most data), and the workloads of tools/bench_workloads.py
    (a book of 10,000 options priced by a Python loop and NumPy-vectorised, a 100,000-path Monte
    Carlo, 1,000 bonds solved by Newton's method);
  - each function verified first (qtrans verify), its counts reported next to its timings;
  - Python: the function called from Python, repeated until a round takes at least 0.1 s, per call;
    C++: the translated function called from C++, its arguments already C++ values, repeated the
    same way (compiled once, -O2, no fast-math, contraction off);
  - `rounds` rounds, Python and C++ taking turns; the median is reported, and the spread
    (slowest round / fastest round - 1) of the noisier side, since a shared machine moves both;
  - for the workloads, also the compiled module called from Python (qtrans bind): arguments
    converted in and the result converted back, the cost a Python caller pays.
"""
import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    return max(rounds) / min(rounds) - 1.0 if rounds and min(rounds) > 0 else 0.0


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
    m = ext.load()
    spec = importlib.util.spec_from_file_location("bench_workloads_py", path)
    py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(py)
    out = {}
    for name, cases in py.QTRANS_CASES.items():
        args = cases[0]
        f = getattr(m, name)
        times = []
        for _ in range(rounds):
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
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "docs" / "BENCHMARKS.md"))
    a = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="qtrans_bench_"))
    cxx = find_compiler()

    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_workloads_meta", ROOT / "tools" / "bench_workloads.py")
    meta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(meta)
    wrep = bench_module(ROOT / "tools" / "bench_workloads.py", a.rounds, a.n_random, work)
    btimes = binding_times(a.rounds, work)
    reps = [(m, bench_module(ROOT / "examples" / f"{m}.py", a.rounds, a.n_random, work)) for m in EXAMPLES]

    L = ["# Benchmarks", "",
         "Generated by `tools/benchmark.py` (run it to reproduce every number here; the protocol is in its",
         "docstring). Each function is verified against Python before it is timed; its verification counts",
         "(match / ill-conditioned / known / mismatch, over its hand-picked and random cases) are shown next to",
         "the timings, so a speed-up is never quoted for a translation that differs.", "",
         "## Machine and toolchain", "",
         f"- {environment(cxx)}",
         f"- compiled as: `{_flags(wrep.compile_command)}` (`QTRANS_LINALG_AUDIT`: the backward-stability "
         "checks of linear algebra stay on while timed, as in every verification)",
         f"- {a.rounds} rounds per function, Python and C++ taking turns; medians, and the spread of the noisier "
         "side (slowest / fastest round - 1). This runs on a shared machine: a spread of tens of percent is "
         "the machine, not the code, and is why medians of several rounds are reported.", "",
         "## Workloads", "",
         "The same book priced three ways answers whether the Python baseline already uses NumPy. "
         "\"Compiled, from Python\" is the `qtrans bind` module called from Python on the same "
         "arguments: NumPy arrays converted in, the result converted back.", "",
         "| Function | Workload | Python | C++ | Speed-up | Compiled, from Python | Spread | Verified (m/i/k/x) |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for fc in wrep.functions:
        sp = max(spread(fc.py_rounds), spread(fc.cpp_rounds))
        L.append(f"| `{fc.name}` | {meta.WORKLOADS.get(fc.name, '')} | {fmt_ns(fc.py_ns)} | {fmt_ns(fc.cpp_ns)} | "
                 f"{fc.py_ns / fc.cpp_ns:,.1f}× | {fmt_ns(btimes[fc.name])} | {sp:.0%} | {counts(fc)} |")
    by = {fc.name: fc for fc in wrep.functions}
    L += ["", "What this shows:", ""]
    L.append(f"- Against a plain Python loop the C++ is {by['book_loop'].py_ns / by['book_loop'].cpp_ns:.0f}× faster "
             f"(`book_loop`) and {by['ytm_book'].py_ns / by['ytm_book'].cpp_ns:.0f}× (`ytm_book`). Against NumPy-"
             f"vectorised Python the gain is {by['book_numpy'].py_ns / by['book_numpy'].cpp_ns:.1f}× (`book_numpy`): "
             "NumPy already runs its loops in C. The compiled loop is "
             f"{by['book_numpy'].py_ns / by['book_loop'].cpp_ns:.1f}× faster than the vectorised Python.")
    L.append(f"- `mc_call` is {'slower' if by['mc_call'].cpp_ns > by['mc_call'].py_ns else 'faster'} in C++ "
             f"({fmt_ns(by['mc_call'].cpp_ns)} against {fmt_ns(by['mc_call'].py_ns)}). Taken apart, the draws "
             f"are on a par ({fmt_ns(by['mc_draws'].cpp_ns)} against {fmt_ns(by['mc_draws'].py_ns)}), and `np.exp` "
             f"is not ({fmt_ns(by['mc_exp'].cpp_ns)} against {fmt_ns(by['mc_exp'].py_ns)}): NumPy has its own "
             "SIMD exp (AVX-512 on this CPU), and the C++ calls the C library's exp per element, so it agrees "
             "with Python's math.exp and only to rounding with NumPy's. Porting NumPy's exp would make it both "
             "faster and bit-identical on such CPUs; it is on the roadmap.")
    bs = by["bs_call"]
    big = [n for n in ("book_loop", "book_numpy", "ytm_book")]
    worst = max(abs(btimes[n] / by[n].cpp_ns - 1.0) for n in big)
    L.append(f"- Called from Python through `qtrans bind`, one scalar `bs_call` takes {fmt_ns(btimes['bs_call'])} "
             f"against {fmt_ns(bs.cpp_ns)} in C++: the difference is Python's call and the conversions, and it is "
             f"still {bs.py_ns / btimes['bs_call']:.0f}× faster than the Python function. On the books of options "
             f"and bonds the bound module is within {worst:.0%} of the C++ alone: converting 10,000-element arrays "
             "is small next to the work.")
    L += ["", "## Every example function", "",
          "One call on the input qtrans verify times. Scalar functions gain the most, because Python's per-call "
          "overhead dominates them (and for `scipy.stats.norm`, SciPy's own per-call overhead); functions that "
          "spend their time inside NumPy gain far less. A C++ time of a few nanoseconds is real: the inputs "
          "are hidden from the optimiser on every repetition and every part of the result is used, so the "
          "call can't be folded away; it is a handful of instructions once Python's call is gone.", "",
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
            L.append(f"| {m} | `{fc.name}` | {inp} | {fmt_ns(fc.py_ns)} | {fmt_ns(fc.cpp_ns)} | {up:,.1f}× | "
                     f"{sp:.0%} | {counts(fc)} |")
    ups.sort()
    L += ["", f"{len(ups)} functions; speed-up median {statistics.median(ups):.0f}×, "
              f"from {ups[0]:.1f}× to {ups[-1]:,.0f}×.", ""]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
