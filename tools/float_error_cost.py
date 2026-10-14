"""What Python's float errors cost in the C++ (1.0's default), measured.

The same example modules are translated twice: as they are (a float division by zero, an
overflow, a math domain error raise as in Python) and with `# qtrans: ieee` at the top (IEEE-754's
inf and NaN, no checks: what qtrans 0.14 wrote outside a `try`). Both are compiled with the same
compiler and flags, and the same nine scalar pricing functions are timed on the same inputs,
the two programs taking turns, best of several rounds (the machine's load moves both the same
way, and the best round is the one least disturbed). Run:

    python tools/float_error_cost.py [rounds]

The README and CHANGELOG quote its output (GCC 13 and Clang 18, x86-64).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qtrans.translator import translate_source  # noqa: E402
from qtrans.verify import environment, find_compiler, find_eigen  # noqa: E402

MODULES = ["black_scholes", "bonds", "futures_options", "implied_vol"]

BENCH = r"""
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "black_scholes.hpp"
#include "bonds.hpp"
#include "futures_options.hpp"
#include "implied_vol.hpp"
template <class F> double best_ns(F f, int n, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0;
        for (int i = 0; i < n; ++i) sink = sink + f(i);
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n;
        if (ns < best) best = ns;
    }
    return best;
}
int main() {
    const int N = 4096;
    std::mt19937_64 g(1);
    std::uniform_real_distribution<double> u(0.5, 1.5);
    std::vector<double> S(N), K(N), sg(N), T(N), y(N), P(N);
    for (int i = 0; i < N; ++i) { S[i] = 100 * u(g); K[i] = 100 * u(g); sg[i] = 0.3 * u(g); T[i] = 2 * u(g); y[i] = 0.05 * u(g); }
    for (int i = 0; i < N; ++i) P[i] = implied_vol::bs_price(S[i], K[i], 0.02, sg[i], T[i], true);
    const int n = 200000, reps = 15;
    std::printf("d1 %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return black_scholes::d1(S[j], K[j], 0.02, sg[j], T[j], 0.01); }, n, reps));
    std::printf("bs_price %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return black_scholes::bs_price(S[j], K[j], 0.02, sg[j], T[j], 0.01, true); }, n, reps));
    std::printf("delta %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return black_scholes::delta(S[j], K[j], 0.02, sg[j], T[j], 0.01, true); }, n, reps));
    std::printf("vega %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return black_scholes::vega(S[j], K[j], 0.02, sg[j], T[j], 0.01); }, n, reps));
    std::printf("black76 %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return futures_options::black76(S[j], K[j], 0.02, sg[j], T[j], true); }, n, reps));
    std::printf("bond_price %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return bonds::bond_price(100.0, 0.05, y[j], 10, 2); }, n, reps));
    std::printf("macaulay_duration %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return bonds::macaulay_duration(100.0, 0.05, y[j], 10, 2); }, n, reps));
    std::printf("ytm_newton %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return bonds::ytm_newton(90.0 + 20.0 * (S[j] / 150.0), 100.0, 0.05, 10, 2); }, n / 10, reps));
    std::printf("implied_vol %.3f\n", best_ns([&](int i) { int j = i & (N - 1); return implied_vol::implied_vol(P[j], S[j], K[j], 0.02, T[j], true); }, n / 10, reps));
}
"""


def build(work, ieee, cxx):
    d = work / ("ieee" if ieee else "checked")
    d.mkdir(exist_ok=True)
    for m in MODULES:
        src = (ROOT / "examples" / f"{m}.py").read_text(encoding="utf-8")
        r = translate_source(("# qtrans: ieee\n" if ieee else "") + src, m, f"{m}.py")
        (d / f"{m}.hpp").write_text(r.header, encoding="utf-8")
    for h in (ROOT / "qtrans" / "runtime").glob("*.hpp"):
        shutil.copy(h, d / h.name)
    (d / "bench.cpp").write_text(BENCH, encoding="utf-8")
    exe = d / ("bench.exe" if sys.platform == "win32" else "bench")
    subprocess.run(cxx.command(d / "bench.cpp", exe, d, [find_eigen()]), check=True, capture_output=True)
    return exe


def main(rounds=7):
    cxx = find_compiler()
    work = Path(tempfile.mkdtemp(prefix="qtrans_cost_"))
    exes = {False: build(work, False, cxx), True: build(work, True, cxx)}
    best = {False: {}, True: {}}
    for _ in range(rounds):
        for ieee in (True, False):  # taking turns
            out = subprocess.run([str(exes[ieee])], capture_output=True, text=True, check=True).stdout
            for line in out.splitlines():
                name, ns = line.split()
                best[ieee][name] = min(best[ieee].get(name, 1e300), float(ns))
    print(f"{environment(cxx)}\nbest of {rounds} rounds, ns per call\n")
    print("| function | IEEE-754 (# qtrans: ieee) | Python's errors (default) | cost |")
    print("|---|---:|---:|---:|")
    for name, a in best[True].items():
        b = best[False][name]
        print(f"| `{name}` | {a:,.1f} | {b:,.1f} | {100 * (b / a - 1):+.1f}% |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
