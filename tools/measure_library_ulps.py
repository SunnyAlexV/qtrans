"""Why Python and C++ can't agree to the last bit when NumPy or SciPy are involved.

Measures, inside Python alone:
  1. scipy.stats.norm.cdf (SciPy's own erf/erfc) vs the same formula built on math.erf /
     math.erfc, which is what C++ std::erf / std::erfc compute (both call the C library).
  2. numpy.exp vs math.exp on the same inputs (NumPy ships its own vectorised exp).
  3. The Python cost of one scalar norm.cdf call, three ways.

The README quotes these numbers; run this to reproduce them on your machine.
"""
import math
import struct

import numpy as np
from scipy.special import ndtr


def ulp_key(x: float) -> int:
    i = struct.unpack("<q", struct.pack("<d", x))[0]
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)


def cdf_via_libm(a: float) -> float:
    """SciPy's ndtr algorithm, but on the C library's erf/erfc (= qt::norm_cdf in C++)."""
    x = a * 0.7071067811865476
    z = abs(x)
    if z < 0.7071067811865476:
        return 0.5 + 0.5 * math.erf(x)
    y = 0.5 * math.erfc(z)
    return 1.0 - y if x > 0 else y


def main(n=200_000, seed=1):
    xs = np.random.default_rng(seed).normal(0.0, 3.0, n)
    d = np.array([abs(ulp_key(cdf_via_libm(float(x))) - ulp_key(float(ndtr(x)))) for x in xs])
    print(f"norm.cdf, {n:,} points ~ N(0, 3^2):")
    print(f"  identical to the bit: {np.mean(d == 0):.1%}   median ULP {np.median(d):.0f}   max ULP {d.max()}")
    same = np.mean([float(np.exp(x)) == math.exp(float(x)) for x in xs[:50_000]])
    print(f"numpy.exp vs math.exp, 50,000 points: identical {same:.1%}, differ {1 - same:.1%}")

    # Cost of one scalar call: this is why functions that call scipy.stats.norm show
    # 1,000x+ speed-ups in RESULTS.md: most of the Python time is SciPy's per-call overhead.
    import timeit
    from scipy.stats import norm
    for label, stmt in [("scipy.stats.norm.cdf(x)", lambda: norm.cdf(0.3)),
                        ("scipy.special.ndtr(x)", lambda: ndtr(0.3)),
                        ("cdf_via_libm(x)  (pure Python)", lambda: cdf_via_libm(0.3))]:
        n_calls = 20_000
        t = min(timeit.repeat(stmt, number=n_calls, repeat=5)) / n_calls
        print(f"  {label:<34} {t * 1e9:>9,.0f} ns per call")


if __name__ == "__main__":
    main()
