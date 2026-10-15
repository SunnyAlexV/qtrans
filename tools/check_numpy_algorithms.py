"""Pure-Python ports of the NumPy algorithms that qtrans_arrays.hpp reimplements, checked
bit-for-bit against NumPy itself on random data. If NumPy ever changes an algorithm, this
script is the first thing that will say so.

    python tools/check_numpy_algorithms.py
"""
import math

import numpy as np

rng = np.random.default_rng(7)


def pairwise(a, lo, n):
    """numpy/_core/src/umath/loops_utils.h.src: pairwise_sum, PW_BLOCKSIZE = 128."""
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[lo + i]
        return r
    if n <= 128:
        r = [a[lo + i] for i in range(8)]
        i = 8
        while i < n - (n % 8):
            for k in range(8):
                r[k] += a[lo + i + k]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2)


NUMPY = tuple(int(x) for x in np.__version__.split(".")[:2])


def reduce_sum(a, n):
    """What np.sum does with pairwise_sum: before NumPy 2.3, through the iterator's 8192-element
    buffer (each chunk pairwise, the chunk sums added in turn from 0.0); from 2.3, one pairwise sum."""
    if NUMPY >= (2, 3) or n <= 8192:
        return pairwise(a, 0, n)
    acc = 0.0
    for lo in range(0, n, 8192):
        acc = acc + pairwise(a, lo, min(8192, n - lo))
    return acc


def quantile(x, q):
    """numpy.quantile, method='linear': index (n - 1) * q, then _lerp."""
    s = sorted(x)
    n = len(s)
    vi = (n - 1) * q
    lo, hi = (n - 1, n - 1) if vi >= n - 1 else (math.floor(vi), math.floor(vi) + 1)
    g = vi - math.floor(vi)
    a, b = s[lo], s[hi]
    d = b - a
    return b - d * (1 - g) if g >= 0.5 else a + d * g


def report(name, ok):
    print(f"  {name:<44} bit-identical {sum(ok):>5}/{len(ok)}")
    return all(ok)


def main():
    results = []
    ok = []
    for n in list(range(0, 300)) + [1000, 4097, 10000, 20001]:
        x = rng.normal(size=n) * 10 ** rng.uniform(-3, 3, size=n)
        ok.append(reduce_sum(list(x), n) == float(np.sum(x)))
    results.append(report("np.sum, 1-D (pairwise)", ok))

    full, ax1, ax0 = [], [], []
    for _ in range(500):
        r, c = (int(v) for v in rng.integers(1, 40, size=2))
        A = rng.normal(size=(r, c)) * 10 ** rng.uniform(-3, 3, size=(r, c))
        full.append(reduce_sum(list(A.ravel()), r * c) == float(A.sum()))
        ax1.append(all(reduce_sum(list(A[i]), c) == A.sum(axis=1)[i] for i in range(r)))
        if c == 1:
            col = np.array([reduce_sum(list(A[:, 0]), r)])   # one column: contiguous, so pairwise
        else:
            col = A[0].copy()
            for i in range(1, r):
                col = col + A[i]                               # otherwise row after row
        ax0.append(np.array_equal(col, A.sum(axis=0)))
    results.append(report("np.sum, 2-D (pairwise over C order)", full))
    results.append(report("np.sum(axis=1) (pairwise per row)", ax1))
    results.append(report("np.sum(axis=0) (rows in sequence)", ax0))

    m_ok, v_ok = [], []
    for _ in range(300):
        n = int(rng.integers(2, 500))
        x = rng.normal(size=n) * 10 ** rng.uniform(-3, 3)
        m = reduce_sum(list(x), n) / n
        m_ok.append(m == float(np.mean(x)))
        dev = [(v - m) * (v - m) for v in x]
        for ddof in (0, 1):
            var = reduce_sum(dev, n) / (n - ddof)
            v_ok.append(var == float(np.var(x, ddof=ddof)) and math.sqrt(var) == float(np.std(x, ddof=ddof)))
    results.append(report("np.mean = pairwise sum / n", m_ok))
    results.append(report("np.var / np.std (ddof 0 and 1)", v_ok))

    ok = []
    for _ in range(5000):
        n = int(rng.integers(1, 300))
        x = rng.normal(size=n)
        q = float(rng.choice([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0, rng.uniform()]))
        ok.append(quantile(list(x), q) == float(np.quantile(x, q)))
    results.append(report("np.quantile (linear)", ok))

    ok = []
    for _ in range(300):
        a, b = rng.normal(size=2) * 100
        n = int(rng.integers(2, 200))
        step = (b - a) / (n - 1)
        y = [i * step + a for i in range(n)]
        y[-1] = b
        ok.append(np.array_equal(y, np.linspace(a, b, n)))
    results.append(report("np.linspace (i * step + start, last = stop)", ok))

    ok = []
    for _ in range(300):
        start, step = float(rng.normal() * 10), float(rng.choice([0.1, 0.25, 1.0, -0.3, 1e-3]))
        stop = start + step * float(rng.integers(1, 400)) * float(rng.uniform(0.9, 1.1))
        n = max(int(math.ceil((stop - start) / step)), 0)
        y = [start, start + step][:n]
        delta = (start + step) - start
        y += [start + i * delta for i in range(2, n)]
        ok.append(np.array_equal(y, np.arange(start, stop, step)))
    results.append(report("np.arange (start + i * (second - first))", ok))

    ok = []
    for _ in range(200):
        x = rng.normal(size=int(rng.integers(1, 300)))
        s, p, cs, cp = 0.0, 1.0, [], []
        for v in x:
            s += v
            p *= v
            cs.append(s)
            cp.append(p)
        ok.append(np.array_equal(cs, np.cumsum(x)) and np.array_equal(cp, np.cumprod(x)))
    results.append(report("np.cumsum / np.cumprod (left to right)", ok))

    x = rng.uniform(0, 10, size=100000)
    fast = [np.array_equal(x ** 0.5, np.sqrt(x)), np.array_equal(x ** 2, x * x), np.array_equal(x ** -1, 1 / x)]
    results.append(report("x**0.5, x**2, x**-1 fast paths", fast))

    print("ALL BIT-IDENTICAL" if all(results) else "SOME ALGORITHMS DIFFER: update qtrans_arrays.hpp")


if __name__ == "__main__":
    main()
