import numpy as np
from qtrans.typing import Vec, IVec


def bootstrap_means(x: Vec, b: int, seed: int) -> Vec:
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.zeros(b)
    for k in range(b):
        idx = rng.integers(0, n, size=n)
        out[k] = x[idx].mean()
    return out


def ranks_sum(n: int) -> int:
    k = np.arange(1, n + 1)
    return int((k * k).sum())


def weights(n: int, lam: float) -> Vec:
    j = np.arange(n - 1, -1, -1)
    w = lam ** j
    return w / w.sum()


def bucket(x: Vec, width: float) -> IVec:
    return np.floor(x / width).astype(int)


def count_in(idx: IVec, lo: int, hi: int) -> int:
    c = 0
    for i in idx:
        if lo <= i < hi:
            c += 1
    return c


def mod_pattern(n: int, m: int) -> IVec:
    a = np.arange(-n, n)
    return a // m + a % m - 3 * a


rng = np.random.default_rng(42)
data = rng.standard_normal(20)
print(bootstrap_means(data, 4, 7))
print(ranks_sum(10), weights(4, 0.5))
idx = rng.integers(0, 10, size=12)
print(idx, idx.sum(), idx.max(), len(idx), idx[3], idx[-1])
print(bucket(data[:6], 0.5), count_in(idx, 2, 7))
print(mod_pattern(4, 3), np.arange(5) * 0.5, np.arange(3) / 2)
print(repr(np.arange(3)), repr(idx[0]), np.full(3, 7), np.array([1, 2, 30]))
