"""Uses rates through every kind of import."""
import math

import numpy as np

import rates as R
from qtrans.typing import Vec
from rates import *  # noqa: F401,F403


def annual(x: float) -> float:
    return x * DAYS


def pv(cf: float, t: float, rf: float = DEFAULT_RF) -> float:
    return cf * discount(rf, t)


def pv2(cf: float, t: float) -> float:
    return cf * R.discount(R.DEFAULT_RF, t)


def fwd(S: float, K: float, T: float) -> float:
    return Option(S, K, T).forward() - R.Option(S, K, T, 0.0).forward()


def weights(w: Vec) -> float:
    normalise(w)
    return float(np.sum(w))


def grow(p: Portfolio, px: Vec, k: float) -> float:
    p.scale(k)
    return p.value(px)


def draws(rng: np.random.Generator, n: int) -> float:
    return float(np.mean(noisy(rng, n)))


def uses_bad(x: float) -> float:
    return bad(x) + 1.0


def uses_table(x: float) -> float:
    return x + R.TABLE


def guarded(x: float) -> float:
    try:
        return safe_log(x)
    except ValueError:
        return 0.0


def make(S: float) -> Option:
    return Option(S, S, 1.0)


def _private(x: float) -> float:
    return x
