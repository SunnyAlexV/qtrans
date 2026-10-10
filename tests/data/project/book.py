"""Three levels: book -> core -> rates."""
import numpy as np

from core import fwd, make, weights
from qtrans.typing import Vec
from rates import Option, Portfolio


def book(S: float, w: Vec) -> float:
    o = make(S)
    return fwd(o.S, o.K, o.T) + weights(w)


def strike_of(o: Option) -> float:
    return o.K


def fresh(n: int) -> Portfolio:
    return Portfolio(np.ones(n) / n)
