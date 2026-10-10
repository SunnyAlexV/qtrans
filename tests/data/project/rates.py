"""Rates and instruments: the bottom of the project."""
import math
from dataclasses import dataclass

import numpy as np

from qtrans.typing import Vec

DAYS = 252
DEFAULT_RF: float = 0.04
TABLE = {"a": 1}


def discount(r: float, t: float) -> float:
    return math.exp(-r * t)


def safe_log(x: float) -> float:
    return math.log(x)


def normalise(w: Vec) -> None:
    w /= np.sum(w)


def first(w: Vec) -> Vec:
    return w


def noisy(rng: np.random.Generator, n: int) -> Vec:
    return rng.standard_normal(n)


def bad(x: float) -> float:
    return TABLE["a"] * x


@dataclass(frozen=True)
class Option:
    S: float
    K: float
    T: float
    r: float = DEFAULT_RF

    def forward(self) -> float:
        return self.S / discount(self.r, self.T)


@dataclass
class Portfolio:
    w: Vec
    cash: float = 0.0

    def scale(self, k: float) -> None:
        self.w *= k

    def value(self, px: Vec) -> float:
        return float(self.w @ px) + self.cash

QTRANS_CASES = {"discount": [(0.05, 2.0)]}
