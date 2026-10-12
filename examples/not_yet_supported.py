"""Code qtrans stage 1 deliberately refuses, to show what the report says.
Two functions here are fine; the rest each use one unsupported feature."""
import math

import numpy as np


def discount(r: float, t: float) -> float:
    return math.exp(-r * t)


def annuity(r: float, n: int) -> float:
    return sum(map(lambda t: discount(r, t), range(1, n + 1)))  # lambda and map()


def portfolio_value(weights: np.ndarray, prices: np.ndarray) -> float:  # arrays: stage 2
    return float(weights @ prices)


def safe_log(x: float) -> float:
    try:
        return math.log(x)
    except ValueError:
        return -math.inf


def lookup(x: float) -> float:
    rates = {"base": x}  # dicts
    return rates["base"]


def no_annotations(x, y):
    return x + y


def uses_annuity(r: float) -> float:
    return annuity(r, 10) * 2  # fine on its own, but it calls a rejected function


def last_index(n: int) -> int:
    for i in range(n):
        pass
    return i  # loop variable read after the loop


def sometimes_none(x: float) -> float:
    if x > 0:
        return x


def forward(S: float, r: float, T: float) -> float:
    return S / discount(r, T)
