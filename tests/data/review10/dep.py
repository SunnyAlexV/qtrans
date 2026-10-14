import numpy as np


def inner(x: float) -> float:  # qtrans: ieee
    return 1.0 / x


def outer(x: float) -> float:
    return inner(x)


def mixed(x: float, y: float) -> float:
    return max(np.exp(x), 0.0) / y


def exact(x: float) -> float:
    return 1.0 / x
