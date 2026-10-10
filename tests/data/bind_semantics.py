"""Small functions that pin down what the bindings must preserve at the Python boundary."""
import numpy as np

from qtrans.typing import Mat, Vec


def scale_inplace(x: Vec, k: float) -> None:
    x *= k


def fill_then_fail(x: Vec, n: int) -> float:
    x[0] = 1.0
    if n > 0:
        raise ValueError("n must be <= 0")
    return float(np.sum(x))


def draws(rng: np.random.Generator, n: int) -> Vec:
    return rng.standard_normal(n)


def draw_then_check(rng: np.random.Generator, cap: float) -> float:
    z = rng.standard_normal()
    if z > cap:
        raise ValueError("draw above cap")
    return z


def solve2(A: Mat, b: Vec) -> Vec:
    return np.linalg.solve(A, b)


def idiv(a: int, b: int) -> int:
    return a // b


def pick(x: Vec, i: int) -> float:
    return x[i]


def maybe(x: float | None = None, s: str = "abc") -> float:
    if x is None:
        return 0.5 if s == "abc" else 1.5
    return x + 1.0


def add_into(a: Vec, b: Vec) -> None:
    a += b


def total(m: Mat) -> float:
    return float(np.sum(m))


QTRANS_CASES = {
    "solve2": [(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 1.0]))],
    "pick": [(np.array([1.0, 2.0]), 5), (np.array([1.0, 2.0]), -1)],
    "idiv": [(7, 0), (-7, 2)],
}
