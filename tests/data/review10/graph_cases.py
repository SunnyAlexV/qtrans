import math
from dataclasses import dataclass


def r(a, b):
    return a / b


def delete(a: float, b: float) -> float:  # qtrans: ieee
    return a / b


class Base:
    def __init__(self, x: float):
        self.x = 1.0 / x


class Sub(Base):
    pass


@dataclass
class Q:
    a: float
    b: float

    def __post_init__(self):  # qtrans: ieee
        self.a = self.a / self.b


def h(a: float, b: float) -> float:
    return a - b


def k(x: float) -> float:  # qtrans: ieee
    return 1.0 / x


def f(x: float, n: int) -> float:
    if n > 0:
        return g(x, n - 1)
    return k(x)


def g(x: float, n: int) -> float:
    return f(x, n)


def overload_order(a: float, b: float, c: float) -> float:
    return h(r(a, b), math.log(c)) + r(1, 2)


def keyword_caller(a: float, b: float) -> float:
    return delete(a, b)


def ctor_order(c: float) -> float:
    return h(math.log(c), Sub(c).x)


def mk(a: float, b: float) -> float:
    return Q(a, b).a


QTRANS_CASES = {
    "overload_order": [(1.0, 0.0, -1.0), (1.0, 2.0, 3.0)],
    "keyword_caller": [(1.0, 0.0)],
    "ctor_order": [(0.0,), (-1.0,), (2.0,)],
    "mk": [(1.0, 0.0), (1.0, 2.0)],
    "f": [(0.0, 0), (0.0, 2), (1.0, 1)],
    "g": [(0.0, 0), (0.0, 1)],
}
