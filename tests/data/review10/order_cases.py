import math
from dataclasses import dataclass


@dataclass
class P:
    v: float

    def m(self, a: float, b: float) -> float:
        return self.v + a + b


@dataclass
class Box:
    v: float


def bump(p: Box) -> float:
    p.v += 1.0
    return 0.0


def g(a: float, b: float) -> float:
    return a + b


def h(a: float, b: float) -> float:
    return a - b


def ratio(a: float, b: float) -> float:
    return a / b


def method_order(a: float, b: float, c: float, d: float) -> float:
    return P(a / b).m(math.log(c), math.sqrt(d))


def effect_then_raise(a: float, b: float) -> float:
    p = Box(0.0)
    try:
        return g(bump(p), a / b)
    except ZeroDivisionError:
        return p.v


def call_order(a: float, b: float, c: float) -> float:
    return h(ratio(a, b), math.log(c))


def aug_index(k: int, a: float, b: float) -> float:
    xs = [1.0, 2.0]
    xs[k] += math.log(a) / math.sqrt(b)
    return xs[0]


QTRANS_CASES = {
    "method_order": [(1.0, 0.0, -1.0, 1.0), (1.0, 2.0, -1.0, -1.0), (1.0, 2.0, 3.0, 4.0)],
    "effect_then_raise": [(1.0, 0.0), (1.0, 2.0)],
    "call_order": [(1.0, 0.0, -1.0), (1.0, 2.0, 3.0)],
    "aug_index": [(5, -1.0, 1.0), (0, -1.0, 1.0), (1, 2.0, 3.0), (-3, 1.0, -1.0)],
}
