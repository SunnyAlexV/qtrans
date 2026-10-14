from dep2 import C, D


def safe(c: C, b: float) -> float:
    try:
        return c.ratio(b)
    except ZeroDivisionError:
        return -1.0


def safe_exact(c: C, b: float) -> float:
    try:
        return c.exact(b)
    except ZeroDivisionError:
        return -1.0


def safe_ctor(a: float) -> float:
    try:
        return D(a).a
    except ZeroDivisionError:
        return -1.0
