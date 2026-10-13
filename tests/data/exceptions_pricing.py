import math


class PricingError(ValueError):
    """A price that can't be computed."""


class StrikeError(PricingError):
    pass


def price(s: float, k: float) -> float:
    if k <= 0:
        raise StrikeError(f"bad strike {k}")
    if s < 0:
        raise PricingError("negative spot")
    return math.log(s / k)


def safe_price(s: float, k: float) -> float:
    try:
        return price(s, k)
    except StrikeError as e:
        print("strike:", e)
        return -1.0
    except ValueError as e:
        print("value:", e)
        return -2.0
    except ZeroDivisionError as e:
        print("zero:", e)
        return -3.0


def ratio(a: float, b: float) -> float:
    try:
        r = a / b
    except ZeroDivisionError:
        r = math.inf
    else:
        r = r * 2.0
    finally:
        print("ratio done")
    return r


def lookup(d: dict[str, float], k: str) -> float:
    try:
        return d[k]
    except (KeyError, IndexError) as e:
        print("missing", e)
        return 0.0


def total(xs: list[float]) -> float:
    t = 0.0
    for x in xs:
        try:
            if x < 0:
                continue
            t += math.sqrt(x)
            if t > 10:
                break
        finally:
            t += 0.5
    return t


print(safe_price(100.0, 0.0))
print(safe_price(-1.0, 5.0))
print(safe_price(0.0, 5.0))
print(safe_price(100.0, 90.0))
print(ratio(1.0, 0.0))
print(ratio(3.0, 2.0))
print(lookup({"a": 1.0}, "a"))
print(lookup({"a": 1.0}, "b"))
print(total([1.0, -4.0, 9.0, 100.0, 4.0]))
try:
    print(1 // 0)
except ArithmeticError as e:
    print(type(e).__name__, e)
try:
    assert 1 > 2, "math is broken"
except AssertionError as e:
    print("assert:", e)
