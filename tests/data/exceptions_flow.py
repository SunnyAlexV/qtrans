import math


class BookError(Exception):
    pass


class MissingQuote(BookError, ):
    """No quote for a ticker."""


class StaleQuote(KeyError):
    pass


def quote(book: dict[str, float], k: str) -> float:
    if k not in book:
        raise MissingQuote(f"no quote for {k}")
    if book[k] < 0:
        raise StaleQuote(k)
    return book[k]


def kept(x: float) -> float:
    try:
        return x * 2.0
    finally:
        x = -1.0
        print("finally sees", x)


def replaced(x: float) -> float:
    try:
        if x > 0:
            raise ValueError("first")
        return x
    finally:
        if x > 1:
            raise RuntimeError("from finally")


def reraise(n: int) -> int:
    try:
        try:
            return 10 // n
        except ZeroDivisionError:
            print("inner saw it")
            raise
    except ArithmeticError as e:
        print("outer:", type(e).__name__, e)
        return -1


def growth(r: float, t: float) -> float:
    try:
        return math.exp(r * t) ** 2.0
    except OverflowError as e:
        print("overflow:", e)
        return math.inf


def nothing() -> str:
    try:
        raise ValueError()
    except ValueError as e:
        return "[" + str(e) + "]"


def parse(s: str) -> int:
    try:
        v = int(s)
    except (ValueError, TypeError) as e:
        print("bad:", e)
        return 0
    else:
        print("parsed")
        return v


def pick(xs: list[float], i: int) -> float:
    try:
        return xs[i]
    except LookupError as e:
        print(type(e).__name__, e)
        return 0.0


book = {"AAPL": 190.5, "OLD": -1.0}
for k in ["AAPL", "MSFT", "OLD"]:
    try:
        print(k, quote(book, k))
    except StaleQuote as e:
        print("stale", e)
    except BookError as e:
        print("book error:", type(e).__name__, e)
print(kept(3.0))
for x in [-2.0, 0.5, 2.0]:
    try:
        print(replaced(x))
    except Exception as e:
        print(type(e).__name__, e)
print(reraise(0))
print(reraise(3))
print(growth(1.0, 2.0))
print(growth(500.0, 2.0))
print(nothing())
print(parse("42"))
print(parse("4x2"))
print(pick([1.0, 2.0], 5))
for d in [2.0, 0.0]:
    try:
        print(1.0 / d, 0.0 ** -1.0 if d == 0.0 else 1.0)
    except ZeroDivisionError as e:
        print("zero:", e)
