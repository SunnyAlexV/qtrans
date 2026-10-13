import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Payoff(ABC):
    """What an option pays at expiry."""

    def __init__(self, strike: float):
        if strike <= 0:
            raise ValueError("strike must be positive")
        self.strike = strike

    @abstractmethod
    def value(self, spot: float) -> float:
        ...

    def name(self) -> str:
        return "payoff"

    def describe(self) -> str:
        return f"{self.name()} K={self.strike}"


class Call(Payoff):
    def value(self, spot: float) -> float:
        return max(spot - self.strike, 0.0)

    def name(self) -> str:
        return "call"


class Put(Payoff):
    def value(self, spot: float) -> float:
        return max(self.strike - spot, 0.0)

    def name(self) -> str:
        return "put"


class CappedCall(Call):
    def __init__(self, strike: float, cap: float):
        super().__init__(strike)
        self.cap = cap

    def value(self, spot: float) -> float:
        return min(super().value(spot), self.cap)

    def name(self) -> str:
        return "capped " + super().name()


def expected_payoff(p: Payoff, spots: list[float]) -> float:
    total = 0.0
    for s in spots:
        total += p.value(s)
    return total / len(spots)


def book_value(book: list[Payoff], spot: float) -> float:
    return sum((p.value(spot) for p in book), 0.0)


def discounted(p: Payoff, spot: float, r: float, t: float) -> float:
    return math.exp(-r * t) * p.value(spot)


@dataclass
class Instrument:
    notional: float

    def exposure(self) -> float:
        return self.notional


@dataclass
class Bond(Instrument):
    coupon: float = 0.05

    def exposure(self) -> float:
        return self.notional * (1.0 + self.coupon)


def total_exposure(xs: list[Instrument]) -> float:
    total = 0.0
    for x in xs:
        total += x.exposure()
    return total


book: list[Payoff] = [Call(100.0), Put(95.0), CappedCall(100.0, 15.0)]
for p in book:
    print(p.describe(), p.value(120.0), p.value(90.0))
print(book_value(book, 90.0), book_value(book, 130.0))
print(expected_payoff(book[2], [80.0, 100.0, 110.0, 140.0]))
print(round(discounted(Call(90.0), 100.0, 0.05, 1.0), 6))
xs = [Instrument(100.0), Bond(200.0), Bond(50.0, 0.1)]
print(total_exposure(xs))
for x in xs:
    print(x)
try:
    Put(-1.0)
except ValueError as e:
    print("refused:", e)
