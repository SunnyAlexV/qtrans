from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Leg(ABC):
    """One leg of a position."""
    qty: float

    @abstractmethod
    def pv(self, spot: float) -> float:
        ...

    def scaled(self, spot: float) -> float:
        return self.qty * self.pv(spot)


@dataclass
class Forward(Leg):
    strike: float = 100.0

    def pv(self, spot: float) -> float:
        return spot - self.strike


@dataclass
class Digital(Leg):
    barrier: float = 100.0
    pays: float = 1.0

    def pv(self, spot: float) -> float:
        return self.pays if spot >= self.barrier else 0.0


@dataclass
class Position:
    leg: Leg
    book: str = "main"

    def value(self, spot: float) -> float:
        return self.leg.scaled(spot)


def net(legs: list[Leg], spot: float) -> float:
    total = 0.0
    for leg in legs:
        total += leg.scaled(spot)
    return total


def pick(flag: bool, q: float) -> Leg:
    if flag:
        leg = Forward(q, 95.0)
    else:
        leg = Digital(q)
    return leg


def double_qty(legs: list[Leg]) -> None:
    for leg in legs:
        leg.qty *= 2.0


legs: list[Leg] = [Forward(2.0), Digital(3.0, 105.0), Forward(-1.0, 90.0)]
print(net(legs, 101.0), net(legs, 110.0))
p = Position(pick(False, 4.0))
print(p, p.value(120.0))
q = Position(pick(True, 1.5), "hedge")
print(q.value(99.0), q)
double_qty(legs)
for leg in legs:
    print(leg, leg.scaled(100.0))
legs.append(q.leg)
print(len(legs), net(legs, 100.0))
