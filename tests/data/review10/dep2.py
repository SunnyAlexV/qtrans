from dataclasses import dataclass


@dataclass
class C:
    a: float

    def ratio(self, b: float) -> float:  # qtrans: ieee
        return self.a / b

    def exact(self, b: float) -> float:
        return self.a / b


@dataclass
class D:
    a: float

    def __post_init__(self):  # qtrans: ieee
        self.a = 1.0 / self.a
