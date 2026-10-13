import math


class Shape:
    def __init__(self, name: str):
        self.name = name
        self.moves = 0

    def area(self) -> float:
        return 0.0

    def move(self) -> None:
        self.moves += 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, area={self.area():.3f})"


class Circle(Shape):
    def __init__(self, name: str, r: float):
        super().__init__(name)
        self.r = r

    def area(self) -> float:
        return math.pi * self.r ** 2


class Square(Shape):
    def __init__(self, name: str, side: float):
        super().__init__(name)
        self.side = side

    def area(self) -> float:
        return self.side * self.side

    def grow(self, k: float) -> None:
        self.side *= k
        self.move()


def total_area(shapes: list[Shape]) -> float:
    total = 0.0
    for s in shapes:
        total += s.area()
    return total


def count_circles(shapes: list[Shape]) -> int:
    n = 0
    for s in shapes:
        if isinstance(s, Circle):
            n += 1
    return n


def bump_all(shapes: list[Shape]) -> int:
    for s in shapes:
        s.move()
    return len(shapes)


def biggest(shapes: list[Shape]) -> str:
    best = shapes[0]
    for s in shapes:
        if s.area() > best.area():
            best = s
    return type(best).__name__ + " " + best.name


shapes: list[Shape] = [Circle("c1", 1.0), Square("s1", 2.0), Shape("plain")]
a = shapes[1]
if isinstance(a, Square):
    print("square first")
print(total_area(shapes), count_circles(shapes))
sq = Square("s2", 3.0)
sq.grow(2.0)
shapes.append(sq)
bump_all(shapes)
for s in shapes:
    print(s, s.moves)
print(biggest(shapes))
same = shapes[3]
print(same is sq, same == sq, shapes[0] == shapes[1])
