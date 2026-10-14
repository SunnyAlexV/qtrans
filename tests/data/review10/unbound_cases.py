def first_even_carry(n: int) -> int:
    t = 0
    for k in range(n):
        if k % 2 == 1:
            last = k
        t += last
    return t


def one_side(p0: float) -> float:
    if p0 > 2.0:
        pass
    else:
        n1 = 3.0
    return n1


def aug(p0: float) -> float:
    if p0 > 1.0:
        acc = 0.0
    acc += p0
    return acc


def caught(p0: float) -> float:
    try:
        return one_side(p0)
    except UnboundLocalError as e:
        print(type(e).__name__, e)
        return -1.0


def short_circuit(p0: float) -> bool:
    if p0 > 0.0:
        flag = True
    return p0 > 0.0 and flag


QTRANS_CASES = {
    "first_even_carry": [(0,), (1,), (2,), (5,)],
    "one_side": [(1.0,), (3.0,)],
    "aug": [(0.5,), (2.0,)],
    "caught": [(1.0,), (3.0,)],
    "short_circuit": [(-1.0,), (1.0,)],
}

x = 1.0
if x > 2.0:
    y = 3.0
print(caught(5.0), aug(3.0))
print(y)
