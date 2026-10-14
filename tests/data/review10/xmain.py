from dep import exact, mixed, outer


def safe(x: float) -> float:
    try:
        return outer(x)
    except ZeroDivisionError:
        return -1.0


def safe_mixed(x: float, y: float) -> float:
    try:
        return mixed(x, y)
    except ZeroDivisionError:
        return -1.0


def safe_exact(x: float) -> float:
    try:
        return exact(x)
    except ZeroDivisionError:
        return -1.0
