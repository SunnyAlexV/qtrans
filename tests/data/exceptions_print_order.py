def noisy(x: float) -> float:
    print("noisy", x)
    return x * 2.0


def safe_ratio(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError as e:
        print("no ratio:", e)
        return 0.0


print(safe_ratio(1.0, 4.0), safe_ratio(1.0, 0.0))
print(f"{noisy(1.0)} and {noisy(2.0)}", noisy(3.0))
