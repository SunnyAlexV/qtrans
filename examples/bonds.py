"""Fixed-coupon bond analytics: price, yield to maturity, duration and convexity.
Exercises loops, int/float mixing and a Newton solver."""


def bond_price(face: float, coupon_rate: float, ytm: float, years: int, freq: int = 2) -> float:
    """Clean price on a coupon date of a bond paying `freq` coupons a year."""
    if freq <= 0 or years <= 0:
        raise ValueError("years and freq must be positive")
    n = years * freq
    c = face * coupon_rate / freq
    y = ytm / freq
    price = 0  # starts as an int; becomes a float as coupons are added
    for k in range(1, n + 1):
        price += c / (1 + y) ** k
    price += face / (1 + y) ** n
    return price


def macaulay_duration(face: float, coupon_rate: float, ytm: float, years: int, freq: int = 2) -> float:
    """Present-value-weighted average time to the cash flows, in years."""
    n = years * freq
    c = face * coupon_rate / freq
    y = ytm / freq
    weighted = 0.0
    pv_total = 0.0
    for k in range(1, n + 1):
        cf = c + face if k == n else c
        pv = cf / (1 + y) ** k
        weighted += k / freq * pv
        pv_total += pv
    return weighted / pv_total


def modified_duration(face: float, coupon_rate: float, ytm: float, years: int, freq: int = 2) -> float:
    return macaulay_duration(face, coupon_rate, ytm, years, freq) / (1 + ytm / freq)


def convexity(face: float, coupon_rate: float, ytm: float, years: int, freq: int = 2) -> float:
    n = years * freq
    c = face * coupon_rate / freq
    y = ytm / freq
    total = 0.0
    price = 0.0
    for k in range(1, n + 1):
        cf = c + face if k == n else c
        pv = cf / (1 + y) ** k
        total += pv * k * (k + 1)
        price += pv
    return total / (price * (1 + y) ** 2 * freq ** 2)


def ytm_newton(price: float, face: float, coupon_rate: float, years: int, freq: int = 2) -> float:
    """Yield to maturity by Newton's method on price(y) - price = 0."""
    if price <= 0:
        raise ValueError("price must be positive")
    y = coupon_rate if coupon_rate > 0 else 0.05
    for _ in range(100):
        p = bond_price(face, coupon_rate, y, years, freq)
        # dP/dy = -modified duration * price
        dp = -modified_duration(face, coupon_rate, y, years, freq) * p
        step = (p - price) / dp
        y -= step
        if abs(step) < 1e-14:
            return y
    raise RuntimeError("yield did not converge")


def payments_remaining(months_to_maturity: int, freq: int) -> int:
    """Coupons still to come; uses Python floor division and modulo semantics."""
    months_per_coupon = 12 // freq
    full, part = months_to_maturity // months_per_coupon, months_to_maturity % months_per_coupon
    return full + (1 if part > 0 else 0)


QTRANS_CASES = {
    "bond_price": [(100.0, cr, y, n, f) for cr in (0.0, 0.03, 0.08) for y in (0.01, 0.05, 0.12)
                   for n in (1, 10, 30) for f in (1, 2, 12)] + [(100.0, 0.05, 0.05, 0, 2)],
    "macaulay_duration": [(100.0, 0.05, y, 10, 2) for y in (0.01, 0.05, 0.2)],
    "modified_duration": [(100.0, 0.05, y, 30, 2) for y in (0.01, 0.05, 0.2)],
    "convexity": [(100.0, cr, 0.04, n, 2) for cr in (0.0, 0.06) for n in (2, 30)],
    "ytm_newton": [(p, 100.0, 0.05, 10, 2) for p in (80.0, 95.5, 100.0, 112.25)],
    "payments_remaining": [(m, f) for m in (0, 1, 5, 6, 7, 59, 60, 61, -5) for f in (1, 2, 4, 12)],
}
