import math
import random


def gbm_terminal(s0: float, mu: float, sigma: float, t: float) -> float:
    z = random.gauss(0.0, 1.0)
    return s0 * math.exp((mu - 0.5 * sigma ** 2) * t + sigma * math.sqrt(t) * z)


def mc_call(s0: float, k: float, sigma: float, t: float, n: int) -> float:
    total = 0.0
    for _ in range(n):
        st = gbm_terminal(s0, 0.0, sigma, t)
        total += max(st - k, 0.0)
    return total / n


def bootstrap_mean(xs: list[float], b: int) -> list[float]:
    means = []
    for _ in range(b):
        draw = random.choices(xs, k=len(xs))
        means.append(math.fsum(draw) / len(draw))
    return means


random.seed(2026)
print(round(mc_call(100.0, 100.0, 0.2, 1.0, 2000), 6))
rets = [random.uniform(-0.02, 0.03) for _ in range(20)]
means = bootstrap_mean(rets, 5)
print(means)
deck = list(range(10))
random.shuffle(deck)
print(deck, random.randint(1, 6), random.choice(["up", "down"]))
print(random.sample(deck, 3))
print(math.fsum([0.1] * 10), sum([0.1] * 10))
