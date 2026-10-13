"""Python's random module in C++ (qtrans_pyrandom.hpp): the Mersenne Twister seeded as CPython
seeds it, and random(), getrandbits, randrange, randint, choice, shuffle, sample, choices,
uniform, triangular, gauss, normalvariate, lognormvariate and expovariate as Lib/random.py
computes them, against the running Python on many seeds and arguments: the same numbers, in the
same order, and the same errors with the same words."""
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qtrans.translator import PYRANDOM_HEADER, PYTHON_HEADER, RUNTIME_HEADER
from qtrans.verify import find_compiler
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).parent / "data"
compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")


def _hex(x):
    return float(x).hex()


def _cpp_hex(text):
    return float.fromhex(text).hex() if text not in ("?",) and ":" not in text else text


def _script(seed):
    """(input line, what Python gives) for one seeded run: every call, odd arguments included."""
    r = random.Random()
    steps = []

    def run(line, f, show):
        try:
            v = f()
            steps.append((line, show(v)))
        except Exception as e:  # noqa: BLE001 - errors are results too
            steps.append((line, f"{type(e).__name__}: {e}"))

    ints = lambda v: " ".join(map(str, v)) or "-"  # noqa: E731
    run(f"seed {seed}", lambda: r.seed(seed), lambda v: "ok")
    g = random.Random(seed ^ 0x5DEECE66D)  # the arguments themselves
    for _ in range(60):
        k = g.randrange(19)
        if k == 0:
            run("random", r.random, _hex)
        elif k == 1:
            b = g.choice([0, 1, 5, 31, 32, 33, 52, 53, 63, 64])
            run(f"getrandbits {b}", lambda: r.getrandbits(b), str)
        elif k == 2:
            a = g.choice([1, 2, 3, 10, 1000, 2 ** 31, 2 ** 32 + 1, 2 ** 62 + 7, 0, -3])
            run(f"randrange1 {a}", lambda: r.randrange(a), str)
        elif k == 3:
            a, b = g.randint(-20, 20), g.randint(-20, 40)
            run(f"randrange2 {a} {b}", lambda: r.randrange(a, b), str)
        elif k == 4:
            a, b, s = g.randint(-20, 20), g.randint(-20, 40), g.choice([-3, -1, 0, 2, 5])
            run(f"randrange3 {a} {b} {s}", lambda: r.randrange(a, b, s), str)
        elif k == 5:
            a, b = g.randint(-10, 10), g.randint(-12, 2 ** 40)
            run(f"randint {a} {b}", lambda: r.randint(a, b), str)
        elif k == 6:
            a, b = g.uniform(-5, 5), g.uniform(-5, 50)
            run(f"uniform {a!r} {b!r}", lambda: r.uniform(a, b), _hex)
        elif k == 7:
            m, s = g.uniform(-3, 3), g.uniform(0, 4)
            run(f"gauss {m!r} {s!r}", lambda: r.gauss(m, s), _hex)
        elif k == 8:
            m, s = g.uniform(-3, 3), g.uniform(0, 4)
            run(f"normalvariate {m!r} {s!r}", lambda: r.normalvariate(m, s), _hex)
        elif k == 9:
            m, s = g.uniform(-1, 1), g.uniform(0, 1)
            run(f"lognormvariate {m!r} {s!r}", lambda: r.lognormvariate(m, s), _hex)
        elif k == 10:
            lam = g.choice([0.5, 1.0, 3.25, -2.0, 0.0])
            run(f"expovariate {lam!r}", lambda: r.expovariate(lam), _hex)
        elif k == 11:
            a, b = g.uniform(-5, 5), g.uniform(-5, 5)
            run(f"triangular2 {a!r} {b!r}", lambda: r.triangular(a, b), _hex)
        elif k == 12:
            a, b, m = g.uniform(-5, 5), g.choice([g.uniform(-5, 5), None]), g.uniform(-5, 5)
            b = a if b is None else b  # high == low: Python returns low
            run(f"triangular3 {a!r} {b!r} {m!r}", lambda: r.triangular(a, b, m), _hex)
        elif k == 13:
            n = g.choice([0, 1, 2, 7, 100])
            run(f"choice {n}", lambda: r.choice(list(range(n))), str)
        elif k == 14:
            n = g.choice([0, 1, 2, 9, 40])
            def shuf(n=n):
                v = list(range(n))
                r.shuffle(v)
                return v
            run(f"shuffle {n}", shuf, ints)
        elif k == 15:
            n = g.choice([0, 3, 10, 30, 200, 2000])
            kk = g.choice([0, 1, 5, 6, 12, 25, n, n + 1, -1])
            run(f"sample {n} {kk}", lambda: r.sample(list(range(n)), kk), ints)
        elif k == 16:
            n, kk = g.choice([0, 1, 4, 13]), g.choice([0, 1, 3, 8])
            run(f"choices {n} {kk}", lambda: r.choices(list(range(n)), k=kk), ints)
        elif k == 17:
            n, kk = g.choice([0, 1, 3, 6]), g.choice([1, 4])
            w = [g.choice([0.0, 0.5, 1.0, 2.25, g.random()]) for _ in range(n)]
            run(f"wchoices {n} {kk} " + " ".join(repr(x) for x in w), lambda: r.choices(list(range(n)), w, k=kk), ints)
        else:
            run("state", lambda: (random.Random.getrandbits(_copy(r), 32), r.gauss_next),
                lambda v: f"{v[0]} {'None' if v[1] is None else _hex(v[1])}")
    return steps


def _copy(r):
    c = random.Random()
    c.setstate(r.getstate())
    return c


SEEDS = [0, 1, 2, 42, -42, 7, 12345, 2 ** 31 - 1, 2 ** 32, 2 ** 32 + 1, 2 ** 63 - 1, -(2 ** 63), 987654321987]


@compiled
def test_the_generator_and_every_function_give_pythons_numbers_and_errors(tmp_path):
    for h in (RUNTIME_HEADER, PYTHON_HEADER, PYRANDOM_HEADER):
        shutil.copy(h, tmp_path / h.name)
    shutil.copy(DATA / "pyrandom_probe.cpp", tmp_path / "probe.cpp")
    exe = tmp_path / ("probe.exe" if sys.platform == "win32" else "probe")
    subprocess.run(find_compiler().command(tmp_path / "probe.cpp", exe, tmp_path), check=True, capture_output=True)
    steps = [s for seed in SEEDS + list(range(100, 160)) for s in _script(seed)]
    out = subprocess.run([str(exe)], input="\n".join(line for line, _ in steps) + "\n", capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert len(out) == len(steps)
    got = [" ".join(_cpp_hex(t) if t.startswith(("0x", "-0x")) else t for t in o.split(" ")) if ":" not in o
           else o.strip() for o in (x.strip() for x in out)]
    bad = [(line, want, g) for (line, want), g in zip(steps, got) if want != g]
    assert not bad, f"{len(bad)} of {len(steps)} differ, first: {bad[:4]}"
    kinds = {line.split()[0] for line, _ in steps}
    assert len(kinds) >= 18 and sum(":" in w for _, w in steps) > 20  # every function, and errors


@compiled
def test_a_seeded_program_and_its_functions_agree_with_python(tmp_path):
    """random.seed, gauss in a Monte Carlo loop, choices for a bootstrap, uniform, shuffle,
    randint, choice and sample in a program; each function verified with the module's generator
    seeded the same way on both sides and compared after the call."""
    from qtrans.verify import verify_file
    f = tmp_path / "random_module_mc.py"
    shutil.copy(DATA / "random_module_mc.py", f)
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert not rep.translation.report.rejected, rep.translation.report.to_text()
    assert rep.program is not None and rep.program.identical, rep.to_text()
    assert rep.ok and rep.warnings == [], rep.to_text()
    worst = max(c.ulps or 0 for fn in rep.functions for c in fn.cases if c.outcome == "match")
    assert worst == 0, "every matching case bit-identical"


def _rejected(src):
    from qtrans import translate_source
    r = translate_source(src, "m")
    return " ".join(f"{x.reason} {x.hint or ''}" for x in r.report.rejected)


@pytest.mark.parametrize("src, words", [
    ("import random\nprint(random.random())\n", "before the program calls random.seed"),
    ("import random\nrandom.seed()\nprint(random.random())\n", "seeds from the operating system"),
    ("import random\nrandom.seed('abc')\n", "only int seeds"),
    ("import random\n\ndef f() -> float:\n    return random.random() - random.random()\n", "two random draws"),
    ("import random\n\ndef f(xs: list[float]) -> None:\n    random.shuffle(xs)\n", "changes the list 'xs' passed in"),
    ("import random\n\ndef f() -> float:\n    r = random.Random(3)\n    return r.random()\n", "objects are not supported yet"),
    ("import random\n\ndef f() -> float:\n    return random.betavariate(1.0, 2.0)\n", "not supported yet"),
    ("import random\n\ndef f(n: int) -> int:\n    return random.getrandbits(n)\n", "0 to 63"),
])
def test_refusals(src, words):
    assert words in _rejected(src)
