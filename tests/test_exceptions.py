"""Exceptions as Python has them: your own exception classes, try / except / else / finally,
raise and re-raise, and Python's float errors (1.0 / 0.0, math.sqrt(-1), math.exp(1000)) raised
where a handler may catch them, with the running Python's own wording (qtrans_python.hpp is
written from it, so 3.10's "float division by zero" and 3.14's "division by zero" both come out
right)."""
import math
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qtrans.translator import PRINT_HEADER, PYTHON_HEADER, RUNTIME_HEADER, UNICODE_HEADER, translate_source
from qtrans.verify import find_compiler, verify_file
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).parent / "data"
compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")


@compiled
@pytest.mark.parametrize("name", ["exceptions_pricing.py", "exceptions_flow.py", "exceptions_print_order.py"])
def test_programs_print_what_python_prints(tmp_path, name):
    """Own classes (and their family), as e / str(e) / type(e).__name__, several classes in one
    clause, else, finally on every way out (return, break, continue, an exception), an error
    raised in finally replacing the first, bare raise to an outer handler, ValueError() with no
    message, KeyError's repr, float errors caught, and a print() line whose values print or raise
    (worked out left to right): each line as Python prints it."""
    f = tmp_path / name
    shutil.copy(DATA / name, f)
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    assert rep.ok and rep.program.identical and rep.warnings == [], rep.to_text()


@compiled
def test_float_errors_and_messages_agree_with_python_on_every_probe(tmp_path):
    """qt::py::math_*, power, truediv, math_pow, math_fmod against this Python, on 0, ±0.0,
    ±inf, NaN, domain edges and a few thousand random values: the same value, or the same error
    with the same message."""
    for h in (RUNTIME_HEADER, PYTHON_HEADER, PRINT_HEADER, UNICODE_HEADER):
        shutil.copy(h, tmp_path / h.name)
    shutil.copy(DATA / "python_errors.cpp", tmp_path / "fe.cpp")
    exe = tmp_path / ("fe.exe" if sys.platform == "win32" else "fe")
    subprocess.run(find_compiler().command(tmp_path / "fe.cpp", exe, tmp_path), check=True, capture_output=True)
    lines, want = _probes()
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert len(out) == len(want)
    # gamma and lgamma are CPython's own code, ported operation for operation: the same values,
    # except where CPython itself was compiled to fuse a multiply and an add (clang does on ARM64)
    fused = platform.machine().lower() in ("arm64", "aarch64")
    bad = [(n, w, g) for (n, w), g in zip(want, out) if w != g and not (
        fused and n.startswith(("gamma", "lgamma")) and "Error" not in w + g)]
    assert not bad, f"{len(bad)} of {len(want)} differ, first: {bad[:3]}"


def _py(f):
    try:
        return repr(f())
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        return f"{type(e).__name__}: {e}"


def _probes():
    names = ["sqrt", "exp", "log", "log10", "log2", "log1p", "expm1", "sin", "cos", "tan", "asin", "acos", "atan",
             "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "erfc", "cbrt", "exp2", "fabs", "gamma", "lgamma"]
    inf, nan = math.inf, math.nan
    vals = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, -1.5, 1e-300, -1e-300, 5e-324, 709.0, 710.0, -710.0, 1000.0,
            -1000.0, 1e308, -1e308, inf, -inf, nan, 171.6, 171.7, 172.0, -171.5, -3.0, -1e20, 1e20, 1 - 1e-16, -1 + 1e-16, -5.0, 3.0]
    rng = random.Random(7)
    vals += [rng.uniform(-800, 800) for _ in range(400)] + [rng.uniform(-3, 3) for _ in range(400)]
    lines, want = [], []
    for n in names:
        if not hasattr(math, n):
            continue
        f = getattr(math, n)
        for v in vals:
            lines.append(f"{n} {v.hex()}")
            want.append((n, _py(lambda: f(v))))
            if math.isfinite(v) and v == int(v) and abs(v) < 1e18:
                lines.append(f"{n}_i {v.hex()}")
                want.append((n, _py(lambda: f(int(v)))))
    xs = [0.0, -0.0, 1.0, -1.0, 2.0, 10.0, -10.0, 0.5, 1e-300, 1e300, inf, -inf, nan, -8.0, 1.5]
    ys = [0.0, -1.0, -0.5, 0.5, 1.0, 2.0, 400.0, 401.0, -2.0, inf, -inf, nan, 1e10, 1 / 3, 3.0]
    for x in xs:
        for y in ys:
            for name, f in (("pow", lambda: x ** y), ("mpow", lambda: math.pow(x, y)),
                            ("fmod", lambda: math.fmod(x, y)), ("div", lambda: x / y)):
                r = _py(f)
                if "j)" in r:
                    continue  # complex: Python's answer is no double
                lines.append(f"{name} {x.hex()} {y.hex()}")
                want.append((name, r))
            if math.isfinite(x) and x == int(x) and math.isfinite(y) and y == int(y) and y < 0:
                lines.append(f"ipow {x.hex()} {y.hex()}")
                want.append(("ipow", _py(lambda: int(x) ** int(y))))
    return lines, want


def test_the_messages_header_is_this_pythons():
    from qtrans import pyfacts
    text = PYTHON_HEADER.read_text(encoding="utf-8")
    assert f'#define QTRANS_PYTHON_VERSION "{".".join(map(str, sys.version_info[:3]))}"' in text
    assert text == pyfacts.generate()
    try:
        1.0 / 0.0
    except ZeroDivisionError as e:
        assert f'float_truediv_zero = "{e}";' in text


SRC = '''
import math


class PricingError(ValueError):
    """Can't price it."""


def price(s: float, k: float) -> float:
    if k <= 0:
        raise PricingError(f"bad strike {k}")
    return math.log(s / k)


def safe(s: float, k: float) -> float:
    try:
        return price(s, k)
    except ValueError as e:
        return -1.0


def plain(a: float, b: float) -> float:
    return a / b


def fast(a: float, b: float) -> float:  # qtrans: ieee
    return a / b
'''


def test_the_cpp_reads_as_written_by_hand():
    h = translate_source(SRC, "m").header
    assert "struct PricingError : qt::ValueError {  // class PricingError(ValueError)" in h
    assert 'const char* type_name() const noexcept override { return "PricingError"; }' in h
    assert 'throw PricingError("bad strike " + qt::py::str(k));' in h
    assert "} catch (const qt::ValueError& e) {  // except ValueError as e" in h
    # since 1.0 a division and a log raise as Python's do everywhere (price runs inside safe's
    # try, plain doesn't), but not in a function marked `# qtrans: ieee`
    assert "return qt::py::math_log(qt::py::truediv(s, k));" in h
    assert "inline double plain(double a, double b) {\n    return qt::py::truediv(a, b);" in h
    assert "inline double fast(double a, double b) {\n    return a / b;" in h


@pytest.mark.parametrize("src, words", [
    ("def f(x: float) -> float:\n    try:\n        return 1.0 / x\n    except:\n        return 0.0\n",
     "bare `except:`"),
    ("def f(x: float) -> float:\n    try:\n        return 1.0 / x\n    except BaseException:\n        return 0.0\n",
     "BaseException"),
    ("def f(x: float) -> float:\n    try:\n        return 1.0 / x\n    finally:\n        return 0.0\n",
     "a return inside `finally:`"),
    ("class E(ValueError):\n    def __init__(self, v: float):\n        self.v = v\n\n"
     "def f(x: float) -> float:\n    raise E(x)\n", "has fields or methods"),
    ("def f(x: float) -> float:\n    raise ValueError('a', x)\n", "several arguments"),
    ("def f(x: float) -> float:\n    raise\n", "bare `raise` outside an except block"),
    ("import numpy as np\nfrom qtrans.typing import Vec\n\ndef f(v: Vec, i: int) -> float:\n    try:\n"
     "        return v[i] / v[0]\n    except ZeroDivisionError:\n        return 0.0\n", None),
])
def test_refusals_and_numpy(src, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    if words is None:  # v[0] is a NumPy number: no ZeroDivisionError in Python either, plain division
        assert not bad and "qt::py::truediv" not in r.header
        return
    reasons = [f.reason for f in bad] + r.report.module_notes
    assert any(words in x for x in reasons), r.report.to_text()
