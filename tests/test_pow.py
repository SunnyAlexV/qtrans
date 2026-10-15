"""x ** y on Python floats is the C library's pow, which for the exponents 2 and -1 is not always
x * x and 1 / x (glibc: about 1 value in 1,000), while a C++ compiler turns std::pow(x, 2.0) into
x * x and std::pow(x, -1.0) into 1.0 / x. The translation calls qt::pow, which keeps the call, and
np.power is NumPy's power loop, with the shortcuts of the NumPy release (qtrans_npmath.hpp)."""
import platform
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest

from qtrans import translate_source
from qtrans.translator import PYTHON_HEADER, RUNTIME_HEADER
from qtrans.verify import find_compiler
from tests.test_semantics import HAVE_CXX

compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
GLIBC = platform.libc_ver()[0] == "glibc"
EXPONENTS = [2.0, -1.0, 3.0, 0.5, -2.0, 1.5]

PROBE = r"""
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "qtrans_runtime.hpp"
int main() {
    std::string a, b;
    while (std::cin >> a >> b) {
        const double x = std::strtod(a.c_str(), nullptr);
        std::printf("%a %a %a\n", qt::pow(x, 2.0), qt::pow(x, -1.0), qt::pow(x, std::strtod(b.c_str(), nullptr)));
    }
}
"""


@compiled
def test_qt_pow_is_pythons_power(tmp_path):
    for h in (RUNTIME_HEADER, PYTHON_HEADER):
        shutil.copy(h, tmp_path / h.name)
    (tmp_path / "p.cpp").write_text(PROBE, encoding="utf-8")
    exe = tmp_path / ("p.exe" if sys.platform == "win32" else "p")
    subprocess.run(find_compiler().command(tmp_path / "p.cpp", exe, tmp_path), check=True, capture_output=True)
    g = random.Random(7)
    xs = [1.431322935583394] + [g.uniform(0, 10) for _ in range(20000)] + [-g.uniform(0, 10) for _ in range(2000)]
    ys = [g.choice(EXPONENTS if x > 0 else [2.0, -1.0, 3.0, -2.0]) for x in xs]  # (x < 0, y = 0.5: complex)
    out = subprocess.run([str(exe)], input="".join(f"{x.hex()} {y.hex()}\n" for x, y in zip(xs, ys)),
                         capture_output=True, text=True, check=True).stdout.split()
    got = [float.fromhex(t) for t in out]
    want = [v for x, y in zip(xs, ys) for v in (x ** 2.0, x ** -1.0, x ** y)]
    bad = [(i, w, c) for i, (w, c) in enumerate(zip(want, got)) if w.hex() != c.hex()]
    assert not bad, f"{len(bad)} of {len(want)} differ: {bad[:3]}"
    if GLIBC:  # the case this guards against is really there (other C libraries round pow differently)
        assert any((x ** 2) != x * x for x in xs)


def test_power_and_np_power_keep_their_own_rules():
    h = translate_source("import numpy as np\n\ndef f(x: float) -> float:\n    return x ** 2 + x ** -1\n\n"
                         "def g(x: float) -> float:\n    return np.power(x, 2) + np.power(x, 0.5) + np.power(x, -1)\n",
                         "m").header
    # (qt::py::power is qt::pow plus Python's errors; both terms may raise, so the left one first)
    assert "const double left = qt::py::power(x, 2.0);\n    return left + qt::py::power(x, -1.0);" in h
    # np.power is NumPy's power loop: its shortcuts follow the NumPy release (qtrans_npmath.hpp)
    assert "qt::npm::power(x, 2.0) + qt::npm::power(x, 0.5) + qt::npm::power(x, -1.0)" in h
    x = np.float64(1.431322935583394)
    if tuple(int(p) for p in np.__version__.split(".")[:2]) >= (2, 1):
        assert np.power(x, 2) == x * x  # NumPy's square, from 2.1
    if GLIBC:  # glibc's pow(x, 2.0) is not x * x here; macOS's and Windows's pow happen to agree for this x
        assert x ** 2 != x * x
