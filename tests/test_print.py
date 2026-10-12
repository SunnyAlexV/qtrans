"""qt::py (qtrans_print.hpp): what a translated program prints is what Python and NumPy print.

One compiled driver formats thousands of values the way print(), str(), repr(), f-strings and
format() would, and every result is compared with Python's own, character for character:
floats (shortest repr, every presentation type of the format mini-language, grouping, padding,
signs, z, #), ints (bases, grouping, c), strings (quoting), round(x, ndigits) (the exact decimal
value, half to even), and NumPy arrays (str and repr of vectors and matrices, the precision
NumPy picks, scientific switching, line wrapping, summarised arrays, nan/inf, empty arrays)."""
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qtrans.verify import find_compiler, find_eigen
from tests.test_semantics import HAVE_CXX

pytestmark = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
RUNTIME = Path(__file__).resolve().parents[1] / "qtrans" / "runtime"

DRIVER = r'''
#include <Eigen/Dense>
#include "qtrans_print.hpp"
#include <iostream>
#include <string>
using Vec = Eigen::Array<double, Eigen::Dynamic, 1>;
using Mat = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;
static std::string unhex(const std::string& h) {
    std::string s;
    for (size_t i = 1; i + 1 < h.size(); i += 2) s += static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16));
    return s;
}
int main() {
    std::string kind;
    while (std::cin >> kind) {
        std::string out;
        try {
            if (kind == "rf" || kind == "ri" || kind == "rn") {  // round(x, n): Python's, an int's, NumPy's
                std::string val;
                long long n;
                std::cin >> val >> n;
                const double x = std::strtod(val.c_str(), nullptr);
                out = kind == "rf" ? qt::py::repr(qt::py::round(x, n))
                    : kind == "rn" ? qt::py::repr(qt::py::round_numpy(x, n))
                                   : std::to_string(qt::py::round(std::stoll(val), n));
            } else if (kind == "f" || kind == "i" || kind == "s") {
                std::string val, spec_hex;
                std::cin >> val >> spec_hex;
                const std::string spec = unhex(spec_hex);
                if (kind == "f") { double x = std::strtod(val.c_str(), nullptr); out = spec == "@repr" ? qt::py::repr(x) : qt::py::format(x, spec); }
                else if (kind == "i") { long long x = std::stoll(val); out = qt::py::format(x, spec); }
                else { std::string x = unhex(val); out = spec == "@repr" ? qt::py::repr(x) : qt::py::format(x, spec); }
            } else {  // "as"/"ar": array str/repr; rows cols (cols < 0: a vector), then the values
                long r, c;
                std::cin >> r >> c;
                const bool s = kind == "as";
                if (c < 0) { Vec v(r); for (long i = 0; i < r; ++i) { std::string t; std::cin >> t; v(i) = std::strtod(t.c_str(), nullptr); }
                    out = s ? qt::py::str(v) : qt::py::repr(v); }
                else { Mat m(r, c); for (long i = 0; i < r; ++i) for (long j = 0; j < c; ++j) { std::string t; std::cin >> t; m(i, j) = std::strtod(t.c_str(), nullptr); }
                    out = s ? qt::py::str(m) : qt::py::repr(m); }
            }
        } catch (const std::exception&) { out = "!ERR"; }
        std::cout << "[";
        for (unsigned char ch : out) { char b[3]; std::snprintf(b, 3, "%02x", ch); std::cout << b; }
        std::cout << "]\n";
    }
}
'''


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    d = tmp_path_factory.mktemp("print")
    src = d / "driver.cpp"
    src.write_text(DRIVER, encoding="utf-8")
    exe = d / ("driver.exe" if sys.platform == "win32" else "driver")
    cxx = find_compiler()
    r = subprocess.run(cxx.command(src, exe, str(RUNTIME), system_includes=[str(find_eigen())]),
                       capture_output=True, text=True, cwd=d)
    assert r.returncode == 0, (r.stderr or r.stdout)[-3000:]

    def run(lines):
        out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True).stdout
        got = [bytes.fromhex(x[1:-1]).decode("utf-8", "backslashreplace") for x in out.split("\n") if x.startswith("[")]
        assert len(got) == len(lines)
        return got
    return run


def _differences(cases, got, expect):
    bad = [(c, g, e) for c, g, e in zip(cases, got, expect) if g != e]
    return "\n".join(f"{c!r}\n  C++:    {g!r}\n  Python: {e!r}" for c, g, e in bad[:10]), len(bad)


SPECS_F = ["", ".2f", ".0f", ".3e", "e", "f", "g", ".3g", ".10g", ".3", ".1", "10.4f", "<12.3f", "^15.2e", "+.2f",
           " .2f", ",.2f", "_.1f", ",", "%", ".1%", "08.2f", "+010.3f", "z.1f", "#.0f", "#g", ".0e", "E", "G", ".2F",
           "*>12,.2f", "015,.2f", ".17g", ".20f", "#.3g", ".0%", "=+12.4f", "n", ".5n", "012", "12", ".12", "#.2",
           "d", "x", "c", "s", "10.3s"]  # the last four are errors in Python, and must be in C++
SPECS_I = ["", "d", "5d", "<5d", "^7d", "+d", " d", ",", "_", "08d", "010,", "b", "#b", "o", "#o", "x", "#X",
           "#010x", "_x", ".2f", "e", "%", "g", ",.1f", "c", "=+8d", "-^9", ".3", "s", "z"]
SPECS_S = ["", "s", "10", "<10", ">10", "^10", "*^11", ".3", "10.2", "@repr", "d", "=10", "+", "010", "0", "05", ",", "#", "z", "x>08"]
STRINGS = ["", "abc", "it's", 'say "hi"', "both ' and \"", "tab\there", "line\nbreak", "back\\slash", "naïve café",
           "€uro", "\x01ctl", "\x7f", "日本"]
EDGE = [0.0, -0.0, 1.0, 0.1, 0.5, 1e16, 1e15, 1e-4, 1e-5, 2.0 ** 53, 123456789.0, 0.125, 0.375, 2.5, float("inf"),
        -float("inf"), float("nan"), 5e-324, 1.7976931348623157e308, 9.999999999999999e22, 1e23, 0.3, 2 / 3, 1e22,
        -1.5, 0.05, 0.15, 0.25, 1234.5]


def _float(rng):
    k = rng.random()
    if k < 0.2:
        return struct.unpack("d", struct.pack("Q", rng.getrandbits(64)))[0]  # any bit pattern
    if k < 0.4:
        return rng.uniform(-1e6, 1e6)
    if k < 0.5:
        return float(rng.randint(-10 ** 6, 10 ** 6)) / 10 ** rng.randint(0, 6)
    if k < 0.6:
        return rng.choice(EDGE)
    if k < 0.8:
        return rng.uniform(-1, 1) * 10.0 ** rng.randint(-30, 30)
    return round(rng.uniform(-100, 100), rng.randint(0, 4))


def _python(v, spec):
    try:
        return repr(v) if spec == "@repr" else format(v, spec)
    except (ValueError, TypeError, OverflowError):
        return "!ERR"


@pytest.mark.parametrize("seed", [1, 2])
def test_scalars_print_as_python_prints_them(driver, seed):
    rng = random.Random(seed)
    cases = [("f", x, s) for x in EDGE for s in SPECS_F + ["@repr"]]
    cases += [("f", _float(rng), rng.choice(SPECS_F + ["@repr"] * 8)) for _ in range(4000)]
    for _ in range(1500):
        v = rng.choice([rng.randint(-10 ** 18, 10 ** 18), rng.randint(-1000, 1000), rng.randint(0, 0x10FFFF),
                        rng.choice([0, -1, 2 ** 63 - 1, -2 ** 63, 1000, -1000, 255])])
        cases.append(("i", v, rng.choice(SPECS_I)))
    cases += [("s", s, sp) for s in STRINGS for sp in SPECS_S]
    lines = []
    for kind, v, sp in cases:
        val = repr(v) if kind == "f" else str(v) if kind == "i" else "S" + v.encode().hex()
        lines.append(f"{kind} {val} S{sp.encode().hex()}")
    got = driver(lines)
    text, n = _differences(cases, got, [_python(v, sp) for _, v, sp in cases])
    assert n == 0, f"{n} of {len(cases)} differ:\n{text}"


def _array(rng, R):
    shape = R.choice([(R.randint(0, 12),), (R.randint(0, 1100),), (R.randint(0, 6), R.randint(0, 6)),
                      (R.randint(1, 40), R.randint(1, 40)), (R.randint(30, 40), R.randint(30, 40))])
    n = int(np.prod(shape))
    kind = R.random()
    if kind < 0.2:
        a = rng.standard_normal(n)
    elif kind < 0.35:
        a = rng.standard_normal(n) * 10.0 ** R.randint(-12, 12)
    elif kind < 0.5:
        a = np.round(rng.uniform(-100, 100, n), R.randint(0, 3))
    elif kind < 0.6:
        a = rng.integers(-1000, 1000, n).astype(float)
    elif kind < 0.7:
        a = np.exp(rng.uniform(-30, 30, n))
    elif kind < 0.8:
        a = np.frombuffer(rng.integers(0, 2 ** 63, n, dtype=np.int64).tobytes(), dtype=np.float64).copy()
    else:
        a = rng.standard_normal(n)
        for _ in range(max(1, n // 5)):
            if n:
                a[R.randrange(n)] = R.choice([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-300, 1e300, 0.5, 1e8, 9.99999999e7])
    return a.reshape(shape)


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < "2.0.0",
                    reason="qtrans prints as NumPy 2 does (NumPy 1 writes reprs differently)")
def test_arrays_print_as_numpy_prints_them(driver):
    rng, R = np.random.default_rng(3), random.Random(3)
    arrays = [_array(rng, R) for _ in range(600)]
    arrays += [np.array(x) for x in ([1.0], [0.0], [-0.0], [np.nan], [np.inf, -np.inf], [1e-5, 1], [0.1, 0.2, 0.3],
                                     [1e16], [123456789.0], [99999999.0], [100000000.0], [1 / 3, 2 / 3], [], [1e-300])]
    arrays += [np.zeros((0, 3)), np.zeros((2, 0)), np.eye(3), np.arange(1200.0), np.arange(1200.0).reshape(30, 40)]
    lines, cases, expect = [], [], []
    for a in arrays:
        r, c = (a.shape[0], -1) if a.ndim == 1 else a.shape
        values = " ".join(repr(float(x)) for x in a.ravel())
        for kind, f in (("as", str), ("ar", repr)):
            lines.append(f"{kind} {r} {c} {values}")
            cases.append((kind, a.shape))
            expect.append(f(a))
    got = driver(lines)
    text, n = _differences(cases, got, expect)
    assert n == 0, f"{n} of {len(cases)} differ:\n{text}"


@pytest.mark.parametrize("seed", [4, 5])
def test_round_to_ndigits_as_python_rounds(driver, seed):
    R = random.Random(seed)
    ties = [0.5, 1.5, 2.5, 2.675, 0.125, 0.375, 1250.0, 1350.0, 5e-324, 1.7976931348623157e308, -0.0, float("inf"),
            float("nan"), 0.05, 0.15, 0.25, 0.35, 1e22, 1e23, 4.0, 5.0, 6.0, 50.0, 0.0005, -4.0, -5.0, -0.5]
    cases = []
    for _ in range(12000):
        k = R.random()
        x = (struct.unpack("d", struct.pack("Q", R.getrandbits(64)))[0] if k < 0.3 else
             R.uniform(-1, 1) * 10 ** R.randint(-30, 30) if k < 0.6 else
             round(R.uniform(-1000, 1000), R.randint(0, 6)) + R.choice([0, 0.5, 0.05, 0.005, 0.125]) if k < 0.8 else
             R.choice(ties))
        n = R.choice([0, 1, 2, 3, 5, 10, 17, 300, 323, 324, -1, -2, -3, -10, -22, -24, -300, -308, -309]) \
            if R.random() < 0.5 else R.randint(-25, 25)
        cases.append(("rf", x, n))
        if -40 < n < 40:
            cases.append(("rn", x, n))  # round() of a NumPy float64
    for _ in range(4000):
        v = R.choice([R.randint(-10 ** 18, 10 ** 18), R.randint(-100000, 100000),
                      R.choice([1250, 1350, -1250, 15, 25, -25, 5, 0, -5])])
        cases.append(("ri", v, R.randint(-20, 3)))

    def python(kind, x, n):
        try:
            with np.errstate(all="ignore"):
                return _round(kind, x, n)
        except OverflowError:
            return "!ERR"

    def _round(kind, x, n):
        v = float(round(np.float64(x), n)) if kind == "rn" else round(x, n)
        return repr(v) if kind in ("rf", "rn") else str(v) if abs(v) < 2 ** 63 else "!ERR"
    got = driver([f"{k} {x!r} {n}" for k, x, n in cases])
    text, bad = _differences(cases, got, [python(*c) for c in cases])
    assert bad == 0, f"{bad} of {len(cases)} differ:\n{text}"
