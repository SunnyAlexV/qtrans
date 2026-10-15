"""NumPy's float64 maths in the runtime (qtrans_npmath.hpp, 1.1): bit for bit against the running
NumPy, on whatever platform and CPU the tests run on (tools/npmath_check.py does the work;
tools/npmath_versions.py runs it under each NumPy release from 1.21 to 2.5)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qtrans import translate_source  # noqa: E402
from qtrans import npmath_check  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return npmath_check.check(n=60_000, seed=3, verbose=False)


def test_the_cpp_runs_what_numpy_runs(result):
    want, got = result["paths"]
    assert {k: got.get(k) for k in want} == want


def test_bit_identical_where_numpy_runs_its_own_code(result):
    assert result["failures"] == []
    exact = [c for c in result["cases"] if c[3]]
    if npmath_check.SAME_LIBM:  # the same C library on both sides: every case is required
        assert len(exact) == len(result["cases"])


def test_numpy_version_number():
    from qtrans.npfacts import generate, version_number
    assert version_number("2.5.4") == 20504
    assert version_number("1.26.0rc1") == 12600
    assert version_number("2.10.1") == 21001
    assert f"#define QTRANS_NUMPY_VERSION {version_number(np.__version__)}" in generate()


def code_of(src: str) -> str:
    return translate_source(src, "m").header


def test_numpy_maths_on_numbers_is_numpy_s():
    h = code_of("import math\nimport numpy as np\n\ndef f(x: float) -> float:\n"
                "    return np.exp(x) + math.exp(x) + np.tanh(x) + np.arctan2(x, 2.0) + np.cbrt(x)\n")
    assert "qt::npm::exp(x)" in h and "qt::py::math_exp(x)" in h  # np.exp is NumPy's, math.exp the C library's
    assert "qt::npm::tanh(x)" in h and "qt::npm::arctan2(x, 2.0)" in h and "qt::npm::cbrt(x)" in h
    assert '#include "qtrans_npmath.hpp"' in h


def test_powers_follow_numpy():
    h = code_of("import numpy as np\nfrom qtrans.typing import Vec\n\n"
                "def f(v: Vec, s: float) -> Vec:\n"
                "    return v ** 2 + v ** 0.5 + v ** s + np.power(v, 2.0) + np.power(v, s) + v ** v\n")
    assert "v.square()" in h and "v.sqrt()" in h  # the ** operator's shortcuts: every NumPy release
    assert "qt::pow(v, s)" in h  # decided at run time, as NumPy does
    assert "qt::np_power(v, 2.0)" in h and "qt::np_power(v, s)" in h  # np.power's depend on the release
    assert "qt::pow(v, v)" in h


def test_power_shortcut_rules_by_release():
    """The power loop's shortcuts, as measured (tools/npmath_check.py checks them against NumPy)."""
    hpp = (ROOT / "qtrans" / "runtime" / "qtrans_npmath.hpp").read_text()
    assert "numpy_version >= 20300" in hpp and "numpy_version >= 20100 && s == 2.0" in hpp


def test_svml_headers_are_numpy_s():
    """The vendored assembly says where it came from, and carries NumPy's licence."""
    for v in ("ha", "la"):
        text = (ROOT / "qtrans" / "runtime" / f"qtrans_svml_{v}.hpp").read_text()
        assert "Written by tools/vendor_svml.py from numpy" in text
        assert "Redistribution and use in source and binary forms" in text
        assert text.count("__asm__(R\"QTSVML(") == (19 if v == "ha" else 22)


def test_pairwise_sum_is_numpys_to_the_bit(tmp_path):
    """np.sum's pairwise summation: the vectorised form (eight partial sums in one register, 1.1)
    adds in NumPy's order, so every length gives NumPy's bits."""
    import struct
    import subprocess
    from qtrans.translator import NPMATH_HEADERS, ARRAYS_HEADER, RUNTIME_HEADER, PYTHON_HEADER
    from qtrans.verify import find_compiler, find_eigen
    if sys.platform == "win32":
        pytest.skip("the probe reads binary stdin; covered on Linux and macOS")
    for h in (ARRAYS_HEADER, RUNTIME_HEADER, PYTHON_HEADER, *NPMATH_HEADERS):
        (tmp_path / h.name).write_bytes(h.read_bytes())
    (tmp_path / "s.cpp").write_text('#include "qtrans_arrays.hpp"\n#include <cstdio>\n'
                                    "int main() { long n; while (std::fread(&n, 8, 1, stdin) == 1) { qt::Vec v(n);\n"
                                    "  if (n && std::fread(v.data(), 8, (size_t)n, stdin) != (size_t)n) return 1;\n"
                                    "  double s = qt::sum(v); std::fwrite(&s, 8, 1, stdout); } }\n")
    exe = tmp_path / "s"
    subprocess.run(find_compiler().command(tmp_path / "s.cpp", exe, tmp_path, [find_eigen()]), check=True, capture_output=True)
    rng = np.random.default_rng(9)
    arrs = [rng.standard_normal(n) * 10 ** rng.uniform(-5, 5, n) for n in range(0, 600)]
    arrs += [rng.standard_normal(n) for n in (4096, 100_000, 123_457)]
    data = b"".join(struct.pack("<q", len(a)) + a.tobytes() for a in arrs)
    got = np.frombuffer(subprocess.run([str(exe)], input=data, capture_output=True).stdout, dtype=np.float64)
    ref = np.array([np.sum(a) for a in arrs])
    assert (got.view(np.uint64) == ref.view(np.uint64)).all()
