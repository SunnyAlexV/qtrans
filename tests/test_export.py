"""`qtrans export`: a C++ package (CMake target, headers, programs, golden tests). Where CMake is
installed the package is configured, built and tested for real, a wrong value is shown to fail
its golden test, and a separate project uses it through find_package."""
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from qtrans.export import export
from qtrans.translator import translate_source
from qtrans.verify import find_compiler, find_eigen
from tests.test_semantics import HAVE_CXX

HAVE_CMAKE = shutil.which("cmake") is not None and HAVE_CXX
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

SCRIPT = '''
import numpy as np

from qtrans.typing import Vec


def discount(r: float, t: float) -> float:
    return float(np.exp(-r * t))


def weights(v: Vec) -> Vec:
    return v / v.sum()


prices = {"AAPL": 190.5, "MSFT": 410.25}
for name, p in prices.items():
    print(name, round(p * discount(0.05, 1.0), 4))
print(weights(np.array([1.0, 3.0])))
'''


@pytest.fixture
def script(tmp_path):
    f = tmp_path / "pricer.py"
    f.write_text(textwrap.dedent(SCRIPT).lstrip(), encoding="utf-8")
    return f


def test_the_package_has_what_a_cpp_team_needs(script, tmp_path):
    out = tmp_path / "pkg"
    lines = export(script, out, cases=10, build=False)
    assert lines[0].startswith(f"wrote {out}: target qtrans::pricer, 1 module(s), 1 program(s)")
    for f in ("CMakeLists.txt", "README.md", "include/pricer/pricer.hpp", "include/pricer/qtrans_arrays.hpp",
              "src/pricer.cpp", "tests/golden.cpp", "tests/golden.txt", "tests/qtrans_driver.cpp",
              "tests/pricer.expected", "tests/pricer.status", "tests/compare_output.cmake", "reports/pricer_report.md"):
        assert (out / f).is_file(), f
    cm = (out / "CMakeLists.txt").read_text()
    assert "add_library(qtrans::pricer ALIAS pricer)" in cm and "cxx_std_17" in cm
    assert "-ffp-contract=off" in cm and "/fp:precise" in cm  # exactness travels with the target
    assert "find_package(Eigen3 3.3 REQUIRED NO_MODULE)" in cm and "PRICER_EIGEN_DIR" in cm
    assert "add_executable(pricer_pricer src/pricer.cpp)" in cm and "add_test(NAME pricer_pricer_output" in cm
    assert "install(EXPORT pricerTargets NAMESPACE qtrans::" in cm
    g = (out / "tests/golden.txt").read_text()
    assert "# 20 cases" in g and "CASE discount 1e-10 1e-13" in g and "@0x" in g
    assert (out / "tests/pricer.expected").read_text().startswith("AAPL 181.2092\nMSFT 390.2419\n[0.25 0.75]\n")
    assert "sha256" in (out / "include/pricer/pricer.hpp").read_text().splitlines()[0]


def test_every_header_refuses_fast_math():
    h = translate_source("def f(x: float) -> float:\n    return x * 2.0\n", "m").header
    assert '#if defined(__FAST_MATH__) || defined(_M_FP_FAST)\n#error "built with fast-math' in h


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_fast_math_really_stops_the_build(tmp_path):
    cxx = find_compiler(None)
    if cxx.kind == "msvc":
        pytest.skip("checked with GCC / Clang flags")
    (tmp_path / "m.hpp").write_text(translate_source("def f(x: float) -> float:\n    return x * 2.0\n", "m").header)
    (tmp_path / "use.cpp").write_text('#include "m.hpp"\nint main() { return m::f(1.0) > 0 ? 0 : 1; }\n')
    for flags, ok in ((["-ffp-contract=off"], True), (["-ffast-math"], False)):
        p = subprocess.run([cxx.path, "-std=c++17", *flags, "-I", str(tmp_path), str(tmp_path / "use.cpp"),
                            "-o", str(tmp_path / "use")], capture_output=True, text=True)
        assert (p.returncode == 0) == ok, p.stderr
        if not ok:
            assert "changes floating-point results" in p.stderr


@pytest.mark.skipif(not HAVE_CMAKE, reason="needs CMake and a C++ compiler")
def test_the_package_builds_and_its_golden_tests_pass_and_can_fail(script, tmp_path):
    out = tmp_path / "pkg"
    lines = export(script, out, cases=10)
    assert lines[-1] == ("built and tested with CMake here: 2 of 2 ctest tests passed (20 of 20 cases agree "
                         "with Python; 1 program(s) print exactly what Python printed)"), lines
    # one recorded Python result off by 1e-8 (relative): the golden test must say so
    g = out / "tests/golden.txt"
    text = g.read_text().split("\n")
    k = next(i for i, l in enumerate(text) if l.startswith("OK 0x") and "CALL discount" in text[i - 1])
    value, scale = text[k][3:].split("@")
    text[k] = f"OK {float.hex(float.fromhex(value) * (1 + 1e-8))}@{scale}"
    g.write_text("\n".join(text))
    expected = out / "tests/pricer.expected"
    expected.write_text(expected.read_text().replace("AAPL", "AAPX"))
    from qtrans.export import build_and_test
    res = build_and_test(out, "pricer", True)
    assert res[0] == "CMake test FAILED:", res
    tail = "\n".join(res)
    assert "golden: 19 of 20 cases agree with Python" in tail and "differs from Python's" in tail


@pytest.mark.skipif(not HAVE_CMAKE, reason="needs CMake and a C++ compiler")
def test_a_project_installs_and_another_project_uses_it(tmp_path):
    out = tmp_path / "bl"
    lines = export(EXAMPLES / "bl_app", out, name="bl_app", cases=5)
    assert lines[0].startswith(f"wrote {out}: target qtrans::bl_app, 4 module(s)")
    assert "ctest tests passed" in lines[-1] and "cases agree with Python" in lines[-1], lines
    b, prefix = tmp_path / "b", tmp_path / "prefix"
    eigen = f"-DBL_APP_EIGEN_DIR={find_eigen()}"
    for cmd in (["cmake", "-S", str(out), "-B", str(b), eigen, "-DBL_APP_BUILD_TESTS=OFF"],
                ["cmake", "--build", str(b), "--config", "Release"],
                ["cmake", "--install", str(b), "--prefix", str(prefix), "--config", "Release"]):
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
    cons = tmp_path / "consumer"
    cons.mkdir()
    (cons / "CMakeLists.txt").write_text(textwrap.dedent(f"""
        cmake_minimum_required(VERSION 3.16)
        project(consumer CXX)
        set(BL_APP_EIGEN_DIR "{find_eigen().as_posix()}")
        find_package(bl_app REQUIRED)
        add_executable(use use.cpp)
        target_link_libraries(use PRIVATE qtrans::bl_app)
        """))
    (cons / "use.cpp").write_text(textwrap.dedent("""
        #include <bl_app/bl_core.hpp>
        #include <cstdio>
        int main() {
            qt::Mat a(2, 2);
            a << 1.0, 0.9, 0.9, 1.0;
            std::printf("%.3f\\n", bl_core::nearest_psd(a)(0, 1));
        }
        """))
    cb = tmp_path / "cb"
    p = subprocess.run(["cmake", "-S", str(cons), "-B", str(cb), f"-DCMAKE_PREFIX_PATH={prefix}"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    p = subprocess.run(["cmake", "--build", str(cb), "--config", "Release"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    exe = next(x for x in cb.rglob("use*") if x.is_file() and x.suffix in ("", ".exe") and x.stat().st_mode & 0o100
               or x.suffix == ".exe")
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.stdout.strip() == "0.900"
