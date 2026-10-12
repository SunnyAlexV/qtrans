"""`qtrans export`: the translation as a C++ package a C++ team builds with their own tools.

    pricing_cpp/
      CMakeLists.txt         header-only target qtrans::pricing (+ install rules, package config)
      include/pricing/       the generated headers and the qtrans runtime headers they include
      src/                   programs (a script's int main()), each an executable target
      tests/golden.cpp       checks the C++ against Python's own results, recorded at export
      tests/golden.txt         time: no Python needed to run it (`ctest`)
      reports/               what translated, what didn't and why
      README.md

The target carries what exactness depends on: C++17, and floating-point arithmetic kept as
written (-ffp-contract=off; /fp:precise on MSVC), so a consumer's build can't fuse a*b + c into
one rounding and drift from Python. Every generated header also refuses to compile under
fast-math.

The golden tests are the cases `qtrans verify` found matching Python: the call's inputs and
Python's result (exact hex floats), compared with the tolerance verify used (arrays normwise).
Cases verify marked ill-conditioned or known-different, and calls whose printed text matched only
up to rounding, are left out: the file says how many.
"""
from __future__ import annotations

import hashlib
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import verify as V
from .translator import VERSION
from .types import BOOL, FLOAT, INT, RNG, STR, VOID, ArrayType, DictType, ListType, OptionalType, StructType, TupleType


def _source_hash(p) -> str:
    """As in the headers: SHA-256 of the source with Windows line ends as Unix ones."""
    return hashlib.sha256(Path(p).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def _cmake_name(name: str) -> str:
    n = re.sub(r"\W", "_", name)
    return ("m_" + n) if n[0].isdigit() else n


# --- Python's result, as the driver prints a result ----------------------------------------------
def _tokens(v, t, scale=None) -> list:
    """Python's value v of type t as the driver's output tokens; a float is written as
    hex@scale, the magnitude its tolerance is relative to (its own size, or an array's largest)."""
    def fl(x, s):
        x = float(x)
        s = abs(x) if s is None else s
        return f"{x.hex()}@{float(s).hex()}"
    if t == VOID:
        return []
    if isinstance(t, TupleType):
        return [tok for x, et in zip(v, t.elts) for tok in _tokens(x, et)]
    if isinstance(t, DictType):
        return [str(len(v))] + [tok for k, x in v.items() for tok in _tokens(k, t.key) + _tokens(x, t.val)]
    if isinstance(t, ListType):
        return [str(len(v))] + [tok for x in v for tok in _tokens(x, t.elem)]
    if isinstance(t, StructType):
        info, _ = V._CLASSES[t.name]
        return [tok for f in info.fields for tok in _tokens(v[f.name] if isinstance(v, dict) else getattr(v, f.name), f.type)]
    if isinstance(t, OptionalType):
        return ["N"] if v is None else _tokens(v, t.inner)
    if t == STR:
        return ["S" + str(v).encode("utf-8").hex()]
    if isinstance(t, ArrayType):
        a = np.asarray(v, dtype=np.float64)
        head = [str(a.shape[0])] if t.rank == 1 else [str(a.shape[0]), str(a.shape[1])]
        s = float(np.max(np.abs(a[np.isfinite(a)]))) if np.isfinite(a).any() else 0.0
        return head + [fl(x, s) for x in a.ravel()]
    if t == FLOAT:
        return [fl(v, scale)]
    if t == RNG and isinstance(v, np.random.Generator):
        # a generator inside an object: the driver prints its next raw output (as verify compares it)
        import copy
        return [str(int(copy.deepcopy(v).bit_generator.random_raw()))]
    if t in (INT, RNG):
        return [str(int(v))]
    if t == BOOL:
        return ["1" if v else "0"]
    raise TypeError(f"no golden encoding for {t}")


GOLDEN_CPP = r"""// golden.cpp: checks the C++ against Python's own results, recorded by `qtrans export`
// (tests/golden.txt). It runs qtrans's test driver over the recorded inputs and compares each
// result with Python's, with the tolerance qtrans verify used. No Python is needed.
//   usage: golden <golden.txt> <scratch file>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define main qtrans_driver_main
#include "qtrans_driver.cpp"
#undef main

namespace golden {
struct Case {
    std::string fn, in, expect, printed;
    double rtol = 0, atol = 0;
    int line = 0;
};

static std::vector<std::string> split(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    std::string t;
    while (in >> t) out.push_back(t);
    return out;
}

static bool number(const std::string& t, double& v) {
    if (t.empty()) return false;
    char* end = nullptr;
    v = std::strtod(t.c_str(), &end);
    return end && *end == '\0';
}

// one token of the C++ against Python's: equal, or numbers within tolerance of `scale`
static bool same(const std::string& cpp, const std::string& py, double rtol, double atol) {
    const auto at = py.find('@');
    if (at == std::string::npos) return cpp == py;  // an int, a length, a string, N
    double a, b, scale;
    if (!number(cpp, a) || !number(py.substr(0, at), b) || !number(py.substr(at + 1), scale)) return false;
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::fabs(a - b) <= atol + rtol * scale;
}
}  // namespace golden

int main(int argc, char** argv) {
    using namespace golden;
    if (argc != 3) {
        std::fprintf(stderr, "usage: golden <golden.txt> <scratch file>\n");
        return 2;
    }
    std::ifstream file(argv[1]);
    if (!file) {
        std::fprintf(stderr, "golden: can't read %s\n", argv[1]);
        return 2;
    }
    std::vector<Case> cases;
    std::string line;
    int n = 0;
    while (std::getline(file, line)) {
        ++n;
        if (line.empty() || line[0] == '#') continue;
        const std::string kind = line.substr(0, line.find(' '));
        const std::string rest = line.size() > kind.size() ? line.substr(kind.size() + 1) : "";
        if (kind == "CASE") {
            Case c;
            std::istringstream in(rest);
            in >> c.fn >> c.rtol >> c.atol;
            c.line = n;
            cases.push_back(c);
        } else if (!cases.empty() && kind == "IN") {
            cases.back().in = rest;
        } else if (!cases.empty() && kind == "PRINTED") {
            cases.back().printed = "PRINTED " + rest;
        } else if (!cases.empty()) {
            cases.back().expect = line;  // OK ... / EXC ...
        }
    }
    std::string input;
    for (const Case& c : cases) input += c.in + "\n";
    std::istringstream feed(input);
    std::cin.rdbuf(feed.rdbuf());
    if (!std::freopen(argv[2], "w", stdout)) {
        std::fprintf(stderr, "golden: can't write %s\n", argv[2]);
        return 2;
    }
    qtrans_driver_main();
    std::fflush(stdout);
    std::ifstream got(argv[2]);
    std::vector<std::string> results, printed;
    while (std::getline(got, line)) {
        if (line.rfind("PRINTED", 0) == 0 && !results.empty()) printed.back() = line;
        else { results.push_back(line); printed.push_back(""); }
    }
    if (results.size() != cases.size()) {
        std::fprintf(stderr, "golden: the driver gave %zu results for %zu cases\n", results.size(), cases.size());
        return 1;
    }
    std::size_t bad = 0;
    for (std::size_t k = 0; k < cases.size(); ++k) {
        const Case& c = cases[k];
        const auto a = split(results[k]), b = split(c.expect);
        bool ok = a.size() == b.size() && printed[k] == c.printed;
        for (std::size_t i = 0; ok && i < a.size(); ++i) ok = same(a[i], b[i], c.rtol, c.atol);
        if (!ok && ++bad <= 20)
        {
            // the arguments: the input after "CALL <function> "
            const std::string args = c.in.size() > 6 + c.fn.size() ? c.in.substr(6 + c.fn.size()) : "";
            std::fprintf(stderr, "golden.txt line %d: %s(%s)\n  C++:    %s\n  Python: %s\n", c.line, c.fn.c_str(),
                         args.c_str(), results[k].c_str(), c.expect.c_str());
        }
    }
    std::fprintf(stderr, "golden: %zu of %zu cases agree with Python\n", cases.size() - bad, cases.size());
    return bad ? 1 : 0;
}
"""


COMPARE_CMAKE = r"""# compare_output.cmake: runs a program and checks it prints exactly what Python printed, and
# exits with the same status (recorded by `qtrans export`).
execute_process(COMMAND ${PROGRAM} OUTPUT_VARIABLE out RESULT_VARIABLE status)
file(READ ${EXPECTED} expected)
file(STRINGS ${STATUS} want LIMIT_COUNT 1)
string(REPLACE "
" "
" out "${out}")
if(NOT out STREQUAL expected)
    file(WRITE ${PROGRAM}.out "${out}")
    message(FATAL_ERROR "the output differs from Python's: compare ${PROGRAM}.out with ${EXPECTED}")
endif()
if(NOT status EQUAL want)
    message(FATAL_ERROR "exit status ${status}; Python's was ${want}")
endif()
message(STATUS "same output as Python, character for character, and the same exit status (${status})")
"""


def _cmake(name, modules, programs, uses_eigen, source_files) -> str:
    P = name.upper()
    lines = [
        "# Generated by `qtrans export` " + VERSION + " from " + ", ".join(source_files) + ".",
        "# Re-export rather than editing: the Python is the source.",
        "cmake_minimum_required(VERSION 3.16)",
        f"project({name} VERSION 0.1.0 LANGUAGES CXX)",
        "",
        f"add_library({name} INTERFACE)",
        f"add_library(qtrans::{name} ALIAS {name})",
        f"target_compile_features({name} INTERFACE cxx_std_17)",
        f"target_include_directories({name} INTERFACE",
        "    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>",
        "    $<INSTALL_INTERFACE:include>)",
        "# The numbers match Python's only if each floating-point operation is rounded as written:",
        "# no fused multiply-add contraction (a*b + c in one rounding), no fast-math.",
        f"target_compile_options({name} INTERFACE",
        "    $<$<CXX_COMPILER_ID:MSVC>:/fp:precise /bigobj>",
        "    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-ffp-contract=off>)",
    ]
    if uses_eigen:
        lines += [
            "",
            "# Eigen (the arrays): its CMake package, or a folder holding Eigen/Dense",
            f"set({P}_EIGEN_DIR \"\" CACHE PATH \"folder containing Eigen/Dense, if find_package(Eigen3) can't find it\")",
            "# (here, for the build tree; an installed package finds Eigen again where it is used:",
            f"# {name}Config.cmake, with {P}_EIGEN_DIR or find_package(Eigen3))",
            f"if({P}_EIGEN_DIR)",
            f"    target_include_directories({name} SYSTEM INTERFACE $<BUILD_INTERFACE:${{{P}_EIGEN_DIR}}>)",
            "else()",
            "    find_package(Eigen3 3.3 REQUIRED NO_MODULE)",
            f"    target_link_libraries({name} INTERFACE $<BUILD_INTERFACE:Eigen3::Eigen>)",
            "endif()",
        ]
    for prog in programs:
        t = f"{name}_{_cmake_name(prog)}"
        lines += ["", f"# the program {prog}.py: its top-level code as int main()",
                  f"add_executable({t} src/{prog}.cpp)",
                  f"target_link_libraries({t} PRIVATE {name})",
                  f"target_include_directories({t} PRIVATE include/{name})"]
    lines += [
        "",
        "# golden tests: the C++ against Python's own results, recorded at export (no Python needed)",
        "if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)",
        "    set(_qtrans_top ON)",
        "else()",
        "    set(_qtrans_top OFF)  # inside another project: its tests are off unless asked for",
        "endif()",
        f"option({P}_BUILD_TESTS \"build the golden tests\" ${{_qtrans_top}})",
        f"if({P}_BUILD_TESTS)",
        "    enable_testing()",
        "  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden.txt)",
        f"    add_executable({name}_golden tests/golden.cpp)",
        f"    target_link_libraries({name}_golden PRIVATE {name})",
        f"    target_include_directories({name}_golden PRIVATE include/{name} tests)",
        f"    add_test(NAME {name}_golden COMMAND {name}_golden",
        "             ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden.txt ${CMAKE_CURRENT_BINARY_DIR}/golden_out.txt)",
        "  endif()",
    ] + [line for prog in programs for line in (
        f"    if(EXISTS ${{CMAKE_CURRENT_SOURCE_DIR}}/tests/{prog}.expected)  # the program prints what Python printed",
        f"        add_test(NAME {name}_{_cmake_name(prog)}_output COMMAND ${{CMAKE_COMMAND}}",
        f"                 -DPROGRAM=$<TARGET_FILE:{name}_{_cmake_name(prog)}>",
        f"                 -DEXPECTED=${{CMAKE_CURRENT_SOURCE_DIR}}/tests/{prog}.expected",
        f"                 -DSTATUS=${{CMAKE_CURRENT_SOURCE_DIR}}/tests/{prog}.status",
        "                 -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_output.cmake)",
        "    endif()")] + [
        "endif()",
        "",
        "# install: headers, the target, and a package config for find_package(" + name + ")",
        "include(GNUInstallDirs)",
        "include(CMakePackageConfigHelpers)",
        "install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})",
        f"install(TARGETS {name} EXPORT {name}Targets)",
        f"install(EXPORT {name}Targets NAMESPACE qtrans:: DESTINATION ${{CMAKE_INSTALL_LIBDIR}}/cmake/{name})",
        f"file(WRITE ${{CMAKE_CURRENT_BINARY_DIR}}/{name}Config.cmake [=[",
        "include(CMakeFindDependencyMacro)",
        f"include(${{CMAKE_CURRENT_LIST_DIR}}/{name}Targets.cmake)",
    ] + ([
        f"# Eigen, where this package is used: a folder ({P}_EIGEN_DIR) or Eigen's CMake package",
        f"if({P}_EIGEN_DIR)",
        f"    set_property(TARGET qtrans::{name} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${{{P}_EIGEN_DIR}})",
        f"    set_property(TARGET qtrans::{name} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${{{P}_EIGEN_DIR}})",
        "else()",
        "    find_dependency(Eigen3 3.3 NO_MODULE)",
        f"    set_property(TARGET qtrans::{name} APPEND PROPERTY INTERFACE_LINK_LIBRARIES Eigen3::Eigen)",
        "endif()",
    ] if uses_eigen else []) + [
        "]=])",
        f"write_basic_package_version_file(${{CMAKE_CURRENT_BINARY_DIR}}/{name}ConfigVersion.cmake",
        "                                 COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)",
        f"install(FILES ${{CMAKE_CURRENT_BINARY_DIR}}/{name}Config.cmake ${{CMAKE_CURRENT_BINARY_DIR}}/{name}ConfigVersion.cmake",
        f"        DESTINATION ${{CMAKE_INSTALL_LIBDIR}}/cmake/{name})",
    ]
    return "\n".join(lines) + "\n"


def export(paths, out_dir, name=None, cases=50, seed=0, golden=True, build=True, compiler=None, log=print):
    """Write the package; with golden=True first verify (and record the matching cases), with
    build=True then configure, build and run its tests with CMake. -> the report text lines."""
    from .project import translate_project
    if isinstance(paths, (str, Path)) and Path(paths).is_dir():  # a folder: its modules, as translate does
        paths = sorted(q for q in Path(paths).glob("*.py") if not q.name.startswith(("test_", "_"))
                       and q.name not in ("conftest.py", "qtrans_cases.py", "setup.py"))
    paths = [Path(paths).resolve()] if isinstance(paths, (str, Path)) else [Path(p).resolve() for p in paths]
    out = Path(out_dir).resolve()
    first = paths[0]
    name = _cmake_name(name or first.stem)
    if out.exists() and any(out.iterdir()):
        if not (out / "CMakeLists.txt").exists():
            raise RuntimeError(f"{out} exists and isn't a qtrans export: choose an empty folder")
        for sub in ("include", "src", "tests", "reports"):
            shutil.rmtree(out / sub, ignore_errors=True)
    stage = out / "_stage"
    shutil.rmtree(stage, ignore_errors=True)
    project = translate_project(paths, stage)
    inc = out / "include" / name
    inc.mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)
    programs = []
    for f in stage.iterdir():
        if f.suffix == ".hpp":
            shutil.copy(f, inc / f.name)
        elif f.name.endswith("_report.md"):
            shutil.copy(f, out / "reports" / f.name)
        elif f.suffix == ".cpp":
            (out / "src").mkdir(exist_ok=True)
            # a program includes the runtime headers by name: they live in include/<name>/
            shutil.copy(f, out / "src" / f.name)
            programs.append(f.stem)
    shutil.rmtree(stage, ignore_errors=True)
    uses_eigen = project.uses_arrays or any("qtrans_arrays.hpp" in (r.program or "") for r in project.modules.values())
    modules = list(project.order)
    sources = [p.name for p in project.paths.values()]
    (out / "CMakeLists.txt").write_text(_cmake(name, modules, programs, uses_eigen, sources), encoding="utf-8")
    report = [f"wrote {out}: target qtrans::{name}, {len(modules)} module(s)"
              + (f", {len(programs)} program(s)" if programs else "")]
    for m in modules:  # what isn't in the package, and why: said, never left to be noticed
        for f in project.modules[m].report.rejected:
            what = "the program (main)" if f.name == "(program)" else f"{f.name}()"
            report.append(f"NOT in the package: {m}: {what} (line {f.reason_line or f.lineno}: {f.reason})")
    recorded = 0
    if golden:
        tests = out / "tests"
        tests.mkdir(exist_ok=True)
        rep = V.verify_project(paths, n_random=cases, seed=seed, compiler=compiler)
        lines = [f"# qtrans golden cases: Python's own results, recorded by qtrans export {VERSION}",
                 f"# on Python {platform.python_version()}, NumPy {np.__version__}, from "
                 + ", ".join(f"{Path(p).name} (sha256 {_source_hash(p)})" for p in project.paths.values()) + ".",
                 "# CASE function rtol atol / IN driver input / OK|EXC expected result (hex@scale for floats) /",
                 "# PRINTED what it printed. Floats agree when |C++ - Python| <= atol + rtol * scale."]
        left_out = {"illcond": 0, "known": 0, "printed": 0, "mismatch": 0, "other": 0}
        single = len(project.modules) == 1
        for m, r in rep.modules.items():
            sigs = r.translation.signatures
            for fc in r.functions:
                sig = sigs[fc.name]
                out_t = V.output_type(sig)
                for c in fc.cases:
                    if c.outcome != "match":
                        left_out[c.outcome if c.outcome in left_out else "other"] += 1
                        continue
                    if c.py_out != c.cpp_out:
                        left_out["printed"] += 1  # agreed only up to rounding in printed numbers
                        continue
                    enc = " ".join(V._encode(v, p.type) for v, p in zip(c.args, sig.params))
                    if c.py[0] == "ok":
                        expect = " ".join(["OK"] + _tokens(c.py[1], out_t))
                    else:
                        expect = f"EXC {c.cpp[1]}"
                    lines += [f"CASE {fc.key} {fc.rtol!r} {fc.atol!r}", f"IN CALL {fc.key} {enc}", expect]
                    if c.py_out:
                        lines.append("PRINTED S" + c.py_out.encode("utf-8").hex())
                    recorded += 1
        skipped = {k: v for k, v in left_out.items() if v}
        for m, r in rep.modules.items():  # a program: Python's exact output and exit status
            pc = r.program
            if pc is None or m not in programs:
                continue
            if pc.ok and pc.py_out == pc.cpp_out:
                (tests / f"{m}.expected").write_text(pc.py_out, encoding="utf-8", newline="")
                (tests / f"{m}.status").write_text(f"{pc.py_code}\n", encoding="utf-8")
                report.append(f"program {m}: Python's output ({len(pc.py_out.splitlines())} lines) and exit status "
                              f"{pc.py_code} recorded in tests/{m}.expected")
            else:
                report.append(f"program {m}: not recorded (its output matched Python's only up to rounding, or "
                              "not at all: see qtrans verify)")
        lines.insert(4, f"# {recorded} cases" + (", left out: " + ", ".join(f"{v} {k}" for k, v in skipped.items())
                                                 if skipped else ""))
        (tests / "compare_output.cmake").write_text(COMPARE_CMAKE, encoding="utf-8")
        if recorded == 0:  # nothing that translated could be checked: no empty test pretending to pass
            report.append("golden tests: none (no function in the package could be checked against Python)")
            (out / "README.md").write_text(_readme(name, modules, programs, uses_eigen, sources, 0), encoding="utf-8")
            if build:
                report += build_and_test(out, name, uses_eigen, compiler)
            return report
        (tests / "golden.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (tests / "golden.cpp").write_text(GOLDEN_CPP, encoding="utf-8")
        if single:
            (mod, res), = project.modules.items()
            drv = V.driver_source(res, f"{mod}.hpp", res.namespace)
        else:
            drv = V.driver_source(project, targets=list(rep.modules))
        (tests / "qtrans_driver.cpp").write_text(drv, encoding="utf-8")
        # the driver always uses these runtime headers; they go beside the others (one copy of
        # each, so #pragma once sees one file)
        from .translator import LIST_HEADER, PRINT_HEADER, RUNTIME_HEADER
        for h in (RUNTIME_HEADER, PRINT_HEADER, LIST_HEADER):
            if not (inc / h.name).exists():
                shutil.copy(h, inc / h.name)
        report.append(f"golden tests: {recorded} cases of Python's results recorded in tests/golden.txt"
                      + (" (left out: " + ", ".join(f"{v} {k}" for k, v in skipped.items()) + ")" if skipped else ""))
        if not rep.ok:
            report.append("NOTE: qtrans verify found mismatches; only the cases that matched were recorded:")
            report += ["  " + x for x in rep.to_text().splitlines() if "MISMATCH" in x or "RESULT" in x]
    (out / "README.md").write_text(_readme(name, modules, programs, uses_eigen, sources, recorded), encoding="utf-8")
    if build:
        report += build_and_test(out, name, uses_eigen, compiler)
    return report


def build_and_test(out, name, uses_eigen, compiler=None) -> list:
    """Configure, build and run the package's tests with CMake, in out/_build, with the compiler
    qtrans verify uses (MSVC through CMake's Visual Studio generator; GCC / Clang through Ninja,
    or MinGW Makefiles / Unix Makefiles)."""
    import os
    cmake = shutil.which("cmake")
    if cmake is None:
        return ["not built here: CMake isn't installed (the package is complete; build it with CMake 3.16+)"]
    b = out / "_build"
    shutil.rmtree(b, ignore_errors=True)
    args = [cmake, "-S", str(out), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"]
    try:
        cxx = V.find_compiler(compiler)
    except RuntimeError:
        cxx = None
    if cxx is not None and cxx.kind != "msvc":
        if shutil.which("ninja"):
            args += ["-G", "Ninja"]
        elif os.name == "nt":
            args += ["-G", "MinGW Makefiles"]
        args.append(f"-DCMAKE_CXX_COMPILER={cxx.path}")
    if uses_eigen:
        try:
            args.append(f"-D{name.upper()}_EIGEN_DIR={V.find_eigen()}")
        except RuntimeError:
            pass  # find_package(Eigen3) may still find it
    steps = [("configure", args), ("build", [cmake, "--build", str(b), "--config", "Release"]),
             ("test", [shutil.which("ctest") or "ctest", "--test-dir", str(b), "-C", "Release", "--output-on-failure", "-V"])]
    for what, cmd in steps:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            all_lines = (p.stdout + p.stderr).strip().splitlines()
            # what says why (a golden case, an output difference, a compiler error), then the end
            key = [x for x in all_lines if re.search(r"golden|differs from Python|exit status|error|C\+\+:|Python:", x)]
            tail = list(dict.fromkeys(key[:40] + all_lines[-15:]))
            return [f"CMake {what} FAILED:"] + ["  " + x for x in tail]
    text = p.stdout + p.stderr
    summary = re.search(r"golden: \d+ of \d+ cases agree with Python", text)
    # each test's own status line ("1/2 Test #1: pricer_golden ....   Passed"), the same on every
    # platform and CMake version (the closing summary's wording isn't)
    statuses = re.findall(r"Test\s+#\d+:\s+\S+\s+\.*\s*(\*{0,3}\w+)", text)
    programs = len(re.findall(r"same output as Python, character for character", text))
    shutil.rmtree(b, ignore_errors=True)
    what = [summary.group(0)[8:]] if summary else []
    if programs:
        what.append(f"{programs} program(s) print exactly what Python printed")
    passed = sum(1 for x in statuses if x == "Passed")
    if not statuses:
        return ["built with CMake here (no tests: nothing in the package could be checked against Python)"]
    return [f"built and tested with CMake here: {passed} of {len(statuses)} ctest tests passed"
            + (f" ({'; '.join(what)})" if what else "")]


def _readme(name, modules, programs, uses_eigen, sources, recorded) -> str:
    prog = "".join(f"- `{name}_{_cmake_name(p)}`: the program `{p}.py` (its top-level code), from `src/{p}.cpp`\n"
                   for p in programs)
    golden = (f"- `{name}_golden`: checks the C++ against Python's own results on {recorded} inputs, recorded at\n"
              "  export (`tests/golden.txt`; no Python needed). Run it on every machine and compiler you ship on.\n"
              if recorded else "")
    return f"""# {name}: C++ generated by qtrans from {', '.join(sources)}

Generated by [qtrans](https://github.com/SunnyAlexV/qtrans) {VERSION}. The Python is the source:
change it and export again rather than editing these files (for your own C++ for a function,
see `qtrans override`).

## Use it

```cmake
add_subdirectory({name})            # or: find_package({name}) after installing it
target_link_libraries(your_target PRIVATE qtrans::{name})
```

```cpp
#include <{name}/{modules[0]}.hpp>  // namespace {modules[0]}
```

The target is header-only and needs C++17{', and Eigen 3.3+ (found with `find_package(Eigen3)`, or pass `-D' + name.upper() + '_EIGEN_DIR=<folder with Eigen/Dense>`)' if uses_eigen else ''}. It
keeps floating-point arithmetic as written (`-ffp-contract=off`, `/fp:precise` on MSVC): the
numbers are Python's only that way, and the headers refuse to compile under fast-math.

## Build and test

```
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

{prog}{golden}
`reports/` says what translated, what didn't, and why.
"""
