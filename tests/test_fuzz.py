"""The program fuzzer (tools/fuzz): its generator, its shrinker, and the bugs it found."""
import random
from pathlib import Path

import pytest

from qtrans import translate_file, verify_file
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).resolve().parent / "data"


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_fuzzer_findings_stay_fixed(tmp_path):
    r = verify_file(DATA / "fuzz_regressions.py", n_random=40, build_dir=tmp_path)
    assert not r.translation.report.rejected, [(f.name, f.reason) for f in r.translation.report.rejected]
    assert r.warnings == [], r.warnings
    assert r.ok, r.to_text()
    two = [c for f in r.functions if f.name == "two_errors" for c in f.cases if c.source == "given"][0]
    assert two.py[0] == "exc" and two.cpp[0] == "exc" and two.outcome == "known" and two.order


def test_unused_names_are_marked_not_warned():
    h = translate_file(DATA / "fuzz_regressions.py").header
    assert "[[maybe_unused]] double x = a * 2.0;" in h
    assert "double param_only_assigned([[maybe_unused]] long long p)" in h
    assert "param_in_dead_branch(double p, [[maybe_unused]] double q)" in h
    assert '#include "qtrans_arrays.hpp"' in h


def test_views_into_an_in_place_parameter_are_refused():
    """f(v[1:]) or f(A.T) with an f that changes its argument: in Python the change reaches the
    original array through the view. That didn't compile; it is refused with the reason."""
    from qtrans import translate_source
    r = translate_source("""
import numpy as np
from qtrans.typing import Mat, Vec

def bump(x: Vec) -> float:
    x += 1.0
    return float(x[0])

def via_slice(v: Vec) -> float:
    return bump(v[1:])

def via_row(A: Mat) -> float:
    return bump(A[0])
""", "m").report
    reasons = {f.name: f.reason for f in r.rejected}
    assert set(reasons) == {"via_slice", "via_row"}
    assert all("is a view" in why for why in reasons.values())


def test_a_row_from_a_column_of_the_same_matrix_is_refused():
    """A[1] = A[:, j]: NumPy copies the overlapping views element by element and can read what it
    just overwrote (np.arange(25.).reshape(5, 5) gives row 1 = [3, 18, 13, 18, 23], not column 3);
    the fuzzer found the C++ disagreeing. Refused, with the .copy() form as the hint; that one
    and every same-orientation overlap (A[0:2] = A[1:3], v[1:] = v[:-1]) NumPy gets right."""
    from qtrans import translate_source
    r = translate_source("""
import numpy as np
from qtrans.typing import Mat, Vec

def cross(A: Mat, j: int) -> Mat:
    A[1] = A[:, j]
    return A

def cross_t(A: Mat, j: int) -> Mat:
    A[:, 0] = A.T[:, j]
    return A

def fine(A: Mat, v: Vec, j: int) -> Mat:
    A[1] = A[:, j].copy()
    A[0:2] = A[1:3]
    v[1:] = v[:-1]
    return A
""", "m").report
    reasons = {f.name: f.reason for f in r.rejected}
    assert set(reasons) == {"cross", "cross_t"}, reasons
    assert "copies a column of 'A' into a row" in reasons["cross"]
    assert "copies a row of 'A' into a column" in reasons["cross_t"]


def test_generated_programs_are_valid_python():
    """Every program the generator writes is well-typed Python that runs (or raises an
    ordinary error) on the verifier's inputs."""
    import numpy as np
    from tools.fuzz.gen import HEADER, Gen
    g = Gen(random.Random(7))
    ns, srcs = {}, []
    for k in range(60):
        srcs.append(g.function(f"g{k}"))
        exec(compile(HEADER + srcs[-1], f"g{k}", "exec"), ns)  # noqa: S102 - our own generated code
    import ast
    values = {"float": 0.5, "int": 2, "bool": True}
    with np.errstate(all="ignore"):
        for k in range(60):
            fdef = ast.parse(srcs[k]).body[0]
            args = [values.get(a.annotation.id) if a.annotation.id in values
                    else (np.ones(3) if a.annotation.id == "Vec" else np.eye(3)) for a in fdef.args.args]
            try:
                ns[f"g{k}"](*args)
            except (ArithmeticError, ValueError, IndexError, np.linalg.LinAlgError):
                pass  # running into an ordinary error is part of what's being tested


def test_shrinker_finds_a_minimal_failure(tmp_path):
    """The shrinker on a planted failure (a translator crash triggered by one call) cuts a
    many-statement function down to the statement that matters."""
    from tools.fuzz import check
    from tools.fuzz.check import Bug
    from tools.fuzz.shrink import shrink
    src = ("def f(a: float, b: float) -> float:\n"
           "    x = a * b + 1.0\n"
           "    y = x - a\n"
           "    for i in range(3):\n"
           "        y += float(i) * b\n"
           "    z = math.sqrt(abs(y))\n"
           "    return z + PLANT(x)\n")
    real = check.translate_only

    def planted(functions, workdir, stem="fz"):  # PLANT(...) crashes the "translator"
        if any("PLANT(" in s for s in functions.values()):
            return None, "Traceback...\nKeyError: 'planted'"
        return real(functions, workdir, stem)
    check.translate_only = planted
    try:
        b = shrink(Bug("crash", "f", src, "KeyError: 'planted'"), tmp_path, budget_seconds=120, log=lambda *_: None)
    finally:
        check.translate_only = real
    assert b.shrunk is not None and "PLANT(" in b.shrunk
    assert len(b.shrunk.splitlines()) <= 2, b.shrunk  # def + return PLANT(...)
