"""The numpy.linalg audit (qtrans_arrays.hpp, -DQTRANS_LINALG_AUDIT) and the per-operation
checker (tools/linalg_check): that the bounds hold for correct results, that they catch wrong
ones, and that the verifier reports an operation beyond its bound as a failure."""
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qtrans import verify_file
from qtrans import verify as V
from tests.test_semantics import HAVE_CXX

compiled = pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
RUNTIME = Path(__file__).resolve().parents[1] / "qtrans" / "runtime"

# Each audit routine on a correct result (ratio <= 1) and on one changed by a relative 1e-9,
# far less than any test of the values would notice on an ill-conditioned matrix but far more
# than a backward-stable algorithm leaves (ratio >> 1).
PROGRAM = textwrap.dedent(r"""
    #include <cstdio>
    #include "qtrans_arrays.hpp"
    using namespace qt;
    static double ratio() { return audit::state().worst; }
    int main() {
        Eigen::MatrixXd a(3, 3);
        a << 4, 1, 0.5,  1, 3, 0.25,  0.5, 0.25, 2;
        const Eigen::MatrixXd b = Eigen::MatrixXd::Constant(3, 1, 1.0);
        Eigen::PartialPivLU<Eigen::MatrixXd> f(a);
        Eigen::MatrixXd x = f.solve(b);
        audit::reset(); audit::lu_solve("solve", a, f, b, x); std::printf("solve %g", ratio());
        x(0, 0) *= 1 + 1e-9;
        audit::reset(); audit::lu_solve("solve", a, f, b, x); std::printf(" %g\n", ratio());

        audit::reset(); audit::lu(a, f); std::printf("lu %g", ratio());
        Eigen::PartialPivLU<Eigen::MatrixXd> g(a);
        const_cast<Eigen::MatrixXd&>(g.matrixLU())(2, 2) *= 1 + 1e-9;
        audit::reset(); audit::lu(a, g); std::printf(" %g\n", ratio());

        Eigen::MatrixXd l = Eigen::LLT<Eigen::MatrixXd>(a).matrixL();
        audit::reset(); audit::cholesky(a, l); std::printf("cholesky %g", ratio());
        l(1, 0) *= 1 + 1e-9;
        audit::reset(); audit::cholesky(a, l); std::printf(" %g\n", ratio());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(a);
        Eigen::MatrixXd v = es.eigenvectors();
        audit::reset(); audit::eigh(a, es.eigenvalues(), v); std::printf("eigh %g", ratio());
        v(0, 1) *= 1 + 1e-9;
        audit::reset(); audit::eigh(a, es.eigenvalues(), v); std::printf(" %g\n", ratio());

        Eigen::MatrixXd x2 = pinv(Eigen::MatrixXd(a).array()).matrix();
        audit::reset(); audit::pinv(a, x2); std::printf("pinv %g", ratio());
        x2(2, 1) *= 1 + 1e-9;
        audit::reset(); audit::pinv(a, x2); std::printf(" %g\n", ratio());
    }
""")


@compiled
def test_audit_passes_correct_results_and_catches_wrong_ones(tmp_path):
    cxx, eigen = V.find_compiler(), V.find_eigen()
    src = tmp_path / "audit_test.cpp"
    src.write_text(PROGRAM, encoding="utf-8")
    exe = tmp_path / ("audit_test.exe" if cxx.kind == "msvc" else "audit_test")
    r = subprocess.run(cxx.command(src, exe, RUNTIME, [eigen], defines=("QTRANS_LINALG_AUDIT",)),
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout.split("\n")
    seen = {}
    for line in filter(None, out):
        name, good, bad = line.split()
        seen[name] = (float(good), float(bad))
    assert set(seen) == {"solve", "lu", "cholesky", "eigh", "pinv"}
    for name, (good, bad) in seen.items():
        assert good <= 1.0, (name, good)
        assert bad > 10.0, (name, bad)  # a 1e-9 change is far beyond rounding: caught


def test_an_operation_beyond_its_bound_is_a_mismatch():
    c = V.Case("f", (), "random")
    c.py, c.cpp = ("ok", 1.0), ("ok", 1.0)  # the values even agree
    c.audit = (3, 2.5, "solve")
    V.classify(c, V.FLOAT, 1e-10, 1e-13)
    assert c.outcome == "mismatch" and c.unstable and not c.certified
    c2 = V.Case("f", (), "random")
    c2.py, c2.cpp, c2.audit = ("ok", 1.0), ("ok", 1.0), (3, 0.2, "solve")
    V.classify(c2, V.FLOAT, 1e-10, 1e-13)
    assert c2.outcome == "match" and c2.certified


@compiled
def test_verifier_audits_every_linalg_call(tmp_path):
    f = tmp_path / "la.py"
    f.write_text(textwrap.dedent("""
        import numpy as np
        from qtrans.typing import Mat, Vec

        def chain(A: Mat, b: Vec) -> float:
            S = A @ A.T + np.eye(A.shape[0])
            L = np.linalg.cholesky(S)
            w, V = np.linalg.eigh(S)
            x = np.linalg.solve(S, b)
            return float(np.sum(L) + np.sum(w) + x @ b + np.linalg.det(S) + np.sum(np.linalg.pinv(A)))
    """), encoding="utf-8")
    rep = verify_file(f, n_random=60, build_dir=tmp_path / "build")
    assert rep.ok, rep.to_text()
    audited = [c for c in rep.functions[0].cases if c.audit and c.audit[0] > 0]
    assert len(audited) > 30
    assert max(c.audit[0] for c in audited) >= 6  # cholesky, eigh, solve (LU + solve), det, pinv
    assert all(c.certified for c in audited)
    assert "linalg audit:" in rep.to_text()


@compiled
def test_every_operation_meets_its_bounds_on_the_battery(tmp_path):
    """A small run of tools/linalg_check (the full one is in CI and in the README)."""
    from tools.linalg_check import check, summarise
    cases, _ = check(seed=1, per_band=1, workdir=tmp_path)
    rows, failures = summarise(cases)
    assert failures == [], failures
    assert all(r["audited"] > 0 for r in rows)


def test_the_audit_is_compiled_out_by_default():
    text = (RUNTIME / "qtrans_arrays.hpp").read_text(encoding="utf-8")
    assert "#define QTRANS_AUDIT(call) ((void)0)" in text
    from qtrans import translate_source
    h = translate_source("import numpy as np\nfrom qtrans.typing import Mat, Vec\n\n"
                         "def f(A: Mat, b: Vec) -> Vec:\n    return np.linalg.solve(A, b)\n", "m").header
    assert "AUDIT" not in h and "audit" not in h
