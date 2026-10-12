"""Checking qtrans's numpy.linalg routines one operation at a time, by what each one guarantees.

    python -m tools.linalg_check [--seed S] [--per-band K] [--out FILE.md]

Comparing an answer with NumPy's only works for well-conditioned problems: for a matrix with
condition number 1e12, two correct implementations of `solve` may differ in the fourth digit.
So each routine is judged by its defining property instead, on a few thousand constructed
matrices (sizes 0 to 50, condition numbers 1 to 1e18 built in as U diag(s) V^T, badly scaled
rows and columns, Hilbert matrices, Wilkinson's growth matrix, rank-deficient, exactly singular,
symmetric with repeated, negative and zero eigenvalues, rectangular), three ways:

1. **Backward stability** (the runtime's own audit, -DQTRANS_LINALG_AUDIT): the residual each
   routine leaves, divided by the bound numerical analysis gives for it (qtrans_arrays.hpp).
   Ratio <= 1: the result is the exact answer for a matrix within rounding of the input.
2. **Errors**: the C++ raises LinAlgError exactly where NumPy does, or the matrix is singular
   (for cholesky: semi-definite) to working precision, where an exactly-zero pivot is a matter
   of rounding.
3. **Agreement with NumPy (or the exact answer) as far as the conditioning allows**: solve and
   inv differ from NumPy by at most (||R_c|| + ||R_np||) / sigma_min(A), R the residuals
   (X - X_exact = A^-1 R exactly, so this bound is rigorous up to the rounding of R itself);
   det is compared with a 50-digit mpmath determinant, within exp(||A^-1|| ||E||) - 1 for its
   LU backward error E; eigenvalues differ by at most what the two residuals allow (first
   order); cholesky and pinv differences are reported against the first-order bound
   kappa * (eta_c + eta_np), not judged.

Exit code 1 if any ratio of 1. or 3. (judged ones) exceeds 1, or an error in 2. is unexplained.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

U = np.finfo(np.float64).eps / 2
HERE = Path(__file__).resolve().parent
OPS = ("solve", "inv", "det", "cholesky", "eigh", "pinv")


# --- the matrices ---------------------------------------------------------------------------
def _orth(rng, n):
    if n == 0:
        return np.zeros((0, 0))
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.where(np.diag(r) == 0, 1.0, np.diag(r)))


def _with_cond(rng, m, n, kappa, mode="geometric"):
    k = min(m, n)
    if k == 0:
        return np.zeros((m, n))
    if mode == "geometric":
        s = np.geomspace(1.0, 1.0 / kappa, k) if k > 1 else np.ones(1)
    else:  # one small singular value, the rest 1 (a "nearly singular" matrix)
        s = np.ones(k)
        s[-1] = 1.0 / kappa
    s = s * 10.0 ** rng.uniform(-3, 3)  # overall scale: the bounds are scale-free
    return _orth(rng, m)[:, :k] @ np.diag(s) @ _orth(rng, n)[:, :k].T


def _sym_with(rng, eigs):
    q = _orth(rng, len(eigs))
    return (q * eigs) @ q.T


def _wilkinson(n):  # partial pivoting's worst case: growth 2^(n-1)
    a = np.eye(n) - np.tril(np.ones((n, n)), -1)
    a[:, -1] = 1.0
    return a


def _hilbert(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


@dataclass
class Case:
    op: str
    a: np.ndarray
    b: np.ndarray | None
    kind: str
    cpp: tuple = ()          # ("OK", audited, worst, values) or ("EXC", type)
    worst_op: str = ""
    py: tuple = ()           # ("OK", value) or ("EXC", type)
    notes: dict = field(default_factory=dict)


def battery(seed=0, per_band=4) -> list:
    rng = np.random.default_rng(seed)
    cases = []
    sizes = (1, 2, 3, 5, 10, 25, 50)
    kappas = (1.0, 1e2, 1e4, 1e8, 1e12, 1e15, 1e18)

    def add(op, a, kind, b=None):
        cases.append(Case(op, np.ascontiguousarray(a, dtype=np.float64), b, kind))

    def rhs(n):
        k = int(rng.integers(1, 4))
        return rng.standard_normal((n, k)) * 10.0 ** rng.uniform(-3, 3)

    for n in sizes:
        for kappa in kappas:
            for mode in ("geometric", "one small"):
                for _ in range(per_band):
                    a = _with_cond(rng, n, n, kappa, mode)
                    for op in ("solve", "inv", "det"):
                        add(op, a, f"cond {kappa:.0e} ({mode})", rhs(n) if op == "solve" else None)
                    tall = _with_cond(rng, n + int(rng.integers(0, 4)), n, kappa, mode)
                    add("pinv", tall if rng.random() < 0.5 else tall.T, f"cond {kappa:.0e} ({mode}), rectangular")
        for _ in range(per_band):
            a = rng.standard_normal((n, n))
            d1, d2 = 10.0 ** rng.uniform(-8, 8, n), 10.0 ** rng.uniform(-8, 8, n)
            graded = d1[:, None] * a * d2[None, :]
            for op in ("solve", "inv", "det", "pinv"):
                add(op, a, "Gaussian", rhs(n) if op == "solve" else None)
                add(op, graded, "rows and columns scaled 1e-8..1e8", rhs(n) if op == "solve" else None)
        w = _wilkinson(n)
        for op in ("solve", "inv", "det", "pinv"):
            add(op, w, "Wilkinson growth matrix", rhs(n) if op == "solve" else None)
        if n <= 25:
            for op in ("solve", "inv", "det", "pinv"):
                add(op, _hilbert(n), "Hilbert", rhs(n) if op == "solve" else None)
        if n >= 2:
            a = rng.standard_normal((n, n))
            a[-1] = a[0]  # exactly singular
            r = rng.standard_normal((n, n - 1)) @ rng.standard_normal((n - 1, n))  # rank n-1, rounded
            for op in ("solve", "inv", "det", "pinv"):
                add(op, a, "two equal rows", rhs(n) if op == "solve" else None)
                add(op, r, "rank n-1 (rounded)", rhs(n) if op == "solve" else None)
            add("pinv", rng.standard_normal((n + 3, 1)) @ rng.standard_normal((1, n)), "rank 1, rectangular")
        # symmetric: cholesky and eigh
        for kappa in kappas:
            for _ in range(per_band):
                lam = np.geomspace(1.0, 1.0 / kappa, n) * 10.0 ** rng.uniform(-3, 3)
                spd = _sym_with(rng, lam)
                add("cholesky", spd, f"SPD cond {kappa:.0e}")
                add("eigh", spd, f"SPD cond {kappa:.0e}")
        for _ in range(per_band):
            lam = rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 3)
            ind = _sym_with(rng, lam)
            add("cholesky", ind, "symmetric indefinite")
            add("eigh", ind, "symmetric indefinite")
            rep = _sym_with(rng, np.repeat(rng.standard_normal(max(1, n // 3)), 3)[:n])
            add("eigh", rep, "repeated eigenvalues")
            add("cholesky", rep @ rep.T, "Gram matrix, repeated eigenvalues")
            semi = _sym_with(rng, np.concatenate([np.abs(rng.standard_normal(n - 1)), [0.0]]))
            add("cholesky", semi, "positive semi-definite (one zero eigenvalue)")
            add("eigh", semi, "positive semi-definite (one zero eigenvalue)")
            g = rng.standard_normal((n, n))
            add("eigh", g, "non-symmetric (lower triangle used)")
            add("cholesky", g @ g.T + n * np.eye(n), "SPD, diagonally dominant")
    # degenerate shapes and values
    for op in OPS:
        add(op, np.zeros((0, 0)), "empty", np.zeros((0, 1)) if op == "solve" else None)
        add(op, np.zeros((1, 1)), "1x1 zero", np.ones((1, 1)) if op == "solve" else None)
        add(op, np.zeros((3, 3)), "zero matrix", np.ones((3, 1)) if op == "solve" else None)
        add(op, np.array([[1e-300, 0.0], [0.0, 1e-300]]), "tiny (1e-300), well conditioned",
            np.ones((2, 1)) if op == "solve" else None)
        add(op, np.array([[1e300, 1.0], [1.0, 1e-300]]), "huge and tiny entries",
            np.ones((2, 1)) if op == "solve" else None)
    for op in ("solve", "inv", "det", "cholesky", "eigh"):
        add(op, np.ones((2, 3)), "not square", np.ones((2, 1)) if op == "solve" else None)
    return cases


# --- running both sides ------------------------------------------------------------------------
def _fmt(m):
    return f"{m.shape[0]} {m.shape[1]} " + " ".join(repr(float(x)) for x in m.ravel())


def build(workdir: Path):
    from qtrans.verify import find_compiler, find_eigen
    cxx, eigen = find_compiler(), find_eigen()
    runtime = Path(__file__).resolve().parents[2] / "qtrans" / "runtime"
    exe = workdir / ("linalg_check.exe" if cxx.kind == "msvc" else "linalg_check")
    cmd = cxx.command(HERE / "driver.cpp", exe, runtime, [eigen], defines=("QTRANS_LINALG_AUDIT",))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    if r.returncode != 0:
        raise RuntimeError(f"compiling the driver failed:\n{r.stdout}\n{r.stderr}")
    return exe, cxx


def run_cpp(cases, exe):
    lines = []
    for c in cases:
        lines.append(f"{c.op} {_fmt(c.a)}" + (f" {_fmt(c.b)}" if c.op == "solve" else ""))
    r = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=3600)
    out = r.stdout.splitlines()
    if len(out) != len(cases):
        raise RuntimeError(f"driver gave {len(out)} lines for {len(cases)} cases: {r.stderr[-400:]}")
    for c, line in zip(cases, out):
        head, _, worst_op = line.partition(" | ")
        t = head.split()
        if t[0] == "EXC":
            c.cpp = ("EXC", t[1])
        else:
            c.cpp = ("OK", int(t[1]), float(t[2]), np.array([float(x) for x in t[3:]]))
        c.worst_op = worst_op.strip()


def _numpy(c):
    la = np.linalg
    try:
        with np.errstate(all="ignore"):
            if c.op == "solve":
                return ("OK", la.solve(c.a, c.b))
            if c.op == "inv":
                return ("OK", la.inv(c.a))
            if c.op == "det":
                return ("OK", np.array([la.det(c.a)]))
            if c.op == "cholesky":
                return ("OK", la.cholesky(c.a))
            if c.op == "eigh":
                w, v = la.eigh(c.a)
                return ("OK", (w, v))
            if c.op == "pinv":
                return ("OK", la.pinv(c.a))
    except la.LinAlgError:
        return ("EXC", "LinAlgError")
    except ValueError:
        return ("EXC", "ValueError")
    raise AssertionError(c.op)


def _cpp_value(c):
    v = c.cpp[3]
    n, m = c.a.shape
    if c.op == "solve":
        return v.reshape(c.b.shape)
    if c.op in ("inv", "cholesky"):
        return v.reshape(n, n)
    if c.op == "det":
        return v
    if c.op == "eigh":
        return v[:n], v[n:].reshape(n, n)
    if c.op == "pinv":
        return v.reshape(m, n)
    raise AssertionError(c.op)


# --- the checks ----------------------------------------------------------------------------------
def _sv(a):
    if a.size == 0 or not np.isfinite(a).all():
        return None
    return np.linalg.svd(a, compute_uv=False)


def singular_to_working_precision(a) -> bool:
    s = _sv(a)
    return s is not None and (s[0] == 0 or s[-1] <= max(a.shape) * 2 * U * s[0])


def semidefinite_to_working_precision(a) -> bool:
    if a.size == 0 or not np.isfinite(a).all():
        return False
    s = np.tril(a) + np.tril(a, -1).T
    w = np.linalg.eigvalsh(s)
    return w[0] <= a.shape[0] * 2 * U * max(np.max(np.abs(w)), np.finfo(float).tiny)


def _resid_slack(a, x, b):  # the rounding of computing B - AX in double (gamma_{n+1}(|B| + |A||X|))
    n = a.shape[0]
    g = (n + 1) * U / (1 - (n + 1) * U)
    return g * np.linalg.norm(np.abs(b) + np.abs(a) @ np.abs(x))


def _det_exact(a):
    import mpmath
    mpmath.mp.dps = 50
    return float(mpmath.det(mpmath.matrix(a.tolist())))


def forward(c):
    """(ratio, judged): the C++ answer's distance from NumPy's (or the exact one) over what
    the conditioning allows; None when the problem is singular to working precision, or its
    scale leaves the bound itself out of range (entries near 1e300)."""
    with np.errstate(all="ignore"):
        r = _forward(c)
    if r is None or not math.isfinite(r[0]):
        return None
    return r


def _forward(c):
    a = c.a
    if c.cpp[0] != "OK" or c.py[0] != "OK" or a.size == 0 or not np.isfinite(a).all() \
            or np.max(np.abs(a)) > 1e150:  # squares of the entries overflow in the bounds
        return None
    xc = _cpp_value(c)
    if c.op in ("solve", "inv"):
        b = c.b if c.op == "solve" else np.eye(a.shape[0])
        xn = c.py[1]
        if not (np.isfinite(xc).all() and np.isfinite(xn).all()):
            return None
        s = _sv(a)
        smin = s[-1] - a.shape[0] * 2 * U * s[0]  # sigma_min, less NumPy's own SVD error
        if smin <= 0:
            return None
        bound = (np.linalg.norm(b - a @ xc) + _resid_slack(a, xc, b)
                 + np.linalg.norm(b - a @ xn) + _resid_slack(a, xn, b)) / smin
        diff = np.linalg.norm(xc - xn)
        return (diff / bound if bound > 0 else (0.0 if diff == 0 else math.inf)), True
    if c.op == "det":
        n = a.shape[0]
        if n > 30 or singular_to_working_precision(a):
            return None
        exact = _det_exact(a)
        if exact == 0 or not math.isfinite(exact):
            return None
        import scipy.linalg as sl
        # E: the C++ LU's backward error is bounded by the audit (ratio <= 1 of
        # 2 gamma_n || |L||U| || + u ||A||); take that bound, with the C++ L and U
        # approximated by a LAPACK LU of the same matrix (the same pivoting rule)
        p, l, u = sl.lu(a)
        g = n * U / (1 - n * U)
        e = 2 * g * np.linalg.norm(np.abs(l) @ np.abs(u)) + U * np.linalg.norm(a)
        ainv = np.linalg.norm(np.linalg.inv(a))
        x = ainv * e
        if x >= 0.5:
            return None
        bound = math.expm1(x) + 2 * n * U  # det(A + E) / det(A) - 1, then n roundings of the product
        rel = abs(xc[0] - exact) / abs(exact)
        c.notes["numpy_rel"] = abs(c.py[1][0] - exact) / abs(exact)
        return (rel / bound if bound > 0 else math.inf), True
    if c.op == "eigh":
        wc, vc = xc
        wn, vn = c.py[1]
        s = np.tril(a) + np.tril(a, -1).T
        if not (np.isfinite(wc).all() and np.isfinite(wn).all()):
            return None
        an = np.linalg.norm(s)
        n = a.shape[0]
        e_c = np.linalg.norm(s @ vc - vc * wc) + an * np.linalg.norm(vc.T @ vc - np.eye(n))
        e_n = np.linalg.norm(s @ vn - vn * wn) + an * np.linalg.norm(vn.T @ vn - np.eye(n))
        slack = 2 * (n + 1) * U * an + n * n * np.finfo(float).tiny  # rounding (and underflow) of the residuals
        bound = e_c + e_n + slack
        diff = np.max(np.abs(wc - wn))
        return (diff / bound if bound > 0 else (0.0 if diff == 0 else math.inf)), True
    if c.op in ("cholesky", "pinv"):
        xn = c.py[1]
        if not (np.isfinite(xc).all() and np.isfinite(xn).all()) or np.linalg.norm(xn) == 0:
            return None
        s = _sv(a)
        if s is None or s[-1] == 0:
            return None
        if c.op == "pinv" and np.any(np.abs(s - 1e-15 * s[0]) <= 10 * max(a.shape) * U * s[0]):
            return None  # a singular value within rounding of NumPy's cutoff: the rank is rounding's call
        kappa = s[0] / s[-1]
        if c.op == "cholesky":
            sym = np.tril(a) + np.tril(a, -1).T
            eta = lambda l: np.linalg.norm(sym - l @ l.T) / np.linalg.norm(sym)  # noqa: E731
        else:
            eta = lambda x: np.linalg.norm(a @ x @ a - a) / (np.linalg.norm(a) ** 2 * np.linalg.norm(x))  # noqa: E731
        bound = kappa * (eta(xc) + eta(xn) + 2 * max(a.shape) * U)
        diff = np.linalg.norm(xc - xn) / np.linalg.norm(xn)
        return (diff / bound if bound > 0 else math.inf), False
    return None


def error_parity(c):
    """'agree', 'knife edge' (explained) or 'DISAGREE'."""
    ce, pe = c.cpp[0] == "EXC", c.py[0] == "EXC"
    if ce == pe:
        if ce and {c.cpp[1], c.py[1]} - {"LinAlgError", "ValueError", "ShapeError"}:
            return "DISAGREE"
        return "agree"
    raised = c.cpp if ce else c.py
    if raised[1] == "LinAlgError":
        if c.op in ("solve", "inv") and singular_to_working_precision(c.a):
            return "knife edge"
        if c.op == "cholesky" and semidefinite_to_working_precision(c.a):
            return "knife edge"
    return "DISAGREE"


def band(c):
    s = _sv(c.a)
    if s is None:
        return "empty / non-finite"
    if s[0] == 0 or singular_to_working_precision(c.a):
        return "singular to working precision"
    k = s[0] / s[-1]
    if k < 1e4:
        return "cond < 1e4"
    if k < 1e10:
        return "cond 1e4 .. 1e10"
    return "cond >= 1e10"


def check(seed=0, per_band=4, workdir=None):
    cases = battery(seed, per_band)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="qtrans_linalg_"))
    exe, cxx = build(workdir)
    run_cpp(cases, exe)
    for c in cases:
        c.py = _numpy(c)
        c.notes["parity"] = error_parity(c)
        c.notes["band"] = band(c)
        f = forward(c)
        c.notes["forward"] = f
    return cases, cxx


def summarise(cases):
    """Per operation: matrices, audited, worst audit ratio by band, error parity, worst forward."""
    rows = []
    failures = []
    for op in OPS:
        mine = [c for c in cases if c.op == op]
        audited = [c for c in mine if c.cpp[0] == "OK" and c.cpp[1] > 0]
        worst = max(audited, key=lambda c: c.cpp[2], default=None)
        by_band = {}
        for c in audited:
            b = c.notes["band"]
            by_band[b] = max(by_band.get(b, 0.0), c.cpp[2])
        parity = {k: sum(1 for c in mine if c.notes["parity"] == k) for k in ("agree", "knife edge", "DISAGREE")}
        fw = [(c.notes["forward"][0], c) for c in mine if c.notes["forward"] is not None]
        judged = any(c.notes["forward"][1] for _, c in fw)
        fworst = max(fw, key=lambda t: t[0], default=(None, None))
        rows.append(dict(op=op, n=len(mine), audited=len(audited), worst=worst.cpp[2] if worst else 0.0,
                         worst_case=worst, by_band=by_band, parity=parity, forward=fworst[0],
                         forward_case=fworst[1], forward_n=len(fw), judged=judged))
        for c in audited:
            if c.cpp[2] > 1.0:
                failures.append(f"{op} ({c.kind}, {c.a.shape}): audit ratio {c.cpp[2]:.3g} in {c.worst_op}")
        for c in mine:
            if c.notes["parity"] == "DISAGREE":
                failures.append(f"{op} ({c.kind}, {c.a.shape}): C++ {c.cpp[:2] if c.cpp[0] == 'OK' else c.cpp}, NumPy {c.py[0] if c.py[0] == 'OK' else c.py}")
        if judged:
            for r_, c in fw:
                if r_ > 1.0:
                    failures.append(f"{op} ({c.kind}, {c.a.shape}): forward ratio {r_:.3g}")
    return rows, failures


BANDS = ("cond < 1e4", "cond 1e4 .. 1e10", "cond >= 1e10", "singular to working precision")


def report(cases, cxx, seed, per_band) -> str:
    from qtrans.verify import environment
    rows, failures = summarise(cases)
    out = ["# numpy.linalg in qtrans, checked one operation at a time", "",
           f"`python -m tools.linalg_check --seed {seed} --per-band {per_band}`: {len(cases):,} matrices.", "",
           f"Platform: {environment(cxx)}", "",
           "Backward stability: the worst audit ratio (residual / the bound numerical analysis gives; "
           "<= 1 means backward stable), by the matrix's condition number.", "",
           "| operation | matrices | audited | " + " | ".join(BANDS) + " | worst overall |",
           "|---|---:|---:|" + "---:|" * (len(BANDS) + 1)]
    for r in rows:
        cells = [f"{r['by_band'][b]:.3f}" if b in r["by_band"] else "-" for b in BANDS]
        out.append(f"| `{r['op']}` | {r['n']} | {r['audited']} | " + " | ".join(cells) + f" | {r['worst']:.3f} |")
    out += ["", "Errors (`LinAlgError` and shape errors) against NumPy's, and agreement with NumPy (or the exact "
            "answer, for `det`) as a fraction of what the conditioning allows:", "",
            "| operation | same error / both return | one side raises, matrix singular to working precision "
            "| unexplained | compared | worst difference / allowed |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        p = r["parity"]
        f = "-" if r["forward"] is None else f"{r['forward']:.3f}" + ("" if r["judged"] else " (reported)")
        out.append(f"| `{r['op']}` | {p['agree']} | {p['knife edge']} | {p['DISAGREE']} | {r['forward_n']} | {f} |")
    out += ["", f"Failures: {len(failures)}"] + [f"- {x}" for x in failures[:50]]
    return "\n".join(out) + "\n", failures
