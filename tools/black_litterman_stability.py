"""Why examples/portfolio.py writes Black-Litterman in its Woodbury form.

Compares the textbook ("information") form and the Woodbury form against the textbook formula
evaluated in 50-digit arithmetic, as a view becomes more confident (smaller omega).
Requires mpmath:  pip install mpmath
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from portfolio import _COV, _P, _Q, _W  # noqa: E402

mp.mp.dps = 50


def textbook(cov, w, P, Q, om, delta=2.5, tau=0.05):
    pi = delta * cov @ w
    tci = np.linalg.inv(tau * cov)
    oi = np.diag(1.0 / om)
    return np.linalg.solve(tci + P.T @ oi @ P, tci @ pi + P.T @ oi @ Q)


def woodbury(cov, w, P, Q, om, delta=2.5, tau=0.05):
    pi = delta * cov @ w
    tc = tau * cov
    return pi + tc @ P.T @ np.linalg.solve(P @ tc @ P.T + np.diag(om), Q - P @ pi)


def reference(cov, w, P, Q, om, delta=2.5, tau=0.05):
    C, W, Pm, Qm = (mp.matrix(a.tolist()) for a in (cov, w, P, Q))
    pi = delta * C * W
    tci = (tau * C) ** -1
    oi = mp.diag([1 / mp.mpf(x) for x in om])
    return np.array([float(v) for v in mp.lu_solve(tci + Pm.T * oi * Pm, tci * pi + Pm.T * oi * Qm)])


print(f"  {'omega of view 1':<18}{'textbook rel err':>18}{'woodbury rel err':>18}")
for om1 in (4e-4, 1e-8, 1e-11, 1e-14):
    om = np.array([om1, 9e-4])
    ref = reference(_COV, _W, _P, _Q, om)
    err = [np.max(np.abs(f(_COV, _W, _P, _Q, om) - ref)) / np.max(np.abs(ref)) for f in (textbook, woodbury)]
    print(f"  {om1:<18.0e}{err[0]:>18.1e}{err[1]:>18.1e}")
