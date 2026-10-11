"""Bugs the program fuzzer (tools/fuzz) found, each as the minimal function that showed it.

Every function here must translate, compile without a warning, and agree with Python
(tests/test_fuzz.py). The comment above each says what was wrong.
"""
import math

import numpy as np

from qtrans.typing import Mat, Vec


# An array's size is Eigen's Index (long), not long long: std::min(n, v.size()) and
# qt::mod(v.size(), 2) didn't compile (no matching / ambiguous overload).
def size_in_min(v: Vec, n: int) -> int:
    return min(n * n, len(v)) + len(v) // 2 + max(len(v), 3) % (n if n != 0 else 1)


# Arrays only as temporaries, in a function with scalar parameters: the header didn't include
# the array runtime (qt::dot, qt::vec were "not a member of qt").
def arrays_inside_only(x: float) -> float:
    return x * float(np.array([1.0, -2.0, 0.5]) @ (np.array([4.0]) - np.array([1.0, -2.0, 0.5])))


# A slice used in a product (v[:2] @ M), in np.diag (v[::-1]) and broadcast against a matrix:
# Eigen's own slice view claimed direct memory access but had none, which didn't compile.
def slice_in_product(v: Vec, A: Mat) -> Vec:
    return v[:2] @ A


def reversed_diag(v: Vec) -> Mat:
    return np.diag(v[::-1])


def slice_broadcast(v: Vec, A: Mat) -> float:
    return float(np.sum(A / v[1:]))


# A local assigned but never read compiled with -Wunused-variable (now [[maybe_unused]]).
def dead_local(a: float) -> float:
    x = a * 2.0
    return a


# A parameter only assigned, or used only in a branch the translation drops, or rebound to
# itself (p = p): -Wunused-but-set-parameter / -Wunused-parameter.
def param_only_assigned(p: int) -> float:
    p = 0
    return 1.5


def param_in_dead_branch(p: float, q: float) -> float:
    if True:
        return p
    else:
        return q


def param_self_assigned(p: float, q: float) -> float:
    q = q
    return p


# A local read only in a branch the translation drops: -Wunused-variable.
def local_in_dead_branch(a: float) -> float:
    n = 3
    if True:
        return a
    else:
        return float(n)


# Two broken checks in one expression (shapes 3 and 2 don't broadcast; row 64 doesn't exist):
# Python raises the first, left to right; C++ leaves the order of a call's arguments
# unspecified. Classified 'known' (both reject the input), never a silent difference.
def two_errors(i: int, A: Mat) -> float:
    v = np.ones(3) / np.zeros(2) * A[i]
    return float(v[0])


# The same with a division by zero as the second error (the fuzzer's minimal form).
def shape_and_zero_division(v: Vec) -> float:
    n = min(len(np.eye(2) @ v), 0.0 % 0)
    return 1.0


# A fresh value passed to a function that changes its argument in place (Python changes the
# temporary and drops it) was refused; now it goes through qt::lvalue.
def _bump(x: Vec) -> float:
    x += 1.0
    return float(x[0])


def temporaries_changed_in_place(v: Vec) -> float:
    return _bump(v + 1.0) + _bump(v.copy()) + float(v[0])


# NumPy treats a boolean index with no elements as an empty index, whatever the array's
# shape: v[np.array([], dtype=bool)] is [] and assigning through it changes nothing. The C++
# raised IndexError (the size check for masks of any other length is kept).
def empty_mask(v: Vec, w: Vec) -> float:
    picked = v[w > 0.5]
    v[w > 0.5] = 0.0
    return float(np.sum(picked)) + float(np.sum(v))


# `p = p` translates to a comment (Clang warns on a self-assignment), so a parameter or local
# set and then assigned to itself is never read in the C++: GCC 13's
# -Wunused-but-set-parameter / -variable. A read only in `x = x` is not a read.
def param_set_then_self_assigned(p: float) -> float:
    p = 0.0
    p = p
    return 1.0


def local_set_then_self_assigned(v: Vec) -> float:
    x = 1.0
    x = x
    return float(np.sum(v))


# x ** 0.5 with x < 0 makes a Python float complex. Inside an array, the verifier's float64
# copy of Python's result dropped the imaginary part (a ComplexWarning) and compared the real
# part with the C++'s NaN; changed in place, NumPy raises UFuncTypeError (a TypeError). Both are
# the documented complex-power difference now, never a comparison of a complex value as real.
def complex_power_in_an_array(x: float) -> Vec:
    return np.array([x ** 0.5, 1.0])


def complex_power_in_place(v: Vec, x: float) -> float:
    v += x ** 0.5
    return float(np.sum(v))


# Eigen solves this system exactly (the identity); NumPy's LAPACK leaves -3.3e-16 and 1.1e-16
# off the diagonal. Divided by zero: NaN in the C++, -inf and inf in Python. Which one is
# rounding's choice, so it is ill-conditioned (the off-diagonal entries of the exact answer are
# 0, so 0 / 0: the C++ has the exact result).
def solve_then_divide_by_zero(z: float) -> Mat:
    A = np.array([[1.0, 2.0], [3.0, 4.5]])
    return np.linalg.solve(A, A) / (z * 0.0)


# NumPy computes the right side before storing it. `.eval()` gives Eigen's expressions a
# temporary, but of a plain array it returns the array itself, so the C++ of v[::-1] = v read
# elements it had already overwritten ([0, 1, 2, 1, 0] for np.arange(5.)). Copied now.
def reversed_self_assignment(v: Vec, A: Mat) -> float:
    v[::-1] = v
    A[:, 1:] = A[:, :1]
    A[1:] = A[:1]
    return float(np.sum(v * np.arange(len(v)))) + float(np.sum(A * np.arange(A.shape[1])))


# np.exp of this matrix has entries from 0.15 to e**100: singular to working precision (its
# condition number is inf to NumPy). LAPACK's LU found no exactly-zero pivot and returned; Eigen's,
# adding in another order, found one and raised "Singular matrix". Which one happens is
# rounding's choice, so it is ill-conditioned: the verifier checks whether a matrix NumPy was
# given to factor was singular to working precision (smallest singular value <= n * eps * largest).
def inv_of_a_numerically_singular_matrix(P: Mat) -> float:
    m = np.linalg.inv(np.exp(P))
    return 0.0


_NEAR_SINGULAR = np.array([
        [11.768729481618932, -0.30953762381091154, 4.3938481955622635, -0.6826455180635933, 2.9469424051391897, 0.04980026550065728],
        [0.0, 0.09834600109181542, -0.0, -0.03465732587072612, 0.19785614767699705, 1.1016847943344539],
        [1.1190728500489497, 0.011175620496846188, 0.05299160201342173, 0.06811273476905509, 0.42208447497617035, 9.03692594155788],
        [-1.9148788913682666, 97.23283999718058, 0.08068602473885904, 0.17664286733254167, 10.255796308088795, 0.13617576333999384],
        [1.1873942103190636, 1.9112239178790111, 25.251864545993048, 0.0470599724035322, 0.08765495108696071, 0.389842124723611],
        [20.67421605903837, 0.09372069939927091, 1.1163704721570613, 70.6629519898316, 100.0, 1.991243168187423],
])


# Two errors in one expression again: Python raises the broadcast error of the left operand
# first; the C++ (GCC, right to left) first runs the call on the right, where math.log(0.0) is
# -inf (IEEE-754; Python raises ValueError) and int() of it raises OverflowError. Both reject
# the input: `known`, with the order flag.
def _log_zero_as_int(k: int) -> Vec:
    k -= int(math.log(0.0))
    return np.ones(3) * k


def broadcast_error_and_overflow(k: int) -> float:
    x = (np.eye(2) - np.ones(3)).mean(axis=0) @ _log_zero_as_int(1)
    return 1.0


# The same difference where no complex value reaches the result: NumPy compares a complex
# number by its real part, so v > x ** 0.5 with x < 0 is v > 6e-17 in Python and v > NaN
# (all False) in C++. The verifier now reruns Python with every a ** b watched, and a power
# that went complex makes the case `known`.
def complex_power_absorbed_by_a_comparison(x: float, v: Vec) -> Vec:
    return v[v > x ** 0.5]


# A parameter read only in a branch the translation drops (`if False:`) and set afterwards: the
# C++ only writes it (GCC 13: -Wunused-but-set-parameter). Mentions on the left of `p = ...;`
# don't count as reads.
def param_read_only_in_dead_code_then_set(p: float) -> float:
    if False:
        return p
    p = 2.0
    return 1.0


QTRANS_CASES = {
    "size_in_min": [(np.ones(4), 3), (np.zeros(0), -2)],
    "slice_in_product": [(np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))],
    "reversed_diag": [(np.array([1.0, 2.0, 3.0]),)],
    "slice_broadcast": [(np.array([5.0, 2.0, 4.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))],
    "two_errors": [(64, np.eye(3))],
    "shape_and_zero_division": [(np.zeros(0),)],
    "temporaries_changed_in_place": [(np.array([1.0, 2.0]),)],
    "complex_power_in_an_array": [(-2.0,), (4.0,)],
    "complex_power_in_place": [(np.array([1.0, 2.0]), -2.0), (np.array([1.0, 2.0]), 4.0)],
    "solve_then_divide_by_zero": [(1.0,)],
    "reversed_self_assignment": [(np.arange(5.0), np.arange(6.0).reshape(2, 3))],
    "inv_of_a_numerically_singular_matrix": [(_NEAR_SINGULAR,)],
    "broadcast_error_and_overflow": [(16,)],
    "complex_power_absorbed_by_a_comparison": [(-1.0, np.array([1.0, -2.0, 3.0])), (4.0, np.array([1.0, 3.0]))],
    "empty_mask": [(np.array([1.0, 2.0, 3.0]), np.zeros(0)), (np.array([1.0, 2.0, 3.0]), np.ones(2))],
}
