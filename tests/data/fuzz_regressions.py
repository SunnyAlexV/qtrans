"""Bugs the program fuzzer (tools/fuzz) found, each as the minimal function that showed it.

Every function here must translate, compile without a warning, and agree with Python
(tests/test_fuzz.py). The comment above each says what was wrong.
"""
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


QTRANS_CASES = {
    "size_in_min": [(np.ones(4), 3), (np.zeros(0), -2)],
    "slice_in_product": [(np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))],
    "reversed_diag": [(np.array([1.0, 2.0, 3.0]),)],
    "slice_broadcast": [(np.array([5.0, 2.0, 4.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))],
    "two_errors": [(64, np.eye(3))],
    "shape_and_zero_division": [(np.zeros(0),)],
    "temporaries_changed_in_place": [(np.array([1.0, 2.0]),)],
    "empty_mask": [(np.array([1.0, 2.0, 3.0]), np.zeros(0)), (np.array([1.0, 2.0, 3.0]), np.ones(2))],
}
