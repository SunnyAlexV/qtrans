"""Functions whose bugs only show on non-square or empty arrays: a transposed dimension is
invisible on n x n inputs. The verifier must find their rectangular shapes by itself."""
import numpy as np

from qtrans.typing import Mat, Vec


def project(A: Mat, x: Vec) -> Vec:
    return A @ x


def gram(A: Mat) -> Mat:
    return A.T @ A


def demean_rows(A: Mat, mu: Vec) -> Mat:
    return A - mu


def blank_like(A: Mat) -> Mat:
    out = np.zeros_like(A)
    out[0, :] = A[0, :] if A.shape[0] > 0 else out[0, :]
    return out


def col_means(A: Mat) -> Vec:
    return np.mean(A, axis=0)


def outer_fill(x: Vec, y: Vec) -> Mat:
    return np.outer(x, y)


def last_col(A: Mat) -> Vec:
    return A[:, -1]
