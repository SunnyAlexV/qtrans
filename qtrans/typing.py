"""Annotations for array code (stage 2).

    from qtrans.typing import Vec, Mat

    def portfolio_vol(w: Vec, cov: Mat) -> float: ...

At run time both are just float64 NumPy arrays, so the Python code runs unchanged. qtrans
reads the names to learn the rank, which C++ needs and `np.ndarray` doesn't say:
Vec is 1-D (qt::Vec in C++), Mat is 2-D (qt::Mat). IVec is a 1-D int64 array (qt::IVec):
np.arange of ints, rng.integers(..., size=n), the indexes a bootstrap draws.
"""
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Mat = npt.NDArray[np.float64]
IVec = npt.NDArray[np.int64]

__all__ = ["Vec", "Mat", "IVec"]
