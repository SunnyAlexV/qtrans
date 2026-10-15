"""qtrans_numpy.hpp: which NumPy release the C++ reproduces.

NumPy computes np.exp, np.log, np.power, ... with different code in different releases (SVML's
high- or low-accuracy routines, its own kernels, the C library; see qtrans_npmath.hpp). The C++
must therefore know which NumPy its Python ran with. This writes that down, from the NumPy
qtrans runs with, as qtrans_python.hpp does for Python's error messages:

    #define QTRANS_NUMPY_VERSION 20504  // NumPy 2.5.4

A C++ build can override it with -DQTRANS_NUMPY_VERSION=... (to match another NumPy).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

SHIPPED = Path(__file__).parent / "runtime" / "qtrans_numpy.hpp"


def version_number(version: str) -> int:
    """'2.5.4' -> 20504, '1.26.0rc1' -> 12600."""
    parts = [int(re.match(r"\d*", p).group() or 0) for p in version.split(".")[:3]]
    parts += [0] * (3 - len(parts))
    return parts[0] * 10000 + min(parts[1], 99) * 100 + min(parts[2], 99)


def numpy_version() -> str:
    import numpy
    return numpy.__version__


def generate(version: str | None = None) -> str:
    version = version or numpy_version()
    return "\n".join([
        f"// qtrans_numpy.hpp: the NumPy release this C++ reproduces, NumPy {version}.",
        "// Written by qtrans.npfacts from the NumPy qtrans ran with; qtrans_npmath.hpp reads it to compute",
        "// np.exp, np.log, np.power, ... with the code that NumPy release runs. Do not edit; to match",
        "// another NumPy, build with -DQTRANS_NUMPY_VERSION=<major*10000 + minor*100 + patch>.",
        "#pragma once",
        "",
        "#ifndef QTRANS_NUMPY_VERSION",
        f"#define QTRANS_NUMPY_VERSION {version_number(version)}  // NumPy {version}",
        "#endif",
        ""])


def _version_of(text: str):
    m = re.search(r"#define QTRANS_NUMPY_VERSION \d+  // NumPy (\S+)", text)
    return m.group(1) if m else None


def header_path() -> Path:
    """The qtrans_numpy.hpp for the running NumPy: the shipped one if it is for this release,
    else one written once into the user's cache."""
    version = numpy_version()
    if SHIPPED.exists() and _version_of(SHIPPED.read_text(encoding="utf-8")) == version:
        return SHIPPED
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "qtrans"
    for root in (base, Path.cwd() / ".qtrans_cache"):
        path = root / f"numpy-{version}" / "qtrans_numpy.hpp"
        try:
            if path.exists() and _version_of(path.read_text(encoding="utf-8")) == version:
                return path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(generate(version), encoding="utf-8")
            os.replace(tmp, path)
            return path
        except OSError:
            continue
    raise RuntimeError("qtrans: could not write qtrans_numpy.hpp to a cache folder")
