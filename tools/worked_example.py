"""Writes docs/WORKED_EXAMPLE.md: one module, docs/worked_example/pricer.py, taken through every
step, with the real output of each command pasted in. Run it to reproduce the page:

    python tools/worked_example.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qtrans.verify import verify_file  # noqa: E402

SRC = ROOT / "docs" / "worked_example" / "pricer.py"


def qtrans(*args, cwd):
    p = subprocess.run([sys.executable, "-m", "qtrans", *args], cwd=cwd, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    return (p.stdout + p.stderr).replace(str(cwd) + "/", "").replace(str(cwd), ".").rstrip()


def main():
    work = Path(tempfile.mkdtemp(prefix="qtrans_worked_"))
    (work / "pricer.py").write_text(SRC.read_text(encoding="utf-8"), encoding="utf-8")
    t = qtrans("translate", "pricer.py", "-o", "cpp_out", cwd=work)
    header = (work / "cpp_out" / "pricer.hpp").read_text(encoding="utf-8")
    body = header[header.index("namespace pricer {"):]
    v = qtrans("verify", "pricer.py", "--cases", "200", cwd=work)
    v = "\n".join(l for l in v.splitlines() if not l.strip().startswith("platform:"))
    rep = verify_file(work / "pricer.py", n_random=50, bench=7, build_dir=work / "bench")
    rows = [f"| `{fc.name}` | {fc.py_ns / 1e3:,.2f} | {fc.cpp_ns / 1e3:,.3f} | {fc.py_ns / fc.cpp_ns:,.0f}× |"
            for fc in rep.functions if fc.py_ns and fc.cpp_ns]
    e = qtrans("export", "pricer.py", "-o", "pricer_cpp", "--cases", "50", cwd=work)

    L = ["# A worked example, from Python to verified C++", "",
         "One module taken through every step, with the real output of each command (written by "
         "`tools/worked_example.py`; run it to reproduce this page). The module is "
         "[`docs/worked_example/pricer.py`](worked_example/pricer.py): a Black-Scholes call, its vega, "
         "implied volatility by Newton's method, and a book of calls.", "",
         "## 1. The Python", "", "```python", SRC.read_text(encoding="utf-8").rstrip(), "```", "",
         "`QTRANS_CASES` are the inputs you want checked by name: here an expired option (`T = 0`, so "
         "`sd = 0`: Python raises `ZeroDivisionError`), a negative spot (the function's own `ValueError`) and "
         "a price no volatility reaches (Newton gives up). `qtrans verify` adds 200 random inputs per function.", "",
         "## 2. Translate", "", "```", "$ qtrans translate pricer.py -o cpp_out", t, "```", "",
         "The header, from its namespace on (above it: the includes and the fast-math guard):", "",
         "```cpp", body.rstrip(), "```", "",
         "What to notice:", "",
         "- `qt::py::truediv`, `qt::py::math_log`, `qt::py::math_sqrt`: Python's float errors, raised as Python "
         "raises them (`ZeroDivisionError: float division by zero` for the expired option), with the running "
         "Python's words.",
         "- In `implied_vol`, `vol -= diff / vega(...)` keeps a plain `/`: `ndtr` returns a NumPy number, so "
         "`bs_call` does, and so `diff` is one, and NumPy's numbers don't raise (`np.float64(1.0) / 0.0` is inf, "
         "with a warning, in Python too). qtrans follows where each number comes from to decide.",
         "- In `vega`, two parts of one expression can raise, so the left ones are worked out first "
         "(`const double left = ...`): Python's order, so Python's error.",
         "- The docstrings became comments, the `raise ValueError(...)` a `throw qt::ValueError(...)`, the "
         "`for _ in range(100)` an ordinary loop; `book` takes its arrays as `const qt::Vec&`, since it only "
         "reads them.",
         "- `ndtr` is `qt::norm_cdf`: SciPy's algorithm on the C library's `erf`, equal to SciPy's to rounding "
         "(the max rel err column of the verify table below says how close; `vega`, which doesn't use it, is "
         "identical to the bit).", "",
         "## 3. Verify", "", "```", "$ qtrans verify pricer.py --cases 200", v, "```", "",
         "Every case is a match: the values within `rtol=1e-10` (most to the last bit), and every error the "
         "same class with the same message.", "",
         "## 4. Time", "",
         "The same functions on the inputs above, 7 rounds, Python and C++ taking turns, medians (µs per call; "
         "`tools/benchmark.py` has the protocol and many more):", "",
         "| Function | Python (µs) | C++ (µs) | Speed-up |", "|---|---:|---:|---:|", *rows, "",
         "## 5. Hand it over", "", "```", "$ qtrans export pricer.py -o pricer_cpp", e, "```", "",
         "`pricer_cpp/` is a CMake package: the headers, and golden tests holding Python's answers, so the C++ "
         "team can build and test it without Python. `qtrans bind pricer.py` would instead compile it into a "
         "Python module, for Python code to call the fast version.", ""]
    out = ROOT / "docs" / "WORKED_EXAMPLE.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
