"""Python next to the C++ it became, line by line, with the reason for every non-obvious choice.

    python -m qtrans explain FILE.py[:FUNCTION] [--html OUT.html] [--json]

The translator marks each C++ line with the Python lines it came from (TranslationResult.line_map).
Here those marks are grouped into rows: a row is a statement (or a compound statement's header,
`if ...:` / `for ...:`) on the left and every C++ line it produced on the right, so the two sides
read in parallel. Each row also carries reasons:

- for each construct in its C++ that isn't a plain transcription (a NumPy broadcasting helper,
  a checked int operation, a negative-index wrap, a copy before a store, ...), a sentence on what
  Python semantics it keeps (`RULES`);
- every note the translator made at that statement: a documented difference that applies there
  (`x ** 0.5` with a negative x), or code left out (the dead branch of `if True:`).

Functions that weren't translated show the reason and the hint at the line that caused it.
"""
from __future__ import annotations

import ast
import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .report import FunctionReport
from .translator import PROGRAM, VERSION

# (pattern in the C++ of a row, kind, why). The pattern is what the reason is about, so a reason
# is only ever shown next to C++ that contains it (tests/test_explain.py checks this).
RULES = [
    (r"qt::py::dict<", "dict",
     "a Python dict: qt::py::dict keeps keys in the order they were first added (as Python does: "
     "printing and loops follow it) and finds them by hash"),
    (r"\.at\(", "dict",
     "d[k] reads with .at(k): a missing key raises KeyError, as in Python (d[k] = v adds it)"),
    (r"qt::broadcast_into\(", "broadcasting",
     "updated in place with NumPy's broadcasting rules: the sizes can't be shown equal here, so a "
     "length-1 side stretches and any other mismatch raises ValueError, at run time"),
    (r"qt::broadcast\(", "broadcasting",
     "NumPy broadcasting: the two sizes can't be shown equal from the code, so a length-1 side "
     "stretches to the other's size and any other mismatch raises ValueError, as in NumPy"),
    (r"qt::store\(", "broadcasting",
     "stored with NumPy's rules: a length-1 right side fills the whole part, a size mismatch raises "
     "ValueError"),
    (r"\.(rowwise|colwise)\(\)", "broadcasting",
     "a vector applied to every row (or column) of a matrix: its length is proven to match, so "
     "Eigen's own row-wise operation does it with no check needed"),
    (r"qt::idx\(", "indexing",
     "the index may be negative: Python counts it from the end (and raises IndexError out of "
     "range), so it is wrapped and checked"),
    (r"qt::(row|col)\(", "indexing",
     "a row or column: its index is checked (negative counts from the end) before the view is "
     "formed, as Python raises IndexError"),
    (r"qt::slice\(", "indexing",
     "Python's slice rules: negative ends count from the end, ends past the array clip, any step; "
     "a view of the array's memory, no copy"),
    (r"qt::block\(", "indexing",
     "a 2-D slice with Python's rules for each axis: a view, no copy"),
    (r"qt::select(_rows)?\(", "indexing",
     "a boolean mask: the elements where it is true, in order (a copy, as in NumPy)"),
    (r"qt::masked_update\(", "indexing",
     "assignment through a boolean mask: only the elements where it is true change"),
    (r"qt::(iadd|isub|imul|ineg|iabs)\(", "int64",
     "int arithmetic that could leave 64 bits: the exact result, or OverflowError where Python's "
     "unbounded int would grow past what a long long holds (C++ overflow is undefined)"),
    (r"qt::ipow\(", "int64",
     "int ** int by repeated squaring, checked for overflow like the other int operations"),
    (r"qt::floordiv\(", "division",
     "Python's //: rounds toward minus infinity (C++ truncates toward zero), and raises "
     "ZeroDivisionError for a zero divisor"),
    (r"qt::mod\(", "division",
     "Python's %: the result has the sign of the divisor (C++'s has the sign of the dividend), "
     "and a zero divisor raises ZeroDivisionError"),
    (r"qt::to_int\(", "conversion",
     "int() of a float: ValueError for NaN and OverflowError for inf or beyond 64 bits, where a "
     "bare cast would be undefined"),
    (r"qt::lvalue\(", "sharing",
     "a fresh value passed to a function that changes its argument: Python changes the temporary "
     "and drops it, and C++ is given the same temporary to change"),
    (r"qt::(Vec|Mat)\([A-Za-z_][\w.>-]*\)\)?;", "aliasing",
     "copied first: the right side reads the array being written, and NumPy computes the right "
     "side completely before storing it"),
    (r"\)\.eval\(\)", "aliasing",
     "evaluated into a temporary first: the right side reads the array being written, and NumPy "
     "computes it completely before storing (Eigen would otherwise read elements it has already "
     "overwritten)"),
    (r"\[\[maybe_unused\]\]", "unused",
     "never read in the C++ (it is assigned only, or read only in code the translation leaves "
     "out): marked so the compiler doesn't warn"),
    (r"^\s*const long long \w+_end = ", "loop",
     "range() is evaluated once, before the loop, as in Python: its bound is a call or reads a "
     "variable the loop changes, so it is computed up front"),
    (r"qt::(maximum|minimum)\(", "nan",
     "np.maximum / np.minimum: NaN from either side propagates and a tie returns the second "
     "argument, as NumPy (std::max does neither)"),
    (r"qt::clip\(", "nan", "np.clip: NaN propagates, ties keep x, and lo > hi gives hi, as NumPy"),
    (r"qt::where\(", "evaluation",
     "np.where: both branches are evaluated, then one chosen per element, as NumPy evaluates its "
     "arguments"),
    (r"qt::(solve|inv)\(", "linalg",
     "LU with partial pivoting, the algorithm of LAPACK's gesv; an exactly zero pivot raises "
     "LinAlgError (\"Singular matrix\"), as NumPy"),
    (r"qt::cholesky\(", "linalg",
     "Cholesky from the lower triangle, as LAPACK's potrf; LinAlgError if not positive definite"),
    (r"qt::eigh\(|qt::eigvalsh\(", "linalg",
     "eigenvalues ascending from the lower triangle, as LAPACK's syevd; each eigenvector only up "
     "to its sign, so code that reads single entries of V may differ from NumPy"),
    (r"qt::pinv\(", "linalg",
     "NumPy's rule: singular values above rcond times the largest are kept; computed from an SVD "
     "(or, at full rank, a pivoted QR checked against the four Penrose conditions)"),
    (r"qt::(matmul|matvec|vecmat|dot|outer)\(", "products",
     "@ / np.dot by the operands' ranks (matrix, vector or both), sizes checked as NumPy does; "
     "sums run in Eigen's order, so the last bits can differ from NumPy's BLAS"),
    (r"qt::npm::\w+\(|qt::(exp|log|log10|log2|log1p|expm1|exp2|cbrt|sin|cos|tan|sinh|cosh|tanh|arcsin|arccos|"
     r"arctan|arcsinh|arccosh|arctanh|np_power|arctan2)\(", "library",
     "NumPy's own float64 maths, computed by the code NumPy runs on this machine for this NumPy release "
     "(Intel's SVML on Linux with AVX-512, NumPy's own kernels, or the C library NumPy calls), so the "
     "result is NumPy's to the bit; np.power's shortcuts (x*x, sqrt, 1/x) follow the release"),
    (r"qt::pow\(", "library",
     "** on an array: NumPy's shortcuts for the exponents -1, 0, 0.5, 1 and 2 (the same in every release), "
     "else NumPy's power kernel, as above"),
    (r"qt::norm_(cdf|sf|pdf|ppf|logcdf|logpdf)\(", "library",
     "computed the way SciPy computes it (ndtr: erf near zero, erfc in the tails), so results "
     "agree to the last bit or two"),
    (r"qt::(mean|var|std_|sum|prod|median|quantile|cumsum|cumprod)\(", "reduction",
     "a NumPy reduction ported with NumPy's own rules (ddof, empty arrays, the axis argument)"),
    (r"qt::value\(", "optional",
     "a value that may be None, used as a number: TypeError for None, as in Python"),
    (r"std::optional<", "optional", "`x: float | None`: a value or None"),
    (r"qt::Rng\b", "random",
     "NumPy's Generator reproduced exactly (SeedSequence, PCG64, ziggurat): the same seed gives the "
     "same numbers in the same order"),
    (r"throw linalg_error|qt::linalg_error", "errors", "numpy.linalg.LinAlgError"),
    (r"catch \(const qt::linalg_error&\)", "errors",
     "except LinAlgError: catches what NumPy's routines raise; statements before the error keep "
     "their effect, as in Python"),
    (r"^int main\(\)", "program",
     "the code that runs when the file runs (its top level, or the `if __name__ == \"__main__\":` block) "
     "becomes main(); an exception it doesn't catch ends it with exit status 1, as Python's does"),
    (r"qt::py::repr\(qt::py::np\(", "printing",
     "made by NumPy, so a NumPy scalar in Python: repr shows it as NumPy 2 does, np.float64(2.5)"),
    (r"qt::py::repr\([^)]*\)?, \"[np]+\"\)", "printing",
     "a tuple shows each element's repr; the ones NumPy made show as np.float64(...)"),
    (r"qt::py::repr\(", "printing", "Python's repr(): a str in quotes, an array as array([...])"),
    (r"qt::py::str\(", "printing",
     "as Python prints it: a float as the shortest decimal that reads back the same (0.1, 1e+16), "
     "an array laid out as NumPy lays it out"),
    (r"qt::py::format\(", "printing",
     "Python's format mini-language with CPython's own rules: the same digits, rounding and padding"),
    (r"\? \"True\" : \"False\"", "printing", "a bool prints as True or False"),
    (r"qt::py::round\(", "rounding",
     "Python's round(x, n): the exact decimal value of x rounded half to even, so round(2.675, 2) is "
     "2.67 (2.675 is really 2.67499999...), as in Python"),
    (r"qt::py::round_numpy\(", "rounding",
     "round() of a NumPy float64 is np.round: x times 10^n, rounded half to even, divided by 10^n, "
     "which differs from Python's round now and then (round(np.float64(1.85), 1) is 1.8)"),
    (r"std::to_string\(", "printing", "an int in decimal, as Python writes it"),
    (r"\b\w+_is_int\b", "int or float",
     "this variable holds an int in some assignments and a float in others: the flag beside it says "
     "which Python holds now, so it prints as Python prints it (0, or 0.5)"),
    (r"qt::py::item\(", "lists", "a list index: negative counts from the end, out of range raises IndexError, "
                                "as in Python"),
    (r"qt::py::slice\(", "lists", "a list slice with Python's rules: ends clip, negatives count from the end, "
                                 "any step; a new list"),
    (r"qt::py::sum(_left)?\(", "lists",
     "sum() as CPython adds: from the int 0, floats compensated (Neumaier) from Python 3.12, left to right "
     "before"),
    (r"qt::py::(min|max)\(", "lists", "min() / max() of a list: the first element no later one beats, as "
                                     "Python picks; ValueError when empty"),
    (r"qt::py::(sorted|sort)\(", "lists", "a stable sort, as Python's (equal elements keep their order)"),
    (r"qt::py::contains\(", "lists", "`x in xs`: compared with =="),
    (r"\[&\] \{", "lists", "a list comprehension: a small function run on the spot builds the list, its loop "
                          "variable local to it, as in Python 3"),
    (r"const std::vector<[^>]*>&", "lists", "a list is passed by const reference: the function never changes "
                                           "it (one that would is refused, as the caller's list would change in Python)"),
    (r"std::cerr <<", "printing", "print(..., file=sys.stderr)"),
]
_COMPILED = [(re.compile(p), k, w) for p, k, w in RULES]
# translator notes that say what a rule above already says about the same C++
_NOTE_KIND = {"int64": "int64", "toint": "conversion", "eigh": "linalg", "pinv": "linalg"}

# Python lines that produce no C++ of their own, and why
PY_RULES = [
    (r"^\s*with np\.errstate\(", "errstate",
     "np.errstate only changes which floating-point warnings NumPy prints, never a value: its body "
     "is translated as it stands"),
    (r"^\s*pass\s*(#.*)?$", "pass", "pass does nothing, in C++ too"),
    (r"^\s*@", "decorator", "the decorator is read by qtrans (a dataclass, a staticmethod, ...) and "
                             "shapes the C++ above"),
]
_PY_COMPILED = [(re.compile(p), k, w) for p, k, w in PY_RULES]

# what each parameter passing mode keeps of Python's sharing
_MODES = {
    "cref": "by const reference: the function only reads it",
    "ref": "by reference: the function changes it in place, and in Python the caller sees that "
           "change, so the C++ caller does too",
    "value": "by value: the function rebinds the name to a new array, which in Python leaves the "
             "caller's array alone, so the C++ works on its own copy",
}


@dataclass
class Reason:
    kind: str
    text: str
    token: str = ""        # the C++ the reason is about ("" for a note on the statement)


@dataclass
class Row:
    py_first: int           # Python lines (1-based, inclusive); 0 when the row has none
    py_last: int
    python: list            # the Python lines' text
    cpp_first: int = 0      # header lines (1-based) of the C++, 0 when none
    cpp: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    error: str | None = None  # a refused function: the reason, on its line


@dataclass
class FunctionView:
    name: str
    first: int
    last: int
    translated: bool
    rows: list
    reason: str | None = None   # not translated: why
    line: int | None = None
    hint: str | None = None
    notes: list = field(default_factory=list)
    params: list = field(default_factory=list)  # [{name, python, cpp, mode, why}]
    spans: list = field(default_factory=list)   # the program: its lines, between the definitions
    override: str | None = None  # your own C++ is used here: its status ("up to date", ...)


@dataclass
class ModuleView:
    module: str
    file: str
    version: str
    functions: list
    includes: list           # the header's includes
    general_notes: list
    module_rows: list = field(default_factory=list)  # imports and constants


def _rows_for(src_lines, header_lines, line_map, first, last, notes):
    """The rows of one function: Python lines first..last against the header lines mapped there."""
    spans = {}
    order = []
    for i, (text, span) in enumerate(zip(header_lines, line_map)):
        if span is None or not (first <= span[0] <= last):
            continue
        if span not in spans:
            spans[span] = []
            order.append(span)
        spans[span].append((i + 1, text))
    rows = []
    covered = set()
    for span in sorted(order, key=lambda s: (s[0], s[1])):
        a, b = span
        if any(ln in covered for ln in range(a, b + 1)):  # overlapping spans join the row they overlap
            target = next(r for r in rows if r.py_first <= a <= r.py_last or r.py_first <= b <= r.py_last)
            target.cpp += [t for _, t in spans[span]]
            continue
        cpp = spans[span]
        rows.append(Row(a, b, src_lines[a - 1:b], cpp[0][0], [t for _, t in cpp]))
        covered.update(range(a, b + 1))
    # Python lines no C++ came from: blank lines, a docstring's continuation, dropped code
    for ln in range(first, last + 1):
        if ln not in covered:
            rows.append(Row(ln, ln, [src_lines[ln - 1]]))
    rows.sort(key=lambda r: (r.py_first, r.py_last))
    merged = []
    for r in rows:  # consecutive lines with no C++ form one row
        if merged and not r.cpp and not merged[-1].cpp and merged[-1].py_last + 1 == r.py_first:
            merged[-1].py_last = r.py_last
            merged[-1].python += r.python
        else:
            merged.append(r)
    for r in merged:
        seen = set()
        for line in r.cpp:
            for rx, kind, why in _COMPILED:
                m = rx.search(line)
                if m and why not in seen:
                    seen.add(why)
                    r.reasons.append(Reason(kind, why, m.group(0)))
        if not r.cpp:
            for line in r.python:
                for rx, kind, why in _PY_COMPILED:
                    if rx.search(line) and why not in seen:
                        seen.add(why)
                        r.reasons.append(Reason(kind, why))
        kinds = {x.kind for x in r.reasons}
        for (a, b), key, text in notes:
            if _NOTE_KIND.get(key.split(":")[0]) in kinds:
                continue  # the same point, already made about the C++ that shows it
            if r.py_first <= a <= r.py_last and text not in seen:
                seen.add(text)
                r.reasons.append(Reason(key.split(":")[0], text))
    return merged


def _function_defs(tree):
    """(qualified name, def node, owner class or None) for every function and method, in order;
    a class itself comes first as ("class Name", ClassDef, None): its fields."""
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node, None))
        elif isinstance(node, ast.ClassDef):
            out.append((f"class {node.name}", node, None))
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((f"{node.name}.{m.name}", m, node))
    return out


def explain_result(result, source: str, filename: str = "<input>") -> ModuleView:
    """The rows of every function of a translated module (a TranslationResult)."""
    src_lines = source.splitlines()
    program = getattr(result, "program", None)
    header_lines = (program if program is not None else result.header).splitlines()
    line_map = list((result.program_map if program is not None else result.line_map) or [None] * len(header_lines))
    line_map += [None] * (len(header_lines) - len(line_map))
    tree = ast.parse(source)
    reports = {f.name: f for f in result.report.functions}
    views = []
    for qual, node, owner in _function_defs(tree):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        while first - 1 >= 1 and src_lines[first - 2].strip().startswith("#"):
            first -= 1  # the comment block directly above belongs to it (it becomes a // comment)
        last = node.end_lineno
        is_class = isinstance(node, ast.ClassDef)
        if is_class:  # the class's own lines: up to its first method
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                m0 = min([methods[0].lineno] + [d.lineno for d in methods[0].decorator_list])
                last = m0 - 1
                while last > node.lineno and (not src_lines[last - 1].strip() or src_lines[last - 1].strip().startswith("#")):
                    last -= 1
        key = node.name if owner is None and not is_class else None
        rep = reports.get(qual) if is_class else reports.get(node.name) if owner is None else reports.get(f"class {owner.name}")
        versions = [r for n, r in reports.items() if owner is None and not is_class and n.startswith(node.name + "(")]
        if rep is None and versions:  # one C++ version per kind of call: shown together
            bad = next((r for r in versions if not r.translated), None)
            rep = bad or FunctionReport(node.name, node.lineno, True, notes=[n for r in versions for n in r.notes])
        translated = bool(rep and rep.translated) and any(
            s is not None and node.lineno <= s[0] <= last for s in line_map)
        notes = (result.line_notes or {}).get(node.name if owner is None else f"{owner.name}.{node.name}", []) \
            if translated else []
        if translated and versions:
            notes = [x for n, v in (result.line_notes or {}).items() if n.startswith(node.name + "(") for x in v]
        if translated and versions:
            # one view per version, each with only its own C++ (the k-th definition of the name)
            starts = [i for i, l in enumerate(header_lines)
                      if re.match(rf"inline .*\b{re.escape(node.name)}\(", l) and l.rstrip().endswith("{")]
            for k, vrep in enumerate(versions):
                if k >= len(starts):
                    break
                end = next((j for j in range(starts[k], len(header_lines)) if header_lines[j] == "}"), len(header_lines) - 1)
                vmap = [s if starts[k] <= i <= end else None for i, s in enumerate(line_map)]
                vnotes = (result.line_notes or {}).get(vrep.name, [])
                views.append(FunctionView(vrep.name, first, last, True,
                                          _rows_for(src_lines, header_lines, vmap, first, last, vnotes),
                                          notes=list(vrep.notes)))
            continue
        if translated:
            rows = _rows_for(src_lines, header_lines, line_map, first, last, notes)
            view = FunctionView(qual, first, last, True, rows, notes=list(rep.notes) if rep else [],
                                override=getattr(rep, "override", None))
            sig = result.signatures.get(key) if key else None
            if sig is not None:
                view.params = _params(sig, node)
                if view.params and view.rows:
                    head = next((r for r in view.rows if r.py_first <= node.lineno <= r.py_last), None)
                    for p in view.params:
                        if p["mode"] and head is not None:
                            head.reasons.append(Reason("passing", f"{p['name']}: {p['why']}", p["cpp"]))
        else:
            why = rep.reason if rep else "not translated"
            at = rep.reason_line if rep and rep.reason_line else node.lineno
            rows = []
            for ln in range(first, last + 1):
                rows.append(Row(ln, ln, [src_lines[ln - 1]], error=why if ln == at else None))
            view = FunctionView(qual, first, last, False, rows, why, at, rep.hint if rep else None)
        views.append(view)
    prog = reports.get(PROGRAM)
    if prog is not None and getattr(result, "main_lines", ()):
        views.append(_program_view(prog, result, src_lines, header_lines, line_map))
    includes = [l for l in header_lines if l.startswith("#include")]
    view = ModuleView(result.module, Path(filename).name, VERSION, views, includes,
                      list(result.report.general_notes))
    view.module_rows = _module_rows(tree, src_lines, header_lines, line_map, views, result.report.module_notes)
    return view


def _program_view(rep, result, src_lines, header_lines, line_map):
    """The program (top-level code, int main()): rows for each of its statements."""
    spans = [tuple(s) for s in result.main_lines]
    first, last = min(a for a, _ in spans), max(b for _, b in spans)
    if rep.translated:
        rows = []
        notes = (result.line_notes or {}).get(PROGRAM, [])
        for a, b in spans:
            rows += _rows_for(src_lines, header_lines, line_map, a, b, notes)
        return FunctionView(PROGRAM, first, last, True, rows, notes=list(rep.notes), spans=spans)
    at = rep.reason_line or first
    rows = [Row(ln, ln, [src_lines[ln - 1]], error=rep.reason if ln == at else None)
            for a, b in spans for ln in range(a, b + 1)]
    return FunctionView(PROGRAM, first, last, False, rows, rep.reason, at, rep.hint, spans=spans)


def _module_rows(tree, src_lines, header_lines, line_map, views, notes=()):
    """The module's own lines: its imports against the header's includes, its constants, and
    every top-level line left out, with the reason."""
    rows = []
    for text in notes:  # "line 12: ..." from the translator: nothing is left out silently
        m = re.match(r"line (\d+): (.*)", text)
        if m and 1 <= int(m.group(1)) <= len(src_lines):
            n = int(m.group(1))
            rows.append(Row(n, n, [src_lines[n - 1]], reasons=[Reason("skipped", m.group(2))]))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports:
        a, b = imports[0].lineno, imports[-1].end_lineno
        if any(l.startswith("namespace") for l in header_lines):
            k = header_lines.index(next(l for l in header_lines if l.startswith("namespace")))
            pre = [l for l in header_lines[:k + 1] if l.startswith(("#include", "namespace"))]
        else:  # a program with nothing but main(): no namespace
            pre = [l for l in header_lines if l.startswith("#include")]
        row = Row(a, b, src_lines[a - 1:b], 1 + header_lines.index(pre[0]) if pre else 0, pre)
        if any(l.startswith("int main()") for l in header_lines):
            row.reasons.append(Reason("module", "the program becomes one .cpp file: the runtime it needs is included "
                                                "(Eigen's arrays, NumPy's generator, Python's printing), the "
                                                "functions go in a namespace named after the module, and the code "
                                                "that runs becomes int main()"))
        else:
            row.reasons.append(Reason("module", "the module becomes a header: the runtime it needs is included "
                                                "(Eigen's arrays, NumPy's generator, the scalar helpers) and "
                                                "everything goes in a namespace named after the module"))
        rows.append(row)
    inside = [s for v in views for s in (v.spans or [(v.first, v.last)])]
    for i, (text, span) in enumerate(zip(header_lines, line_map)):
        if span and not any(a <= span[0] <= b for a, b in inside) and not (imports and imports[0].lineno <= span[0] <= imports[-1].end_lineno):
            rows.append(Row(span[0], span[1], src_lines[span[0] - 1:span[1]], i + 1, [text],
                            [Reason("constant", "a module-level constant: an inline const, computed once")]))
    return sorted(rows, key=lambda r: (r.py_first, r.py_last))


def _params(sig, node):
    out = []
    for p in sig.params:
        big = getattr(p.type, "rank", None) or "Struct" in type(p.type).__name__ \
            or p.type.cpp() in ("qt::Rng", "std::string") or getattr(getattr(p.type, "inner", None), "rank", None)
        mode = getattr(p, "mode", None) if big else None
        cpp_type = p.type.cpp()
        if mode == "cref":
            cpp_type = f"const {cpp_type}&"
        elif mode == "ref":
            cpp_type = f"{cpp_type}&"
        why = _MODES.get(mode, "") if mode else ""
        if p.type.cpp() == "qt::Rng":
            why = ("by reference: drawing from a generator advances its state, and in Python the "
                   "caller's generator advances too")
        out.append({"name": p.name, "python": str(p.type), "cpp": f"{cpp_type} {p.cpp}", "mode": mode, "why": why})
    return out


def explain_file(path, function: str | None = None) -> ModuleView:
    from .translator import translate_file
    path = Path(path)
    result = translate_file(path)
    view = explain_result(result, path.read_text(encoding="utf-8-sig"), str(path))
    if function:
        view.functions = [f for f in view.functions if f.name == function or f.name.split(".")[-1] == function]
        if not view.functions:
            raise KeyError(f"no function '{function}' in {path.name}")
    return view


# --- output ------------------------------------------------------------------------------------
def to_json(view: ModuleView) -> str:
    return json.dumps(asdict(view), indent=2)


def to_text(view: ModuleView, width: int = 0) -> str:
    """Two columns: Python | C++, with the reasons under each row."""
    import shutil
    width = width or max(100, shutil.get_terminal_size((140, 40)).columns)
    rows = [r for f in view.functions for r in f.rows] + list(view.module_rows)
    left = min(max((len(l) for r in rows for l in r.python), default=20) + 7, width // 2)
    right = width - left - 3
    n = len(view.functions)
    out = [f"qtrans explain: {view.file} ({n} function{'s' if n != 1 else ''})", ""]
    module = [FunctionView("module", 0, 0, True, view.module_rows)] if view.module_rows else []
    for f in module + view.functions:
        status = (f"your C++, {f.override}" if f.override else "translated") if f.translated \
            else f"NOT TRANSLATED: {f.reason}"
        out.append(f"── {f.name}" + (f" (lines {f.first}-{f.last}): {status}" if f.first else ""))
        if not f.translated and f.hint:
            out.append(f"   hint: {f.hint}")
        def chunks(text, w):  # a long line continues on the next, nothing is cut off
            return [text[i:i + w] for i in range(0, len(text), w)] or [""]

        for r in f.rows:
            pys = [(f"{r.py_first + k:>4} ", part if j == 0 else "  " + part)
                   for k, t in enumerate(r.python) for j, part in enumerate(chunks(t, left - 7))]
            pys = [(num if j == 0 or not num.strip() else "     ", t) for j, (num, t) in enumerate(pys)]
            seen_num = set()
            fixed = []
            for num, t in pys:  # the line number only on a line's first part
                fixed.append((num if num not in seen_num else "     ", t))
                seen_num.add(num)
            ccs = [part if j == 0 else "  " + part for t in r.cpp for j, part in enumerate(chunks(t, right - 2))]
            for k in range(max(len(fixed), len(ccs), 1)):
                num, py = fixed[k] if k < len(fixed) else ("     ", "")
                cc = ccs[k] if k < len(ccs) else ""
                out.append(f"{num}{py:<{left - 5}} │ {cc}")
            if r.error:
                out.append(f"{'':<{left}} │ ✗ {r.error}")
            for why in r.reasons:
                import textwrap
                for k, part in enumerate(textwrap.wrap(why.text, max(right - 4, 30))):
                    out.append(f"{'':<{left}} │ {'  ↳ ' if k == 0 else '    '}{part}")
        out.append("")
    for n in view.general_notes:
        out.append(f"note: {n}")
    return "\n".join(out)


_PY_KW = r"\b(def|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|import|from|as|class|try|except|raise|assert|pass|break|continue|lambda|with)\b"
_CPP_KW = r"\b(inline|const|return|if|else|for|while|double|long|bool|void|auto|struct|namespace|try|catch|throw|static_cast|true|false|nullptr|using|template|typename)\b"


_STR = r"\"\"\"[^\n]*?(?:\"\"\"|$)|'''[^\n]*?(?:'''|$)|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"
_NUM = r"\b\d+\.?\d*(?:e[-+]?\d+)?\b"
_TOKEN = {
    "py": re.compile(r"(?P<c>#.*$)|(?P<s>" + _STR + r")|(?P<n>" + _NUM + r")|(?P<k>" + _PY_KW + r")"),
    "cpp": re.compile(r"(?P<c>//.*$)|(?P<s>\"(?:[^\"\\]|\\.)*\")|(?P<n>" + _NUM + r")|(?P<k>" + _CPP_KW + r")"
                      r"|(?P<ns>\b(?:qt|std|Eigen)::)"),
}


def _highlight(text: str, lang: str) -> str:
    """A small, safe highlighter: comments, strings, numbers, keywords (one pass, escaped)."""
    out, pos = [], 0
    for m in _TOKEN[lang].finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        out.append(f'<span class="{m.lastgroup}">{html.escape(m.group(0))}</span>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def to_html(view: ModuleView) -> str:
    """One self-contained page: for each function, Python on the left and its C++ on the right,
    row by row, with the reasons under the C++ (toggle them with the switch at the top)."""
    funcs = []
    nav = []
    module = FunctionView("module", 0, 0, True, view.module_rows) if view.module_rows else None
    for f in ([module] if module else []) + view.functions:
        anchor = "f-" + re.sub(r"\W", "-", f.name)
        badge = '<span class="ok">translated</span>' if f.translated else '<span class="no">not translated</span>'
        nav.append(f'<a href="#{anchor}">{html.escape(f.name)}</a>')
        rows = []
        in_str = [None]  # the quote of a triple-quoted string still open from an earlier line

        def py_line(t):
            if in_str[0]:
                end = t.find(in_str[0])
                if end < 0:
                    return f'<span class="s">{html.escape(t)}</span>'
                head, t = t[:end + 3], t[end + 3:]
                in_str[0] = None
                return f'<span class="s">{html.escape(head)}</span>' + py_line(t)
            for q in ('"""', "'''"):
                i = t.find(q)
                if i >= 0 and t.count(q) % 2 == 1 and "#" not in t[:i]:
                    in_str[0] = q
                    return _highlight(t[:i], "py") + f'<span class="s">{html.escape(t[i:])}</span>'
            return _highlight(t, "py")

        for r in f.rows:
            py = "".join(f'<div class="ln"><span class="no-sel num">{r.py_first + k}</span>'
                         f'<code>{py_line(t) or "&nbsp;"}</code></div>' for k, t in enumerate(r.python))
            cc = "".join(f'<div class="ln"><span class="no-sel num">{r.cpp_first + k}</span>'
                         f'<code>{_highlight(t, "cpp") or "&nbsp;"}</code></div>' for k, t in enumerate(r.cpp))
            if r.error:
                cc += f'<div class="err">✗ {html.escape(r.error)}</div>'
            why = "".join(f'<li><span class="kind">{html.escape(x.kind)}</span>{html.escape(x.text)}</li>'
                          for x in r.reasons)
            if why:
                cc += f'<ul class="why">{why}</ul>'
            cls = "row" + ("" if r.cpp or r.error else " empty")
            rows.append(f'<div class="{cls}"><div class="py">{py}</div><div class="cpp">{cc}</div></div>')
        extra = ""
        if not f.translated:
            extra = f'<p class="refusal">✗ line {f.line}: {html.escape(f.reason or "")}'
            if f.hint:
                extra += f'<br><span class="hint">hint: {html.escape(f.hint)}</span>'
            extra += "</p>"
        funcs.append(f'<section id="{anchor}"><h2>{html.escape(f.name)} {badge if f is not module else ""}'
                     + (f'<span class="lines">lines {f.first}–{f.last}</span>' if f is not module else "") + f'</h2>{extra}'
                     f'<div class="cols"><div class="colhead">Python</div><div class="colhead">C++</div></div>'
                     f'{"".join(rows)}</section>')
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in view.general_notes)
    ok = sum(1 for f in view.functions if f.translated)
    return _PAGE.format(title=html.escape(f"{view.file} in C++"), file=html.escape(view.file),
                        version=html.escape(view.version), ok=ok, n=len(view.functions),
                        nav=" ".join(nav), body="".join(funcs),
                        notes=f'<ul class="notes">{notes}</ul>' if notes else "")


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1f23; --muted:#6a6f78; --line:#e4e1da; --py:#f3f1ec; --cpp:#eef2f6;
  --k:#7a3e9d; --s:#2f7d32; --n:#b25b00; --c:#8a8f98; --ns:#1f5f99; --hover:#fff4cc; --err:#b3261e; --ok:#2f7d32; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#16181c; --fg:#e6e6e6; --muted:#9aa0a8;
  --line:#2b2f36; --py:#1d2026; --cpp:#1a1f27; --k:#c792ea; --s:#a5d6a7; --n:#ffb74d; --c:#7d848e; --ns:#82b1ff;
  --hover:#3a3420; --err:#ff8a80; --ok:#a5d6a7; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#16181c; --fg:#e6e6e6; --muted:#9aa0a8; --line:#2b2f36; --py:#1d2026; --cpp:#1a1f27;
  --k:#c792ea; --s:#a5d6a7; --n:#ffb74d; --c:#7d848e; --ns:#82b1ff; --hover:#3a3420; --err:#ff8a80; --ok:#a5d6a7; color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:env(safe-area-inset-top, 0px); z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; }}
header h1 {{ font-size:17px; margin:0 0 4px; }}
header .meta {{ color:var(--muted); font-size:13px; }}
header nav {{ margin-top:6px; display:flex; flex-wrap:wrap; gap:6px 12px; font-size:13px; }}
header nav a {{ color:var(--ns); text-decoration:none; }}
label.toggle {{ float:right; font-size:13px; color:var(--muted); cursor:pointer; }}
main {{ padding:8px 16px 40px; }}
section {{ margin:22px 0; }}
h2 {{ font-size:16px; margin:0 0 6px; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }}
h2 .lines {{ color:var(--muted); font-weight:400; font-size:13px; }}
.ok {{ color:var(--ok); font-size:12px; font-weight:600; }} .no {{ color:var(--err); font-size:12px; font-weight:600; }}
.cols, .row {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); }}
.colhead {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); padding:4px 8px; }}
.row {{ border-top:1px solid var(--line); }}
.row:hover .py, .row:hover .cpp {{ background:var(--hover); }}
.py {{ background:var(--py); padding:3px 0; }} .cpp {{ background:var(--cpp); padding:3px 0; border-left:1px solid var(--line); }}
.ln {{ display:flex; font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
code {{ white-space:pre-wrap; overflow-wrap:anywhere; flex:1; min-width:0; }}
.num {{ flex:0 0 3.2em; text-align:right; padding-right:.8em; color:var(--muted); user-select:none; }}
code {{ font:inherit; }}
.row.empty .cpp {{ background:transparent; }}
.k {{ color:var(--k); }} .s {{ color:var(--s); }} .n {{ color:var(--n); }} .c {{ color:var(--c); font-style:italic; }} .ns {{ color:var(--ns); }}
ul.why {{ margin:2px 8px 4px 3.6em; padding:0; list-style:none; font-size:12.5px; color:var(--muted); }}
ul.why li {{ margin:2px 0; }}
.kind {{ display:inline-block; min-width:6.5em; margin-right:6px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
body.no-why ul.why {{ display:none; }}
.err, .refusal {{ color:var(--err); font-size:13px; padding:2px 8px 2px 3.6em; }}
.refusal {{ padding:4px 0; }} .hint {{ color:var(--muted); }}
.notes {{ color:var(--muted); font-size:13px; }}
@media (max-width:760px) {{ .cols {{ display:none; }} .row {{ grid-template-columns:1fr; }} .cpp {{ border-left:0; border-top:1px dashed var(--line); }} }}
</style></head>
<body>
<header>
<label class="toggle"><input type="checkbox" id="why" checked> reasons</label>
<h1>{file}</h1>
<div class="meta">qtrans {version} · {ok} of {n} functions translated · each row: a Python statement and the C++ it became</div>
<nav>{nav}</nav>
</header>
<main>{body}{notes}</main>
<script>
document.getElementById("why").addEventListener("change", e => document.body.classList.toggle("no-why", !e.target.checked));
</script>
</body></html>
"""
