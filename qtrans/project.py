"""Multi-module projects (0.8): several Python files that import from each other.

    from qtrans import translate_project
    p = translate_project(["bl_core.py", "bl_inference.py"], "cpp_out")

Each module becomes its own header and C++ namespace, laid out as the Python is: bl_core.py ->
bl_core.hpp, `namespace bl_core`, and a module that does `from bl_core import nearest_psd`
includes bl_core.hpp and calls bl_core::nearest_psd. The modules a file imports from are found
next to it (or through `from . import m` / `from .m import f`) and translated too.

The analysis is project-wide: modules are translated in import order, each after the ones it
imports, so a function knows exactly how every imported function treats its arguments (which
arrays it modifies in place, which generator it draws from, which argument it may hand back),
and calling an imported function that could not be translated is refused with its reason.

Import cycles are refused: Python tolerates two modules importing each other, C++ headers
can't include each other. Class names must be unique across the project.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .translator import (NPMATH_HEADERS, ARRAYS_HEADER, LIST_HEADER, UNICODE_HEADER, PYTHON_HEADER, PYRANDOM_HEADER, PRINT_HEADER, RANDOM_HEADER, RUNTIME_HEADER, ModuleTranslator, TranslationResult)


class ProjectError(Exception):
    """The files can't be translated as one project (an import cycle, two classes of one name)."""


def _imports_of(tree, folder: Path) -> list:
    """Modules of the project a module imports at top level: those with a .py file next to it."""
    out = []

    def add(name):
        if name and "." not in name and (folder / f"{name}.py").exists() and name not in out:
            out.append(name)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                add(a.name)
        elif isinstance(node, ast.ImportFrom) and node.level <= 1:
            if all(a.name.startswith("QTRANS_") for a in node.names):
                continue  # verifier inputs (QTRANS_CASES) kept in another file: not part of the C++
            if node.module is None:
                for a in node.names:
                    add(a.name)
            else:
                add(node.module)
    return out


@dataclass
class ProjectResult:
    modules: dict                   # module name -> TranslationResult, in import order
    paths: dict                     # module name -> Path of its .py file
    requested: list                 # the modules asked for (the others were pulled in by imports)
    imports: dict = field(default_factory=dict)  # module -> project modules it imports
    translators: dict = field(default_factory=dict, repr=False)

    @property
    def order(self) -> list:
        return list(self.modules)

    def classes(self) -> dict:
        """Every translated class of the project: name -> (ClassInfo, module)."""
        return {n: (info, m) for m, r in self.modules.items() for n, info in (r.classes or {}).items()}

    @property
    def uses_arrays(self) -> bool:
        return any(r.uses_arrays for r in self.modules.values())

    @property
    def uses_random(self) -> bool:
        return any(r.uses_random for r in self.modules.values())

    def write(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, r in self.modules.items():
            if r.library or r.program is None:
                (out / f"{name}.hpp").write_text(r.header, encoding="utf-8")
            if r.program is not None:
                (out / f"{name}.cpp").write_text(r.program, encoding="utf-8")
            (out / f"{name}_report.md").write_text(r.report.to_markdown(), encoding="utf-8")
        headers = [RUNTIME_HEADER, PYTHON_HEADER, PYRANDOM_HEADER]
        texts = [r.header + (r.program or "") for r in self.modules.values()]
        if any("qtrans_print.hpp" in t or "qtrans_list.hpp" in t for t in texts):
            headers += [PRINT_HEADER, LIST_HEADER, UNICODE_HEADER]
        if self.uses_arrays or any("qtrans_arrays.hpp" in t or "qtrans_random.hpp" in t for t in texts):
            headers.append(ARRAYS_HEADER)
        if self.uses_arrays or any(m in t for t in texts for m in ("qtrans_arrays.hpp", "qtrans_random.hpp", "qtrans_npmath.hpp")):
            headers += list(NPMATH_HEADERS)
        if self.uses_random or any("qtrans_random.hpp" in t for t in texts):
            headers.append(RANDOM_HEADER)
        if any(r.uses_runtime or r.uses_arrays or "qtrans_print.hpp" in r.header + (r.program or "")
               or "qtrans_list.hpp" in r.header + (r.program or "")
               for r in self.modules.values()):
            for h in headers:
                (out / h.name).write_text(h.read_text(encoding="utf-8-sig"), encoding="utf-8")
        return out


def project_cases(folder) -> tuple[dict, dict]:
    """QTRANS_CASES / QTRANS_TOLERANCE from a qtrans_cases.py next to the modules:
    {module: {function: ...}} (verifier inputs kept out of the code itself). -> (cases, tolerances)"""
    path = Path(folder) / "qtrans_cases.py"
    if not path.exists():
        return {}, {}
    from .verify import load_module
    mod = load_module(path)
    return getattr(mod, "QTRANS_CASES", {}) or {}, getattr(mod, "QTRANS_TOLERANCE", {}) or {}


def discover(paths) -> tuple[dict, dict]:
    """-> ({module: path}, {module: [project modules it imports]}), following imports to
    the modules next to each file."""
    found, edges, todo = {}, {}, [Path(p).resolve() for p in paths]
    while todo:
        path = todo.pop()
        name = path.stem
        if name in found:
            if found[name] != path:
                raise ProjectError(f"two modules named {name}: {found[name]} and {path}")
            continue
        found[name] = path
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
        edges[name] = _imports_of(tree, path.parent)
        todo += [path.parent / f"{d}.py" for d in edges[name]]
    return found, edges


def _order(edges) -> list:
    """Modules so that each comes after the ones it imports; an import cycle is an error."""
    out, state = [], {}

    def visit(m, chain):
        if state.get(m) == "done":
            return
        if state.get(m) == "active":
            cyc = chain[chain.index(m):] + [m]
            raise ProjectError("import cycle: " + " -> ".join(cyc) + ". C++ headers can't include each "
                               "other; move what both need into a third module")
        state[m] = "active"
        for d in edges[m]:
            visit(d, chain + [m])
        state[m] = "done"
        out.append(m)
    for m in sorted(edges):
        visit(m, [])
    return out


def translate_project(paths, out_dir=None) -> ProjectResult:
    """Translate the given files and every project module they import (from the same folder)."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        paths = sorted(q for q in p.glob("*.py") if not q.name.startswith(("test_", "_"))
                       and q.name not in ("conftest.py", "qtrans_cases.py", "setup.py")) \
            if p.is_dir() else [p]
    found, edges = discover(paths)
    done, results = {}, {}
    for name in _order(edges):
        path = found[name]
        t = ModuleTranslator(path.read_text(encoding="utf-8-sig"), name, str(path), deps=done)
        results[name] = t.translate()
        done[name] = t
    owners = {}
    for name, t in done.items():
        for cname, info in t.classes.items():
            if info.external:
                continue
            if cname in owners:
                raise ProjectError(f"class {cname} is defined in both {owners[cname]} and {name}: class names must "
                                   "be unique across the project (they become one Python type in the bindings)")
            owners[cname] = name
    requested = [Path(p).resolve().stem for p in paths]
    proj = ProjectResult(results, found, requested, edges, done)
    if out_dir is not None:
        proj.write(out_dir)
    return proj
