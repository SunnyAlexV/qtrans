"""`qtrans inspect`: what qtrans will make of a file, without compiling anything.

    python -m qtrans inspect FILE.py [--json]

For each function: translated or not (and if not, the reason, its line and the hint), the C++
signature and how each array or object parameter is passed and why, the documented Python/C++
differences that apply to it and where, the numpy.linalg routines it uses, and the functions of
the module it calls.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .explain import _MODES, _params

_LINALG = re.compile(r"qt::(solve|inv|det|cholesky|eigh|eigvalsh|pinv)\(")


@dataclass
class FunctionInfo:
    name: str
    line: int
    translated: bool
    signature: str = ""
    params: list = field(default_factory=list)
    differences: list = field(default_factory=list)  # [{line, note}]
    linalg: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    reason: str | None = None
    reason_line: int | None = None
    hint: str | None = None


def inspect_file(path) -> list:
    from .translator import translate_file
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    result = translate_file(path)
    tree = ast.parse(source)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    header = result.header.splitlines()
    lmap = result.line_map or []
    out = []
    for rep in result.report.functions:
        node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and (n.name == rep.name or f"class {n.name}" == rep.name)), None)
        info = FunctionInfo(rep.name, rep.lineno,
                            rep.translated)
        if node is not None and isinstance(node, ast.FunctionDef):
            info.calls = sorted({c.func.id for c in ast.walk(node) if isinstance(c, ast.Call)
                                 and isinstance(c.func, ast.Name) and c.func.id in names and c.func.id != node.name})
        if not rep.translated:
            info.reason, info.reason_line, info.hint = rep.reason, rep.reason_line, rep.hint
            out.append(info)
            continue
        sig = result.signatures.get(rep.name)
        if sig is not None and node is not None:
            first, last = node.lineno, node.end_lineno
            lines = [h for h, s in zip(header, lmap) if s is not None and first <= s[0] <= last]
            info.signature = next((l.strip()[len("inline "):].rstrip(" {") for l in lines if l.startswith("inline ")), "")
            info.params = _params(sig, node)
            info.linalg = sorted({m.group(1) for l in lines for m in _LINALG.finditer(l)})
            for (a, b), key, text in (result.line_notes or {}).get(rep.name, []):
                info.differences.append({"line": a, "note": text})
        out.append(info)
    return out


def to_text(path, infos) -> str:
    ok = sum(1 for i in infos if i.translated)
    out = [f"qtrans inspect: {Path(path).name}: {ok} of {len(infos)} translate", ""]
    for i in infos:
        if not i.translated:
            out.append(f"✗ {i.name} (line {i.line}): not translated")
            out.append(f"    line {i.reason_line}: {i.reason}")
            if i.hint:
                out.append(f"    hint: {i.hint}")
            out.append("")
            continue
        out.append(f"✓ {i.name} (line {i.line})")
        if i.signature:
            out.append(f"    C++: {i.signature}")
        for p in i.params:
            if p["mode"]:
                out.append(f"    {p['name']}: {p['why']}")
        if i.linalg:
            out.append(f"    numpy.linalg: {', '.join(i.linalg)} (audited for backward stability by qtrans verify)")
        if i.calls:
            out.append(f"    calls: {', '.join(i.calls)}")
        for d in i.differences:
            out.append(f"    line {d['line']}: {d['note']}")
        out.append("")
    return "\n".join(out)


def to_json(path, infos) -> str:
    return json.dumps({"file": str(path), "functions": [asdict(i) for i in infos]}, indent=2)


__all__ = ["inspect_file", "to_text", "to_json", "_MODES"]
