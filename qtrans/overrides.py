"""Your own C++ for a function: used in place of qtrans's translation, checked against the
Python by `qtrans verify`, and kept across re-translations.

The C++ lives next to the Python, in <module>_overrides.hpp:

    // qtrans: override price  python=3f9a1c2b7e5d
    inline double price(double S, double K) {
        ... your C++ ...
    }
    // qtrans: end price

Lines before the first block (#include <immintrin.h>, helper functions) are copied into the
header, before the module's namespace. `qtrans override pricing.py price` writes the first
version of a block (qtrans's own translation, to edit).

Each block records a fingerprint of the Python it was written for: the function's syntax tree
(not its text: comments and spacing don't count; docstrings neither), together with everything
it uses from the project -- the functions it calls (and theirs), the module constants and
classes it reads. When the Python changes, the block is out of date: it is still used, verify
still checks it, and every report says so until `qtrans override --accept` (which runs verify
first, and refuses if your C++ no longer agrees with the Python).

The function's head (return type, name, parameter types) must be the one qtrans gives the
Python; parameter names and attributes ([[gnu::hot]]) are yours. Defaults are written once, in
the declaration qtrans puts at the top of the namespace.
"""
from __future__ import annotations

import ast
import copy
import datetime as _dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

OPEN = re.compile(r"^\s*//\s*qtrans:\s*override\s+([A-Za-z_]\w*)\b(.*)$")
CLOSE = re.compile(r"^\s*//\s*qtrans:\s*end\s+([A-Za-z_]\w*)\s*$")
PY_PRINT = re.compile(r"\bpython=([0-9a-f]{12})\b")
ACCEPTED = re.compile(r"\baccepted=(\S+)")


class OverrideError(Exception):
    """The overrides file can't be used as it stands (an unclosed block, a name used twice...)."""


@dataclass
class Override:
    name: str
    head: str            # the function's head, up to its opening brace
    body: list           # the lines between the braces
    comments: list       # comment lines written above the head, inside the block
    fingerprint: str | None
    accepted: str | None
    line: int            # the block's first line in the file
    status: str = ""     # "up to date" / "out of date" (set when matched against the Python)
    current: str = ""    # the Python's fingerprint now


@dataclass
class OverridesFile:
    path: Path
    preamble: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)  # name -> Override, in file order
    text: str = ""


def path_for(source_path) -> Path:
    p = Path(source_path)
    return p.with_name(f"{p.stem}_overrides.hpp")


def load(source_path) -> OverridesFile | None:
    """The overrides next to a .py file, or None when there is no such file."""
    if source_path is None or str(source_path).startswith("<"):
        return None
    p = path_for(source_path)
    if not p.is_file():
        return None
    return parse(p.read_text(encoding="utf-8-sig"), p)


def parse(text: str, path=Path("<overrides>")) -> OverridesFile:
    of = OverridesFile(Path(path), text=text)
    lines = text.splitlines()
    k = 0
    seen_block = False
    while k < len(lines):
        m = OPEN.match(lines[k])
        if m is None:
            if CLOSE.match(lines[k]):
                raise OverrideError(f"{path.name}:{k + 1}: `// qtrans: end {CLOSE.match(lines[k]).group(1)}` "
                                    "without its `// qtrans: override` line")
            if not seen_block:
                of.preamble.append(lines[k])
            elif lines[k].strip() and not lines[k].strip().startswith("//"):
                raise OverrideError(f"{path.name}:{k + 1}: C++ between blocks; put helpers before the first "
                                    "`// qtrans: override` line (they go into the header first)")
            k += 1
            continue
        seen_block = True
        name, rest = m.group(1), m.group(2)
        start = k
        k += 1
        inner = []
        while k < len(lines) and not CLOSE.match(lines[k]):
            if OPEN.match(lines[k]):
                raise OverrideError(f"{path.name}:{start + 1}: the block for {name}() has no "
                                    f"`// qtrans: end {name}` line before the next block")
            inner.append(lines[k])
            k += 1
        if k == len(lines):
            raise OverrideError(f"{path.name}:{start + 1}: the block for {name}() has no `// qtrans: end {name}` line")
        if CLOSE.match(lines[k]).group(1) != name:
            raise OverrideError(f"{path.name}:{k + 1}: `// qtrans: end {CLOSE.match(lines[k]).group(1)}` closes "
                                f"the block for {name}()")
        if name in of.blocks:
            raise OverrideError(f"{path.name}:{start + 1}: a second block for {name}() "
                                f"(the first is on line {of.blocks[name].line})")
        k += 1
        comments = []
        while inner and (inner[0].strip().startswith("//") or not inner[0].strip()):
            comments.append(inner.pop(0))
        code = "\n".join(inner)
        brace = _first_brace(code)
        end = code.rstrip().rfind("}")
        if brace < 0 or end < brace:
            raise OverrideError(f"{path.name}:{start + 1}: the block for {name}() must hold one function: "
                                "its head, then { body }")
        head = code[:brace].strip()
        body = code[brace + 1:end].strip("\n").split("\n")
        fp = PY_PRINT.search(rest)
        acc = ACCEPTED.search(rest)
        of.blocks[name] = Override(name, head, body, comments, fp.group(1) if fp else None,
                                   acc.group(1) if acc else None, start + 1)
    # leading lines qtrans wrote at the top of the file (its own header comment) aren't code
    while of.preamble and (of.preamble[0].strip().startswith("//") or not of.preamble[0].strip()):
        of.preamble.pop(0)
    while of.preamble and not of.preamble[-1].strip():
        of.preamble.pop()
    return of


def _first_brace(code: str) -> int:
    """The `{` that opens the body: the first one outside comments, strings and the parameter list."""
    depth = 0
    k = 0
    while k < len(code):
        c = code[k]
        if code.startswith("//", k):
            k = code.find("\n", k)
            if k < 0:
                return -1
            continue
        if code.startswith("/*", k):
            k = code.find("*/", k) + 2
            continue
        if c in "\"'":
            q = c
            k += 1
            while k < len(code) and code[k] != q:
                k += 2 if code[k] == "\\" else 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "{" and depth == 0:
            return k
        k += 1
    return -1


# --- does the head match the Python's? -------------------------------------------------------------
def _strip_head(head: str) -> str:
    h = re.sub(r"//[^\n]*|/\*.*?\*/", " ", head, flags=re.S)
    h = re.sub(r"\[\[.*?\]\]", " ", h)
    return re.sub(r"\s+", " ", h).strip()


def _params(text: str) -> list:
    out, depth, cur = [], 0, ""
    for c in text:
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return [p.strip() for p in out]


def _param_type(p: str) -> str:
    p = re.split(r"=(?!=)", p, maxsplit=1)[0].strip()  # no default
    m = re.match(r"^(.*?[\s&*>])\s*([A-Za-z_]\w*)$", p)
    t = m.group(1) if m else p
    return re.sub(r"\s*([&*<>,:])\s*", r"\1", re.sub(r"\s+", " ", t)).strip()


def head_parts(head: str):
    """(return type and name, [parameter types]) of a C++ function head, or None."""
    h = _strip_head(head)
    a, b = h.find("("), h.rfind(")")
    if a < 0 or b < a:
        return None
    before = re.sub(r"\b(inline|static|constexpr|QT_HOT|QT_COLD|QT_FORCE_INLINE|QT_NOINLINE)\b", " ", h[:a])
    return re.sub(r"\s+", " ", before).strip(), [_param_type(p) for p in _params(h[a + 1:b])]


def head_mismatch(user_head: str, expected_head: str) -> str | None:
    """Why the user's head isn't the one the Python's signature gives (None: it is)."""
    u, e = head_parts(user_head), head_parts(expected_head)
    if u is None:
        return "its head can't be read (expected something like: " + expected_head + ")"
    if u[0] != e[0]:
        return f"it is declared `{u[0]}(...)`; the Python's signature gives `{e[0]}(...)`"
    if len(u[1]) != len(e[1]):
        return f"it takes {len(u[1])} parameters; the Python function takes {len(e[1])}"
    for k, (x, y) in enumerate(zip(u[1], e[1])):
        if x != y:
            return f"parameter {k + 1} is `{x}`; the Python's type gives `{y}`"
    return None


def without_defaults(head: str) -> str:
    """The head with any `= default` removed (defaults are written once, in the declaration)."""
    a, b = head.find("("), head.rfind(")")
    if a < 0 or b < a:
        return head
    ps = [re.split(r"=(?!=)", p, maxsplit=1)[0].rstrip() for p in _params(head[a + 1:b])]
    return head[:a + 1] + ", ".join(ps) + head[b:]


# --- the fingerprint of the Python ------------------------------------------------------------------
def _canonical(node) -> str:
    n = copy.deepcopy(node)
    for x in ast.walk(n):
        body = getattr(x, "body", None)
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body and \
                isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            x.body = body[1:] or [ast.Pass()]  # a docstring doesn't change what the function does
    return ast.dump(n, include_attributes=False)


def _top_level(tree):
    defs = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = n
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            defs[n.target.id] = n
    return defs


def fingerprint(source: str, name: str, imported=None, dep_sources=None) -> str:
    """12 hex digits: the function `name` of this module source, and everything of the project it
    reaches (its callees, the constants and classes it reads), by meaning rather than spelling.
    imported: local name -> (module, name there); dep_sources: module -> source."""
    imported = imported or {}
    dep_sources = dep_sources or {}
    trees = {}

    def defs_of(module):
        if module not in trees:
            src = source if module is None else dep_sources.get(module)
            trees[module] = _top_level(ast.parse(src)) if src is not None else {}
        return trees[module]

    seen, parts = set(), []
    todo = [(None, name)]
    while todo:
        module, n = todo.pop()
        if (module, n) in seen:
            continue
        seen.add((module, n))
        defs = defs_of(module)
        if n not in defs:
            if module is None and n in imported:
                todo.append(imported[n])
            continue
        node = defs[n]
        parts.append(f"{module or ''}.{n}={_canonical(node)}")
        for x in ast.walk(node):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                todo.append((module, x.id))
            elif isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name):
                todo.append((module, x.value.id))  # R.TABLE: what R is
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()[:12]


# --- writing the file ---------------------------------------------------------------------------------
HEADER = """// {name}: your own C++ for functions of {py}, used by qtrans in place of its
// translation and checked against the Python by `qtrans verify`. Keep the `// qtrans:` lines.
// Anything above the first block (#include <...>, helper functions) goes into the header first.
"""


def block_text(name, head, body_lines, fp, accepted=None, comments=()) -> str:
    stamp = f"  python={fp}" + (f"  accepted={accepted}" if accepted else "")
    out = [f"// qtrans: override {name}{stamp}"] + list(comments) + [head.rstrip() + " {"]
    out += body_lines + ["}", f"// qtrans: end {name}"]
    return "\n".join(out)


def today() -> str:
    return _dt.date.today().isoformat()


def add_block(source_path, name, head, body_lines, fp, comments=()) -> Path:
    p = path_for(source_path)
    text = p.read_text(encoding="utf-8-sig") if p.is_file() else \
        HEADER.format(name=p.name, py=Path(source_path).name)
    if p.is_file() and name in parse(text, p).blocks:
        raise OverrideError(f"{p.name} already has a block for {name}()")
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + block_text(name, head, body_lines, fp, comments=comments) + "\n"
    p.write_text(text, encoding="utf-8")
    return p


def restamp(source_path, name, fp) -> Path:
    """Record that the block for `name` was checked against the Python whose fingerprint is fp."""
    p = path_for(source_path)
    lines = p.read_text(encoding="utf-8-sig").split("\n")
    for k, line in enumerate(lines):
        m = OPEN.match(line)
        if m and m.group(1) == name:
            lines[k] = f"// qtrans: override {name}  python={fp}  accepted={today()}"
            p.write_text("\n".join(lines), encoding="utf-8")
            return p
    raise OverrideError(f"{p.name} has no block for {name}()")


def remove_block(source_path, name) -> Path:
    p = path_for(source_path)
    lines = p.read_text(encoding="utf-8-sig").split("\n")
    start = next((k for k, l in enumerate(lines) if OPEN.match(l) and OPEN.match(l).group(1) == name), None)
    if start is None:
        raise OverrideError(f"{p.name} has no block for {name}()")
    end = next(k for k in range(start, len(lines)) if CLOSE.match(lines[k]))
    while start > 0 and not lines[start - 1].strip():
        start -= 1
    del lines[start:end + 1]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# --- qtrans override ... ------------------------------------------------------------------------------
def command(file, name=None, accept=False, remove=False, show=False, out=print) -> int:
    """`qtrans override FILE NAME`: start your own C++ for NAME from qtrans's translation.
    --accept: check it against the Python (qtrans verify) and, if it agrees, record that it was
    written for this Python. --remove: go back to qtrans's translation. --list: every block, and
    whether it is up to date."""
    from .project import translate_project
    from .translator import overrides_off
    src = Path(file).resolve()
    if show or name is None:
        of = load(src)
        if of is None:
            out(f"{path_for(src).name}: no overrides yet (qtrans override {src.name} FUNCTION starts one)")
            return 0
        res = translate_project([src], None).modules[src.stem]
        for n, ov in of.blocks.items():
            if n in getattr(res, "overrides", {}):
                ov = res.overrides[n]
                out(f"  {n:<24} {ov.status:<18} line {ov.line}")
            else:
                why = next((x for x in res.report.module_notes if f"{n}()" in x), None) or \
                    next((f.reason for f in res.report.functions if f.name == n and not f.translated), "not used")
                out(f"  {n:<24} {'NOT USED':<18} line {ov.line}: {why}")
        return 0
    if remove:
        p = remove_block(src, name)
        out(f"removed {name}() from {p.name}: qtrans's translation is used again")
        return 0
    if accept:
        from .verify import verify_file
        rep = verify_file(src)
        f = next((x for x in rep.functions if x.name == name), None)
        if f is None or f.override is None:
            out(f"{name}(): no override of it is in use (see `qtrans override {src.name} --list`)")
            return 1
        bad = f.count("mismatch")
        if bad:
            out(rep.to_text())
            out(f"\nnot accepted: your C++ for {name}() disagrees with the Python on {bad} of {len(f.cases)} "
                "inputs (above)")
            return 1
        restamp(src, name, f.override.current)
        out(f"accepted {name}(): {f.count('match')} of {len(f.cases)} inputs match the Python"
            + (f" ({f.count('illcond')} ill-conditioned)" if f.count("illcond") else "")
            + f"; {path_for(src).name} now records python={f.override.current}")
        return 0
    with overrides_off():
        res = translate_project([src], None).modules[src.stem]
    if name not in res.heads:
        out(f"{src.name} has no function {name}() that can have its own C++"
            + (f" ({res.rejections[name]})" if name in res.rejections else ""))
        return 1
    fp = res.fingerprint_of(name)
    if name in res.bodies:
        from .functions import strip_marks
        body = strip_marks("\n".join(res.bodies[name]))[0].split("\n")
        comments = [f"// started from qtrans's translation of {name}(); edit freely: `qtrans verify` checks it"]
    else:
        fdef = next(n for n in ast.parse(src.read_text(encoding="utf-8-sig")).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        py = ast.get_source_segment(src.read_text(encoding="utf-8-sig"), fdef) or ""
        body = [f"    // qtrans can't translate {name}() itself ({res.rejections.get(name, 'see the report')});",
                "    // write it here. The Python, for reference:"] + \
               [f"    //   {line}" for line in py.splitlines()] + \
               [f'    throw std::logic_error("{name}(): not written yet");']
        comments = []
    p = add_block(src, name, res.heads[name], body, fp, comments)
    out(f"wrote {p.name}: {name}() is now your C++ (from line {parse(p.read_text(encoding='utf-8'), p).blocks[name].line}); "
        f"edit it, run `qtrans verify {src.name}`, then `qtrans override --accept {src.name} {name}`")
    return 0
