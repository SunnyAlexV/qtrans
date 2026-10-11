"""Checking generated functions: translate, compile, verify, and say what went wrong.

A bug is one of:
  crash     the translator raised something other than a clean refusal (Unsupported), or a
            refusal that admits to being an internal error ("internal: ...")
  compile   the generated C++ doesn't compile
  warning   it compiles with a warning (the generated code is meant to be warning-free)
  mismatch  the verifier found a case where C++ and Python disagree (ill-conditioned cases and
            the documented differences are their own outcomes, not mismatches)
A refusal with a reason is not a bug: the generator writes plenty of things qtrans doesn't do.
"""
from __future__ import annotations

import math
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools.fuzz.gen import HEADER


@dataclass
class Bug:
    kind: str            # crash / compile / warning / mismatch
    name: str            # the function
    source: str          # the function's source
    detail: str          # exception, compiler output, or the disagreeing case
    args: tuple = None   # mismatch: the input that shows it
    profile: str = "strict"
    shrunk: str = None   # the minimal reproducer, once shrunk
    deps: dict = field(default_factory=dict)  # the functions it calls (kept as they are while shrinking)

    def signature(self):
        """What makes two bugs the same: for compile errors and warnings, the compiler's first
        message with names and positions taken out; for a crash, the exception."""
        import re
        if self.kind in ("compile", "warning"):
            m = re.search(r"(error|warning): (.*)", self.detail)
            msg = m.group(2) if m else ""
            msg = re.sub(r"\b[xnbvmpi]\d+\b|\bf\d+_\d+\b", "_", msg)
            return (self.kind, re.sub(r"\d+", "N", msg)[:120])
        if self.kind == "crash":
            return (self.kind, self.detail.strip().splitlines()[-1][:100])
        return (self.kind, self.name)


@dataclass
class Outcome:
    translated: list = field(default_factory=list)
    refused: dict = field(default_factory=dict)   # name -> reason
    bugs: list = field(default_factory=list)
    cases: int = 0


def literal(v) -> str:
    """Python source for an argument value (so a failing input can be frozen in QTRANS_CASES)."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return "float('nan')"
        if math.isinf(v):
            return "float('inf')" if v > 0 else "float('-inf')"
        return repr(v)
    if isinstance(v, np.ndarray):
        body = ", ".join(literal(x) for x in v.ravel())
        return f"np.array([{body}], dtype=float).reshape({tuple(v.shape)!r})"
    raise TypeError(type(v))


def module_source(functions: dict, cases: dict | None = None) -> str:
    src = HEADER + "\n\n" + "\n\n".join(functions.values())
    if cases:
        items = ", ".join(f"{n!r}: [({', '.join(literal(a) for a in args)},)]" for n, args in cases.items())
        src += f"\n\nQTRANS_CASES = {{{items}}}\n"
    return src


def _write(src: str, workdir: Path, stem: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"{stem}_", dir=workdir))
    p = d / f"{stem}.py"
    p.write_text(src, encoding="utf-8")
    return p


def translate_only(functions: dict, workdir: Path, stem="fz"):
    """-> (translation result or None, crash text or None)."""
    from qtrans.project import translate_project
    p = _write(module_source(functions), workdir, stem)
    try:
        r = translate_project([p]).modules[stem]
        return r, None
    except Exception:  # noqa: BLE001 - anything but a clean refusal is a translator bug
        return None, traceback.format_exc(limit=6)


def verify(functions: dict, workdir: Path, n_random: int, seed: int, profile="strict", cases=None, stem="fz"):
    """Run the verifier on these functions. -> (report or None, error text or None)."""
    from qtrans import verify_file
    p = _write(module_source(functions, cases), workdir, stem)
    try:
        return verify_file(p, n_random=n_random, seed=seed, build_dir=p.parent / "build", profile=profile), None
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(limit=8)


def with_callees(name: str, functions: dict) -> dict:
    """name and every function of the batch it calls, directly or not (callees first)."""
    import re
    out, todo = {}, [name]
    while todo:
        n = todo.pop()
        if n in out or n not in functions:
            continue
        out[n] = functions[n]
        todo += [c for c in re.findall(r"\b(f\d+_\d+)\(", functions[n]) if c != n]
    return {k: out[k] for k in functions if k in out}  # in the module's order (callees are earlier)


def _internal(reason: str) -> bool:
    return "internal" in reason.lower()


def check(functions: dict, workdir: Path, n_random: int, seed: int, profile="strict") -> Outcome:
    """Check a batch; isolate the function behind any batch-level failure."""
    out = Outcome()
    tr, crash = translate_only(functions, workdir)
    if crash:  # find which function(s): translate each alone
        keep = {}
        for n, src in functions.items():
            group = with_callees(n, functions)
            _, c = translate_only(group, workdir)
            if c and not any(b.name in group for b in out.bugs):
                out.bugs.append(Bug("crash", n, src, c, profile=profile, deps=_deps(n, group)))
            elif not c:
                keep[n] = src
        functions = keep
        if not functions:
            return out
        tr, crash = translate_only(functions, workdir)
        if crash:  # only fails together: report the batch's first function with the trace
            n = next(iter(functions))
            out.bugs.append(Bug("crash", n, functions[n], crash, profile=profile))
            return out
    for f in tr.report.rejected:
        if _internal(f.reason):
            out.bugs.append(Bug("crash", f.name, functions[f.name], f.reason, profile=profile))
        out.refused[f.name] = f.reason
    ok = {n: s for n, s in functions.items() if n not in out.refused}
    if not ok:
        return out
    rep, err = verify(ok, workdir, n_random, seed, profile)
    if err:  # a compile error (or a verifier failure): which function? Bisect, then check each suspect
        suspects = _bisect(list(ok), ok, workdir, seed, profile)
        if not suspects:  # fails only together: report the batch as one bug, the last function first
            last = list(ok)[-1]
            kind = "compile" if "compilation failed" in err else "crash"
            out.bugs.append(Bug(kind, last, ok[last], err, profile=profile, deps=_deps(last, ok)))
            return out
        for n, src in ok.items():
            if n not in suspects:
                continue
            group = with_callees(n, ok)
            if any(b.name in group for b in out.bugs):
                continue  # a callee's bug, already found
            r1, e1 = verify(group, workdir, 3, seed, profile)
            if e1:
                kind = "compile" if "compilation failed" in e1 else "crash"
                out.bugs.append(Bug(kind, n, src, e1, profile=profile, deps=_deps(n, group)))
            elif _warnings_of(r1, n):
                out.bugs.append(Bug("warning", n, src, "\n".join(r1.warnings), profile=profile, deps=_deps(n, group)))
        return out
    if rep.warnings:
        for n, src in ok.items():
            group = with_callees(n, ok)
            if any(b.name in group for b in out.bugs):
                continue
            r1, _ = verify(group, workdir, 3, seed, profile)
            if r1 is not None and _warnings_of(r1, n):
                out.bugs.append(Bug("warning", n, src, "\n".join(r1.warnings), profile=profile, deps=_deps(n, group)))
    out.translated = list(ok)
    for f in rep.functions:
        out.cases += len(f.cases)
        bad = [c for c in f.cases if c.outcome == "mismatch"]
        if bad:
            c = bad[0]
            out.bugs.append(Bug("mismatch", f.name, ok[f.name],
                                f"{f.name}{c.args!r}: Python {c.py!r}, C++ {c.cpp!r}", args=tuple(c.args),
                                profile=profile, deps=_deps(f.name, with_callees(f.name, ok))))
    return out


def _bisect(names, functions, workdir, seed, profile) -> set:
    """The functions (with their callees) that fail to build on their own: halves of the batch
    are compiled, and only a failing half is split further."""
    group = {}
    for n in names:
        group.update(with_callees(n, functions))
    group = {k: functions[k] for k in functions if k in group}
    _, err = verify(group, workdir, 1, seed, profile)
    if not err:
        return set()
    if len(names) == 1:
        return set(names)
    mid = len(names) // 2
    return _bisect(names[:mid], functions, workdir, seed, profile) | _bisect(names[mid:], functions, workdir, seed, profile)


def _deps(name, group):
    return {k: v for k, v in group.items() if k != name}


def _warnings_of(rep, name) -> bool:
    """Warnings in the function's own C++ (its callees' are theirs)."""
    import re
    if rep is None or not rep.warnings:
        return False
    header = rep.translation.header.splitlines()
    for w in rep.warnings:
        m = re.search(r"\.hpp:(\d+):", w)
        if not m:
            return True
        k = int(m.group(1)) - 1
        while k >= 0 and not re.match(r"inline .*\b(\w+)\(", header[k]):
            k -= 1
        fn = re.match(r"inline .*?\b(\w+)\(", header[k]).group(1) if k >= 0 else None
        if fn in (None, name):
            return True
    return False


def still_fails(bug: Bug, source: str, args, workdir: Path) -> bool:
    """Does this (smaller) version of the function fail the same way? For shrinking."""
    name = bug.name
    funcs = {**bug.deps, name: source}
    if bug.kind == "crash":
        tr, crash = translate_only(funcs, workdir)
        if crash:
            return crash.strip().splitlines()[-1].split(":")[0] == bug.detail.strip().splitlines()[-1].split(":")[0]
        if tr is not None:
            return any(f.name == name and _internal(f.reason) for f in tr.report.rejected)
        return False
    tr, crash = translate_only(funcs, workdir)
    if crash or tr is None or any(f.name == name for f in tr.report.rejected):
        return False  # must still translate (a refusal is no longer the bug)
    if bug.kind in ("compile", "warning"):
        rep, err = verify(funcs, workdir, 0, 0, bug.profile)
        if bug.kind == "compile":
            return err is not None and "compilation failed" in err
        return _warnings_of(rep, name)
    rep, err = verify(funcs, workdir, 0, 0, bug.profile, cases={name: args})
    if rep is None:
        return False
    return any(c.outcome == "mismatch" for f in rep.functions if f.name == name for c in f.cases)
