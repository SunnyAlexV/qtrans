"""Shrinking a failing function to a minimal one that fails the same way (delta debugging on the
syntax tree).

Each round proposes smaller versions of the function, biggest cuts first: delete a statement,
replace an `if` by one of its branches, replace a `for` by its body with the loop variable set
to 0, replace an expression by one of its parts or by a constant, drop a parameter the body no
longer uses (with its argument), simplify a number. The first proposal that still fails the
same way (check.still_fails, on the same frozen input) is kept and the round starts again;
when none does, the function is minimal for these moves. Candidates that aren't valid any more
(wrong types, a name used before it is set) simply don't fail the same way and are dropped.
"""
from __future__ import annotations

import ast
import copy
import time

from tools.fuzz.check import Bug, still_fails


def _size(src: str) -> tuple:
    return (sum(1 for _ in ast.walk(ast.parse(src))), len(src))


def _blocks(fn):
    """Every statement list in the function (bodies of if / for / else)."""
    out = []
    for node in ast.walk(fn):
        for attr in ("body", "orelse"):
            b = getattr(node, attr, None)
            if isinstance(b, list) and b and isinstance(b[0], ast.stmt):
                out.append(b)
    return out


class _Sub(ast.NodeTransformer):
    def __init__(self, name, value):
        self.name, self.value = name, value

    def visit_Name(self, n):
        return copy.deepcopy(self.value) if n.id == self.name and isinstance(n.ctx, ast.Load) else n


def _candidates(src: str):
    """Smaller versions of the function, biggest cuts first. Yields (source, removed param index or None)."""
    tree = ast.parse(src)
    fn = tree.body[0]
    # 1. delete a run of statements: halves, then quarters, ... then single ones (ddmin), so a
    #    long function loses most of itself in a few tries (an emptied block gets `pass`)
    for bi, block in enumerate(_blocks(fn)):
        size = max(len(block) // 2, 1)
        while size >= 1:
            for start in range(0, len(block), size):
                t = copy.deepcopy(tree)
                b = _blocks(t.body[0])[bi]
                del b[start:start + size]
                if not b:
                    b.append(ast.Pass())
                yield ast.unparse(t), None
            size //= 2
    # 2. an if by its body or its else; a for by its body with the loop variable = 0
    for bi, block in enumerate(_blocks(fn)):
        for si, s in enumerate(block):
            if isinstance(s, ast.If):
                for branch in ("body", "orelse"):
                    t = copy.deepcopy(tree)
                    b = _blocks(t.body[0])[bi]
                    repl = getattr(b[si], branch) or [ast.Pass()]
                    b[si:si + 1] = repl
                    yield ast.unparse(t), None
            if isinstance(s, ast.For) and isinstance(s.target, ast.Name):
                t = copy.deepcopy(tree)
                b = _blocks(t.body[0])[bi]
                body = [_Sub(s.target.id, ast.Constant(0)).visit(x) for x in b[si].body]
                b[si:si + 1] = body
                yield ast.unparse(t), None
    # 3. an expression by one of its parts, or by a constant
    exprs = [n for n in ast.walk(fn) if isinstance(n, ast.expr) and not isinstance(n, ast.Constant)
             and not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))]
    for k in range(len(exprs)):
        node = exprs[k]
        parts = [c for c in ast.iter_child_nodes(node) if isinstance(c, ast.expr)
                 and not isinstance(c, ast.Slice) and not (isinstance(c, ast.Tuple) and isinstance(node, ast.Subscript))]
        if isinstance(node, ast.Call):
            parts = list(node.args)
        if isinstance(node, ast.Attribute):
            parts = []
        for repl in parts + [ast.Constant(0.0), ast.Constant(1.0), ast.Constant(0), ast.Constant(1)]:
            t = copy.deepcopy(tree)
            target = [n for n in ast.walk(t.body[0]) if isinstance(n, ast.expr) and not isinstance(n, ast.Constant)
                      and not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))][k]
            if _replace(t, target, copy.deepcopy(repl)):
                yield ast.unparse(t), None
    # 4. drop a parameter the body doesn't use
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    for i, a in enumerate(fn.args.args):
        if a.arg not in used:
            t = copy.deepcopy(tree)
            del t.body[0].args.args[i]
            yield ast.unparse(t), i
    # 5. simpler numbers
    consts = [n for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
              and not isinstance(n.value, bool)]
    for k, c in enumerate(consts):
        for v in ([0, 1] if isinstance(c.value, int) else [0.0, 1.0, float(round(c.value))]):
            if v == c.value and type(v) is type(c.value):
                continue
            t = copy.deepcopy(tree)
            target = [n for n in ast.walk(t.body[0]) if isinstance(n, ast.Constant)
                      and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)][k]
            target.value = v
            yield ast.unparse(t), None


def _replace(tree, target, repl) -> bool:
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is target:
                setattr(parent, field, repl)
                return True
            if isinstance(value, list):
                for i, x in enumerate(value):
                    if x is target:
                        value[i] = repl
                        return True
    return False


def shrink(bug: Bug, workdir, budget_seconds=900, log=print) -> Bug:
    """Shrink bug.source in place of the original; returns the bug with .shrunk (and .args) set."""
    src, args = ast.unparse(ast.parse(bug.source)), bug.args
    if not still_fails(bug, src, args, workdir):
        bug.shrunk = None  # doesn't reproduce on its own (depends on the batch): leave as found
        return bug
    seen, t0, steps = {src}, time.time(), 0
    progress = True
    while progress and time.time() - t0 < budget_seconds:
        progress = False
        for cand, dropped in _candidates(src):
            if time.time() - t0 > budget_seconds:
                break
            if cand in seen:
                continue
            try:
                if _size(cand) >= _size(src):
                    continue
            except SyntaxError:  # e.g. A[:, j] with the subscript replaced by its index tuple
                seen.add(cand)
                continue
            seen.add(cand)
            a = args if dropped is None or args is None else args[:dropped] + args[dropped + 1:]
            steps += 1
            if still_fails(bug, cand, a, workdir):
                src, args, progress = cand, a, True
                log(f"    shrunk to {_size(src)[0]} nodes")
                break
    bug.shrunk, bug.args = src, args
    log(f"    {steps} tries, {time.time() - t0:.0f} s")
    return bug
