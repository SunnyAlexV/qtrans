"""Reads of a local variable that may not be assigned yet.

Python raises UnboundLocalError (NameError at a program's top level) when a path reaches a read of
a variable that no assignment on that path has set:

    if x > 2.0:
        pass
    else:
        n = 3
    return n          # UnboundLocalError when x > 2.0

The C++ declares every variable up front (`long long n = 0;`) and would return 0 there, so such a
read is refused. This is definite assignment, the way Python runs the code: a loop's body may run
zero times; `while True:` is left only through a `break`; `return`, `raise`, `break` and
`continue` end a path; a `try` block may stop before any of its statements, so its handlers
start from what was set before it; `except ... as e` unbinds e when the handler ends.
"""
from __future__ import annotations

import ast

ALL = "<every name>"  # after return / raise: no path continues, so anything counts as set


def _meet(a, b):
    if a is ALL:
        return b
    if b is ALL:
        return a
    return a & b


def _targets(t):
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.Starred) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
    return out


class _Checker:
    def __init__(self, locals_):
        self.locals = locals_
        self.bad = []          # (node, name): a read that may find the name unset
        self.breaks = []       # per enclosing loop: the states at its `break`s

    # --- expressions -------------------------------------------------------------------------
    def reads(self, node, state, bound=frozenset()):
        if node is None:
            return state
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = set(bound)
            for g in node.generators:
                state = self.reads(g.iter, state, frozenset(inner))
                inner |= _targets(g.target)
                for c in g.ifs:
                    self.reads(c, state, frozenset(inner))
            parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            for p in parts:
                self.reads(p, state, frozenset(inner))
            return state
        if isinstance(node, ast.Lambda):
            args = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
            return self.reads(node.body, state, frozenset(bound | args))
        if isinstance(node, ast.NamedExpr):
            state = self.reads(node.value, state, bound)
            return state if state is ALL else state | {node.target.id}
        if isinstance(node, ast.BoolOp):  # a and b: b may not run; its walrus doesn't count after
            state = self.reads(node.values[0], state, bound)
            for v in node.values[1:]:
                self.reads(v, state, bound)
            return state
        if isinstance(node, ast.IfExp):
            state = self.reads(node.test, state, bound)
            return _meet(self.reads(node.body, state, bound), self.reads(node.orelse, state, bound))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in self.locals and node.id not in bound and state is not ALL and node.id not in state:
                self.bad.append((node, node.id))
            return state
        for c in ast.iter_child_nodes(node):
            state = self.reads(c, state, bound)
        return state

    # --- statements -------------------------------------------------------------------------
    def block(self, body, state):
        for s in body:
            state = self.stmt(s, state)
        return state

    def stmt(self, s, state):
        if state is ALL:
            return ALL  # (unreachable)
        if isinstance(s, ast.Assign):
            state = self.reads(s.value, state)
            for t in s.targets:
                state = self.reads_in_target(t, state)
            return state | set().union(*(_targets(t) for t in s.targets))
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                return state
            state = self.reads(s.value, state)
            state = self.reads_in_target(s.target, state)
            return state | _targets(s.target)
        if isinstance(s, ast.AugAssign):
            t = s.target
            if isinstance(t, ast.Name):  # x += 1 reads x first (reported at the target itself)
                if t.id in self.locals and state is not ALL and t.id not in state:
                    self.bad.append((t, t.id))
            else:
                state = self.reads_in_target(t, state, load=True)
            return self.reads(s.value, state) | _targets(t)
        if isinstance(s, (ast.Return, ast.Raise)):
            for c in ast.iter_child_nodes(s):
                self.reads(c, state)
            return ALL
        if isinstance(s, ast.Break):
            if self.breaks:
                self.breaks[-1].append(state)
            return ALL
        if isinstance(s, ast.Continue):
            return ALL
        if isinstance(s, ast.If):
            state = self.reads(s.test, state)
            return _meet(self.block(s.body, state), self.block(s.orelse, state))
        if isinstance(s, (ast.For, ast.AsyncFor)):
            state = self.reads(s.iter, state)
            self.breaks.append([])
            self.block(s.body, state | _targets(s.target))
            after = self.block(s.orelse, state)  # (the body may run zero times; else runs when no break)
            for b in self.breaks.pop():
                after = _meet(after, b)
            return after
        if isinstance(s, ast.While):
            state = self.reads(s.test, state)
            self.breaks.append([])
            self.block(s.body, state)
            brk = self.breaks.pop()
            forever = isinstance(s.test, ast.Constant) and bool(s.test.value)
            after = ALL if forever else self.block(s.orelse, state)
            for b in brk:
                after = _meet(after, b)
            return after
        if isinstance(s, ast.Try):
            body = self.block(s.body, state)
            outs = [self.block(s.orelse, body)]
            for h in s.handlers:
                if h.type is not None:
                    self.reads(h.type, state)
                hs = state | ({h.name} if h.name else set())
                out = self.block(h.body, hs)
                if h.name and out is not ALL:
                    out = out - {h.name}  # Python deletes it when the handler ends
                outs.append(out)
            merged = ALL
            for o in outs:
                merged = _meet(merged, o)
            if s.finalbody:
                fin = self.block(s.finalbody, state)  # (it may run after an error anywhere)
                if merged is ALL or fin is ALL:
                    return ALL if merged is ALL else merged
                return merged | (fin - state)
            return merged
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                state = self.reads(it.context_expr, state)
                if it.optional_vars is not None:
                    state = state | _targets(it.optional_vars)
            return self.block(s.body, state)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return state | {s.name}
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            return state | {(a.asname or a.name).split(".")[0] for a in s.names}
        if isinstance(s, (ast.Global, ast.Nonlocal, ast.Pass)):
            return state
        if isinstance(s, ast.Delete):
            for t in s.targets:
                state = self.reads_in_target(t, state, load=True)
            return state - {t.id for t in s.targets if isinstance(t, ast.Name)}
        for c in ast.iter_child_nodes(s):  # Expr, Assert, ...
            state = self.reads(c, state)
        return state

    def reads_in_target(self, t, state, load=False):
        """xs[i] = v reads xs and i; a.b = v reads a; x = v reads nothing."""
        if isinstance(t, ast.Name):
            if load:
                self.reads(ast.Name(id=t.id, ctx=ast.Load(), lineno=t.lineno, col_offset=t.col_offset), state)
            return state
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                state = self.reads_in_target(e, state, load)
            return state
        if isinstance(t, ast.Starred):
            return self.reads_in_target(t.value, state, load)
        if isinstance(t, ast.Subscript):
            state = self.reads(t.value, state)
            return self.reads(t.slice, state)
        if isinstance(t, ast.Attribute):
            return self.reads(t.value, state)
        return state


def possibly_unbound(body, params=(), known=()):
    """[(node, name)]: reads in `body` (a function's statements, or a program's) of a local that
    may not be set yet. params and known (module names, for a program) are set from the start."""
    assigned = set()
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n not in body:
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                assigned.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                assigned.add(n.name)
            elif isinstance(n, ast.NamedExpr):
                assigned.add(n.target.id)
    # comprehension variables are their comprehension's own, not the function's
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for g in n.generators:
                    for name in _targets(g.target):
                        if not any(isinstance(m, ast.Name) and m.id == name and isinstance(m.ctx, ast.Store)
                                   and not _inside_comprehension(m, body) for m in _all_nodes(body)):
                            assigned.discard(name)
    c = _Checker(assigned - set(params) - set(known))
    c.block(body, set(params) | set(known))
    return c.bad


def _all_nodes(body):
    for s in body:
        yield from ast.walk(s)


def _inside_comprehension(node, body):
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                if any(m is node for g in n.generators for m in ast.walk(g.target)):
                    return True
    return False
