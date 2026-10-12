"""enumerate() and zip() loops, rewritten as the index loops a C++ programmer writes.

    for i, x in enumerate(xs):            for i in range(len(xs)):
        ...                        ->         x = xs[i]
                                              ...
    for a, b in zip(xs, ys):              for k in range(min(len(xs), len(ys))):
        ...                        ->         a = xs[k]
                                              b = ys[k]
                                              ...

and the same inside comprehensions ([a * b for a, b in zip(xs, ys)]). This runs on the Python
syntax tree before translation, so the rest of qtrans only ever sees range() loops, which it
already translates and checks.

Why this is exact: Python's list and array iterators read element k by position, one step at a
time, so `x = xs[k]` reads the same value the iterator would, even when the body changes
xs[j] for some j. What would differ is a loop that changes the LENGTH of a list it goes
through (append / pop / ...) or rebinds its name: those are left as they are, and refused
with the reason recorded here.

A generator expression passed to sum / min / max / sorted / list / str.join is read to the end
by Python anyway, so it becomes the list comprehension it reads the same as. any() / all()
stop early, so those keep their generator (lists.py translates them as a loop that returns).
"""
from __future__ import annotations

import ast

LENGTH_CHANGERS = {"append", "extend", "insert", "pop", "remove", "clear"}
READS_TO_THE_END = {"sum", "min", "max", "sorted", "list"}
WHY = "qt_why"  # attribute set on a loop left alone: why it couldn't be rewritten


def _names_in(tree) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _defined(tree) -> set:
    """Names the module itself binds (a user's own zip() or len() turns the rewrite off)."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[0])
    return out


def _call_of(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _loc(new, old):
    ast.copy_location(new, old)
    new.end_lineno, new.end_col_offset = old.lineno, old.col_offset
    for n in ast.walk(new):
        if not hasattr(n, "lineno") and "lineno" in n._attributes:
            n.lineno, n.col_offset = old.lineno, old.col_offset
            n.end_lineno, n.end_col_offset = old.lineno, old.col_offset
    return new


def _name(id_, store=False):
    return ast.Name(id=id_, ctx=ast.Store() if store else ast.Load())


def _name_view(s):
    """xs -> ("xs", 0, 0); xs[1:] -> ("xs", 1, 0); xs[:-1] -> ("xs", 0, 1); xs[2:-1] -> ("xs", 2, 1):
    element k of it is xs[k + first], and it has len(xs) - first - cut elements. None: not that."""
    if isinstance(s, ast.Name):
        return s.id, 0, 0
    if not (isinstance(s, ast.Subscript) and isinstance(s.value, ast.Name) and isinstance(s.slice, ast.Slice)
            and s.slice.step is None):
        return None
    lo, hi = s.slice.lower, s.slice.upper
    first = 0 if lo is None else (lo.value if isinstance(lo, ast.Constant) and type(lo.value) is int
                                  and lo.value >= 0 else None)
    cut = 0
    if hi is not None:
        if isinstance(hi, ast.UnaryOp) and isinstance(hi.op, ast.USub) and isinstance(hi.operand, ast.Constant) \
                and type(hi.operand.value) is int and hi.operand.value > 0:
            cut = hi.operand.value
        else:
            return None
    return None if first is None else (s.value.id, first, cut)


class _Plan:
    """What one enumerate/zip loop becomes: the sequences it reads, how each target name is
    bound (to `seq[k]`, or to `k + start`), and the reason when it can't be rewritten."""

    def __init__(self):
        self.seqs: list = []          # the sequence expressions, in order
        self.binds: list = []         # (target name, ("item", seq index) | ("index", start node))
        self.why: str | None = None


def _plan(target, it, builtins_ok) -> _Plan | None:
    """None: not an enumerate/zip loop over a tuple of names (nothing to rewrite)."""
    if not (isinstance(target, ast.Tuple) and (_call_of(it, "enumerate") or _call_of(it, "zip"))):
        return None
    p = _Plan()
    if not builtins_ok:
        p.why = "this module defines its own enumerate, zip, len, min or range"
        return p

    def add(t, it):
        if _call_of(it, "enumerate"):
            start = None
            args = list(it.args)
            for k in it.keywords:
                if k.arg != "start":
                    p.why = f"enumerate() has no '{k.arg}' argument"
                    return
                start = k.value
            if len(args) == 2 and start is None:
                start = args.pop()
            if len(args) != 1:
                p.why = "enumerate() takes one sequence (and a start)"
                return
            if not (isinstance(t, ast.Tuple) and len(t.elts) == 2):
                p.why = "unpack enumerate() into two names: for i, x in enumerate(xs)"
                return
            bind(t.elts[0], ("index", start))
            seq(t.elts[1], args[0])
        elif _call_of(it, "zip"):
            if it.keywords:
                p.why = "zip(..., strict=True) is not supported" if any(k.arg == "strict" for k in it.keywords) \
                    else "zip() takes only sequences"
                return
            if any(isinstance(a, ast.Starred) for a in it.args) or not it.args:
                p.why = "zip() needs its sequences written out"
                return
            if not (isinstance(t, ast.Tuple) and len(t.elts) == len(it.args)):
                p.why = f"unpack zip() into {len(it.args)} names, one per sequence"
                return
            for e, a in zip(t.elts, it.args):
                seq(e, a)

    def seq(t, a):
        if _call_of(a, "enumerate") or _call_of(a, "zip"):
            if _call_of(a, "zip") and _call_of(it, "zip"):
                p.why = "zip() inside zip() is not supported; zip all the sequences at once"
                return
            add(t, a)  # enumerate(zip(xs, ys)) or zip(xs, enumerate(ys))
            return
        if _call_of(a, "range"):
            p.why = "a range() inside enumerate() or zip() is not supported; loop over range() and index"
            return
        p.seqs.append(a)
        bind(t, ("item", len(p.seqs) - 1))

    def bind(t, how):
        if not isinstance(t, ast.Name):
            p.why = p.why or "the loop must unpack into plain names"
            return
        p.binds.append((t.id, how))

    add(target, it)
    if p.why is None and len({n for n, _ in p.binds if n != "_"}) != len([n for n, _ in p.binds if n != "_"]):
        p.why = "a name appears twice in the loop target"
    return p


def _changes(body, seq_name) -> str | None:
    """How a loop body changes the length of list `seq_name`, or rebinds it (None: it doesn't)."""
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Name) and n.id == seq_name and isinstance(n.ctx, (ast.Store, ast.Del)):
            return f"the loop assigns '{seq_name}' while going through it"
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == seq_name:
            return f"the loop changes '{seq_name}' while going through it"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in LENGTH_CHANGERS \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == seq_name:
            return f"the loop changes the length of '{seq_name}' ({n.func.attr}) while going through it"
        if isinstance(n, ast.Delete) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                             and t.value.id == seq_name for t in n.targets):
            return f"the loop deletes from '{seq_name}' while going through it"
    return None


class Rewriter(ast.NodeTransformer):
    def __init__(self, tree):
        self.taken = _names_in(tree)
        self.top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | \
            {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
             for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
        self.direct = self._direct_indexes(tree)
        defined = _defined(tree)
        self.builtins_ok = not (defined & {"enumerate", "zip", "len", "min", "range"})
        self.generators_ok = not (defined & READS_TO_THE_END)
        self.scope = [tree]  # the function (or module) whose body is being rewritten

    def fresh(self, base):
        """A new name for a temporary (unique in the module)."""
        name, k = base, 2
        while name in self.taken:
            name, k = f"{base}{k}", k + 1
        self.taken.add(name)
        return name

    def fresh_index(self, node):
        """A loop index name nothing in the function (nor the rewritten loop) uses: i, j, k, ..."""
        scope = self.scope[-1]
        used = _names_in(node) | (self.taken if isinstance(scope, ast.Module) else _names_in(scope) | self.top)
        for name in ("i", "j", "k", "n", "idx"):
            if name not in used:
                return name
        k = 2
        while f"i{k}" in used:
            k += 1
        return f"i{k}"

    @staticmethod
    def _direct_indexes(tree) -> set:
        """ids of the `for i, x in enumerate(xs)` loops whose `i` can be the C++ loop index
        itself: every use of i in its function is inside a loop that binds i, and no loop
        body assigns it."""
        out = set()
        scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for sc in scopes:
            own = []  # nodes of this scope (not of functions inside it)
            stack = list(sc.body)
            while stack:
                n = stack.pop()
                own.append(n)
                if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                    stack.append(n.generators[0].iter)  # the rest is the comprehension's own scope
                elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    stack.extend(ast.iter_child_nodes(n))
            fors = [n for n in own if isinstance(n, ast.For)]
            for f in fors:
                if not (isinstance(f.target, ast.Tuple) and len(f.target.elts) == 2 and _call_of(f.iter, "enumerate")
                        and isinstance(f.target.elts[0], ast.Name)):
                    continue
                i = f.target.elts[0].id
                covered = set()
                def index_of_its_own(g):
                    """g binds i as its loop index: `for i in ...`, or enumerate from 0 (or none)"""
                    if isinstance(g.target, ast.Name):
                        return True
                    if not (_call_of(g.iter, "enumerate") and isinstance(g.target, ast.Tuple)
                            and len(g.target.elts) == 2 and isinstance(g.target.elts[0], ast.Name)
                            and g.target.elts[0].id == i and not any(isinstance(t, ast.Name) and t.id == i
                                                                     for t in ast.walk(g.target.elts[1]))):
                        return False
                    st = g.iter.args[1] if len(g.iter.args) == 2 else next(
                        (k.value for k in g.iter.keywords if k.arg == "start"), None)
                    return st is None or (isinstance(st, ast.Constant) and st.value == 0)
                for g in fors:
                    if any(isinstance(t, ast.Name) and t.id == i for t in ast.walk(g.target)):
                        if not index_of_its_own(g):
                            covered = None
                            break
                        covered |= {id(x) for part in [g.target] + g.body for x in ast.walk(part)}
                        if any(isinstance(x, ast.Name) and x.id == i and isinstance(x.ctx, ast.Store)
                               for b in g.body for x in ast.walk(b)):
                            covered = None
                            break
                if covered is not None and all(id(n) in covered for n in own if isinstance(n, ast.Name) and n.id == i):
                    out.add(id(f))
        return out

    def _scope_names_outside(self, loop) -> set:
        """Names used in the enclosing function (or module code) outside this loop."""
        inside = {id(n) for n in ast.walk(loop)}
        return {n.id for n in ast.walk(self.scope[-1]) if isinstance(n, ast.Name) and id(n) not in inside}

    def visit_FunctionDef(self, node):
        self.scope.append(node)
        self.generic_visit(node)
        self.scope.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _length(self, seqs):
        lens = [ast.Call(func=_name("len"), args=[s], keywords=[]) for s in seqs]
        return lens[0] if len(lens) == 1 else ast.Call(func=_name("min"), args=lens, keywords=[])

    # --- for-loops ----------------------------------------------------------------------------
    def visit_For(self, node):
        self.generic_visit(node)
        p = _plan(node.target, node.iter, self.builtins_ok)
        if p is None:
            return node
        if p.why is None:
            for s in p.seqs:
                if isinstance(s, ast.Name):
                    p.why = _changes(node.body + node.orelse, s.id)
                    if p.why:
                        break
        if p.why is not None:
            setattr(node, WHY, p.why)
            return node
        before = []
        seqs = []
        for k, s in enumerate(p.seqs):
            if isinstance(s, ast.Name):
                seqs.append(s)
                continue
            # zip(xs[1:], xs[:-1]), enumerate(f(x)): Python evaluates the sequence once, before
            # the loop; so does a named copy
            owner = next((n for n, how in p.binds if how == ("item", k) and n != "_"), "seq")
            tmp = self.fresh(f"{owner}_values")
            value = ast.List(elts=s.elts, ctx=ast.Load()) if isinstance(s, ast.Tuple) else s
            before.append(_loc(ast.Assign(targets=[_name(tmp, True)], value=value), node))
            seqs.append(_name(tmp))
        outside = self._scope_names_outside(node)
        index = next(((n, how[1]) for n, how in p.binds if how[0] == "index"), None)
        start_zero = index is None or index[1] is None or (isinstance(index[1], ast.Constant) and index[1].value == 0)
        # `for i, x in enumerate(xs)`: i itself is the loop index, unless it is used elsewhere
        # (read after the loop) or counts from another start
        if index is not None and index[0] != "_" and start_zero and id(node) in self.direct:
            k = index[0]
        else:
            k = self.fresh_index(node)
        assigns = []
        for name, how in p.binds:
            if name == "_" and "_" not in outside:
                continue
            if how[0] == "item":
                value = ast.Subscript(value=seqs[how[1]], slice=_name(k), ctx=ast.Load())
            elif name == k:
                continue
            elif start_zero:
                value = _name(k)
            else:
                value = ast.BinOp(left=_name(k), op=ast.Add(), right=how[1])
            assigns.append(_loc(ast.Assign(targets=[_name(name, True)], value=value), node))
        if not start_zero and not (isinstance(index[1], ast.Constant) or isinstance(index[1], ast.Name)):
            # enumerate(xs, f()): the start is evaluated once, before the loop
            tmp = self.fresh("start")
            before.append(_loc(ast.Assign(targets=[_name(tmp, True)], value=index[1]), node))
            for a in assigns:
                if isinstance(a.value, ast.BinOp):
                    a.value.right = _name(tmp)
        loop = ast.For(target=_name(k, True), iter=ast.Call(func=_name("range"), args=[self._length(seqs)], keywords=[]),
                       body=assigns + node.body, orelse=node.orelse, type_comment=None)
        _loc(loop, node)
        loop.end_lineno, loop.end_col_offset = node.end_lineno, node.end_col_offset
        loop.qt_rewritten = ast.unparse(node.iter)
        return before + [loop] if before else loop

    # --- comprehensions -----------------------------------------------------------------------
    def _comprehension(self, node):
        self.generic_visit(node)
        if len(node.generators) != 1:
            return node
        g = node.generators[0]
        p = _plan(g.target, g.iter, self.builtins_ok)
        if p is None:
            return node
        views = [_name_view(s) for s in p.seqs]
        if p.why is None and None in views:
            p.why = ("in a comprehension, enumerate() and zip() take names or plain slices of them "
                     "(xs, xs[1:], xs[:-1]); name the sequence first")
        parts = [node.elt] + list(g.ifs) if not isinstance(node, ast.DictComp) else [node.key, node.value] + list(g.ifs)
        if p.why is None and any(isinstance(n, (ast.NamedExpr, ast.Lambda, ast.ListComp, ast.GeneratorExp,
                                                ast.SetComp, ast.DictComp)) for e in parts for n in ast.walk(e)):
            p.why = "the comprehension holds another comprehension, a lambda or :="
        index = next(((n, how[1]) for n, how in p.binds if how[0] == "index"), None)
        if p.why is None and index is not None and index[1] is not None \
                and not isinstance(index[1], (ast.Constant, ast.Name)):
            p.why = "enumerate()'s start must be a number or a name here"
        if p.why is not None:
            setattr(g, WHY, p.why)
            return node
        index_zero = index is not None and index[0] != "_" and (
            index[1] is None or (isinstance(index[1], ast.Constant) and index[1].value == 0))
        k = index[0] if index_zero and index[0] not in {s.id for s in p.seqs} else self.fresh_index(node)
        subst = {}
        for name, how in p.binds:
            if name == k:
                continue
            if how[0] == "item":
                def item(view=views[how[1]]):
                    name_, lo, _ = view
                    at = _name(k) if lo == 0 else ast.BinOp(left=_name(k), op=ast.Add(), right=ast.Constant(lo))
                    return ast.Subscript(value=_name(name_), slice=at, ctx=ast.Load())
                subst[name] = item
            elif how[1] is None or (isinstance(how[1], ast.Constant) and how[1].value == 0):
                subst[name] = lambda: _name(k)
            else:
                subst[name] = lambda st=how[1]: ast.BinOp(left=_name(k), op=ast.Add(), right=st)

        class Sub(ast.NodeTransformer):
            def visit_Name(self, n):
                if n.id in subst and isinstance(n.ctx, ast.Load):
                    return ast.copy_location(subst[n.id](), n)
                return n

        g.ifs = [Sub().visit(c) for c in g.ifs]
        if isinstance(node, ast.DictComp):
            node.key, node.value = Sub().visit(node.key), Sub().visit(node.value)
        else:
            node.elt = Sub().visit(node.elt)
        g.target = _name(k, True)
        g.qt_rewritten = True
        lens = []
        for name_, lo, cut in views:  # len(xs[1:]) is len(xs) - 1 (range() of a negative is empty)
            n = ast.Call(func=_name("len"), args=[_name(name_)], keywords=[])
            if lo + cut:
                n = ast.BinOp(left=n, op=ast.Sub(), right=ast.Constant(lo + cut))
            lens.append(n)
        g.iter = ast.Call(func=_name("range"), args=[lens[0] if len(lens) == 1 else
                                                     ast.Call(func=_name("min"), args=lens, keywords=[])], keywords=[])
        for n in ast.walk(node):
            if "lineno" in n._attributes and not hasattr(n, "lineno"):
                n.lineno, n.col_offset, n.end_lineno, n.end_col_offset = \
                    node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
        return node

    visit_ListComp = visit_SetComp = visit_DictComp = _comprehension

    def visit_GeneratorExp(self, node):
        return self._comprehension(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        # sum(x * w for x, w in zip(xs, ws)) reads the generator to the end: it is the list
        reads_all = (isinstance(node.func, ast.Name) and node.func.id in READS_TO_THE_END and self.generators_ok) \
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "join")
        if reads_all and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            gen = node.args[0]
            node.args[0] = ast.copy_location(ast.ListComp(elt=gen.elt, generators=gen.generators), gen)
            node.args[0].end_lineno, node.args[0].end_col_offset = gen.end_lineno, gen.end_col_offset
        return node


def rewrite(tree: ast.Module) -> ast.Module:
    tree = Rewriter(tree).visit(tree)
    ast.fix_missing_locations(tree)
    return tree
