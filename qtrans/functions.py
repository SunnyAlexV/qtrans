"""Function translation: one Python `def` -> one C++ function.

The work happens in passes over the function body:

  1. scan       reject unsupported statements; collect every assignment ("event")
  2. loops      check for-loop variables are only used inside their loop
  3. infer      give every local variable a single C++ type (fixpoint over events)
  4. plan       decide where each variable is declared in C++
  5. warn       definite-assignment check (Python's UnboundLocalError paths)
  6. emit       write the C++ statements, carrying comments across
"""
from __future__ import annotations

import ast
import re
import inspect

from . import library as lib
from .errors import Unsupported
import collections

from .core import dims_of, int_token, iv_of, same_dim
from .core import cpp_string, float_literal
from .expressions import (P_ATOM, P_REL, P_UNARY, Expr, ExprTranslator, _Unknown, cpp_ident, wrap)
from .structs import THIS
from .classes import member_safe_name
from .types import VEC as VEC_T
from .types import RNG, STR, ListType
from .types import (BOOL, FLOAT, INT, MAT, NONE, VOID, ArrayType, MaskType, OptionalType, Scalar, StructType,
                    TupleType, assignable, is_array, join)

_STATEMENT_MESSAGES = {
    "FunctionDef": ("nested functions", "move the function to module level"),
    "AsyncFunctionDef": ("async functions", None),
    "ClassDef": ("classes inside functions", "classes and dataclasses arrive in stage 4"),
    "Try": ("try/except blocks", None),
    "TryStar": ("try/except* blocks", None),
    "With": ("with-blocks", None),
    "AsyncWith": ("async with blocks", None),
    "AsyncFor": ("async for loops", None),
    "Global": ("global statements", "pass the value in as a parameter"),
    "Nonlocal": ("nonlocal statements", None),
    "Delete": ("del statements", None),
    "Import": ("imports inside functions", "move the import to module level"),
    "ImportFrom": ("imports inside functions", "move the import to module level"),
    "Match": ("match statements", "use if/elif"),
}


def target_leaves(t):
    """The individual targets of an assignment (flattening `a, b = ...`)."""
    if isinstance(t, (ast.Tuple, ast.List)):
        return [x for e in t.elts for x in target_leaves(e)]
    return [t]


def subscript_root(node):
    """`a[i]`, `A[i, j]`, `a[1:3]` -> 'a' (the array variable being indexed); through fields
    too: `self.w[i]`, `port.weights[0]` -> 'self', 'port' (changing a field changes the object)."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def view_root(node, types, mod=None):
    """If `node` evaluates to an existing array or object, or a view of one (NumPy shares
    memory for `b = a`, slices, rows and `.T`; Python shares objects and their fields), return
    that variable's name; otherwise None (a new array or object)."""
    if isinstance(node, ast.Name):
        t = types.get(node.id)
        if is_array(t) or (mod is not None and isinstance(t, StructType) and not mod.immutable(t)):
            return node.id
        return None
    if isinstance(node, ast.Attribute) and node.attr != "T" and mod is not None:
        t = _chain_type(node, types, mod)
        if is_array(t) or (isinstance(t, StructType) and not mod.immutable(t)):
            return subscript_root(node)  # obj.w is obj's own array
    if isinstance(node, ast.Attribute) and node.attr == "T":
        return view_root(node.value, types, mod)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "transpose" \
            and not node.args:
        return view_root(node.func.value, types, mod)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ravel" \
            and not node.args:
        return view_root(node.func.value, types, mod)  # ravel of a contiguous array is a view
    if isinstance(node, ast.Call) and node.args and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "asarray")
            or (isinstance(node.func, ast.Name) and node.func.id == "asarray")):
        return view_root(node.args[0], types, mod)  # np.asarray returns its argument when it is float64 already
    if isinstance(node, ast.Subscript):
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        base_t = types.get(node.value.id) if isinstance(node.value, ast.Name) else None
        row = base_t == MAT and len(parts) == 1 and not _is_mask_index(parts[0], types)
        is_view = any(isinstance(p, ast.Slice) for p in parts) or row
        if is_view:
            return view_root(node.value, types, mod)
    return None


def _chain_type(node, types, mod):
    """The type of `x`, `x.field`, `x.inner.w` from the variable types and the class fields."""
    if isinstance(node, ast.Name):
        return types.get(node.id)
    if isinstance(node, ast.Attribute):
        t = _chain_type(node.value, types, mod)
        if isinstance(t, StructType) and t.name in mod.classes:
            f = mod.classes[t.name].field(node.attr)
            return f.type if f is not None else None
    return None


def _is_mask_index(node, types) -> bool:
    """A[mask] selects rows into a copy; A[i] is a view of row i."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return True
    if isinstance(node, ast.Name) and isinstance(types.get(node.id), MaskType):
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
        node.func.attr in ("isfinite", "isnan", "isinf")


def is_docstring(stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)




def _contains_break(stmts) -> bool:
    for s in stmts:
        if isinstance(s, ast.Break):
            return True
        if isinstance(s, ast.If) and (_contains_break(s.body) or _contains_break(s.orelse)):
            return True
        if isinstance(s, ast.Try) and (_contains_break(s.body) or any(_contains_break(h.body) for h in s.handlers)):
            return True
    return False


def always_returns(stmts) -> bool:
    """Does every path through `stmts` end in return or raise?"""
    for s in stmts:
        if isinstance(s, (ast.Return, ast.Raise)):
            return True
        if isinstance(s, ast.If) and always_returns(s.body) and always_returns(s.orelse):
            return True
        if isinstance(s, ast.Try) and always_returns(s.body) and all(always_returns(h.body) for h in s.handlers):
            return True
        if isinstance(s, ast.While) and isinstance(s.test, ast.Constant) and s.test.value is True \
                and not _contains_break(s.body):
            return True
    return False


def _try_state(s, A, block):
    """Variables certainly assigned after `try: body except ...: handler`. The handler may start
    after any statement of the body, so it can rely only on what was assigned before the try."""
    a = block(s.body, set(A))
    outs = [a] + [block(h.body, set(A)) for h in s.handlers]
    done = [o for o in outs if o is not None]
    if not done:
        return None
    out = set(done[0])
    for o in done[1:]:
        out &= o
    return out


class _Event:
    """One assignment to a local variable: name <- payload."""
    __slots__ = ("name", "kind", "payload", "index", "stmt")

    def __init__(self, name, kind, payload, stmt, index=None):
        self.name, self.kind, self.payload, self.stmt, self.index = name, kind, payload, stmt, index


def _orientation(node):
    """For a bare view of a matrix variable: (name, 'row' | 'col'), else None.
    A[i], A[i, :] -> row; A[:, j] -> col; through .T the other way round."""
    if not isinstance(node, ast.Subscript):
        return None
    base, flip = node.value, False
    if isinstance(base, ast.Attribute) and base.attr == "T":
        base, flip = base.value, True
    if not isinstance(base, ast.Name):
        return None
    parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    full = lambda p: isinstance(p, ast.Slice) and p.lower is None and p.upper is None and p.step is None  # noqa: E731
    if len(parts) == 1 and not isinstance(parts[0], ast.Slice):
        o = "row"
    elif len(parts) == 2 and not isinstance(parts[0], ast.Slice) and full(parts[1]):
        o = "row"
    elif len(parts) == 2 and full(parts[0]) and not isinstance(parts[1], ast.Slice):
        o = "col"
    else:
        return None
    return base.id, ({"row": "col", "col": "row"}[o] if flip else o)


def _crossing_view(target, value):
    """A[i] = A[:, j] (a column into a row of the same matrix, or the other way): NumPy copies
    such overlapping 1-D views element by element without a temporary, so its result depends on
    its loop order. -> the right side's orientation, or None."""
    t, v = _orientation(target), _orientation(value)
    if t and v and t[0] == v[0] and t[1] != v[1]:
        return v[1]
    return None


_PLAIN = re.compile(r"[A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*)*")


def _evaluated(code: str, t) -> str:
    """The right side as a temporary, read completely before the store. `.eval()` makes one of
    an Eigen expression, but of a plain array (a variable, a field) it returns the array itself:
    `v[::-1] = v` then reads elements it has already overwritten. Those are copied."""
    if _PLAIN.fullmatch(code):
        return f"{t.cpp()}({code})"
    return f"({code}).eval()"


# --- the source map ---------------------------------------------------------------------------
# Each emitted C++ line carries, until the header is assembled, an invisible mark naming the
# Python lines it came from; ModuleTranslator.render strips the marks into TranslationResult.line_map.
_MARK = "\x1f"
_MARK_RE = re.compile("\x1f(\\d+):(\\d+)\x1f")


def source_mark(first: int, last: int) -> str:
    return f"{_MARK}{first}:{last}{_MARK}"


def strip_marks(text: str):
    """(clean text, per line: (first, last) Python line or None)."""
    lines, spans = [], []
    for line in text.split("\n"):
        found = _MARK_RE.findall(line)
        spans.append((int(found[-1][0]), int(found[-1][1])) if found else None)
        lines.append(_MARK_RE.sub("", line))
    return "\n".join(lines), spans


def python_reads(fdef) -> set:
    """Names the function reads. `x = x` is not a read: its translation is a comment (Clang
    warns on a self-assignment), so a name read only there is unused in the C++."""
    selfs = {id(n.value) for n in ast.walk(fdef) if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Name)
             and n.value.id == n.targets[0].id}
    return {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and id(n) not in selfs}


class FunctionTranslator(ExprTranslator):
    def __init__(self, mod, fdef, sig):
        super().__init__(mod)
        self.fdef, self.sig = fdef, sig
        self.params = {p.name: p.type for p in sig.params}
        self.body = list(fdef.body)
        self.doc_end = fdef.lineno
        if self.body and is_docstring(self.body[0]):
            self.doc_end = self.body[0].end_lineno
            self.body = self.body[1:]
        self.loop_vars: set[str] = set()
        self.loop_types: dict = {}      # loop variable -> INT (range) or FLOAT (iterating a Vec)
        self.loop_lb: dict = {}         # loop variable -> proven lower bound (for index emission)
        self.loop_iv: dict = {}         # range() loop variable -> the interval it stays in
        self.events: list[_Event] = []
        self.annotated: dict = {}
        self.local_types: dict = {}
        self.locals_order: list[str] = []
        self.ignored: set[str] = set()
        self.aug_binops: dict = {}
        self.inferring = False
        self.lines: list[str] = []
        self.depth = 1
        self._tail = None
        self.comment_floor = self.doc_end
        self.hoist_at: dict = {}
        self.inline_at: dict = {}
        self._locals_cache: set = set()
        self.all_names = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name)} | set(self.params)
        self.owner = mod.classes.get(sig.owner) if sig.owner else None
        self.py_names = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name)} | {a.arg for a in fdef.args.args}
        self.has_self = sig.kind in ("instance", "property", "post_init", "init")
        self.all_names.discard("self")
        # size tokens (for NumPy broadcasting): see _dim_analysis
        self.dims_h: dict = {}
        self._now_ok = False
        self._dim_analysis()

    # --- sizes -------------------------------------------------------------------------
    def _dim_analysis(self):
        """Which names are never rebound, so that a size token built from them means the same
        size wherever it is used. A parameter that is never assigned, and a local assigned
        exactly once, qualify (in-place `x += y` on an array keeps its shape, so it doesn't
        count). A token only ever names such values, so it stays true inside loops too; a
        fresh token for an otherwise unknown size is only made outside loops, where the one
        assignment runs at most once per call."""
        stores, aug, in_loop = collections.Counter(), collections.Counter(), set()

        def visit(node, loop):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                stores[node.id] += 1
                if loop:
                    in_loop.add(node.id)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                aug[node.target.id] += 1
            elif isinstance(node, ast.ExceptHandler) and node.name:
                stores[node.name] += 2
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for n in node.names:
                    stores[n] += 2
            loop = loop or isinstance(node, (ast.For, ast.While, ast.comprehension, ast.Lambda))
            for c in ast.iter_child_nodes(node):
                visit(c, loop)
        for st in self.body:
            visit(st, False)
        self._stores, self._aug, self._in_loop = stores, aug, in_loop

    def _rebinds(self, name, t) -> int:
        return self._stores[name] - (self._aug[name] if is_array(t) else 0)

    @staticmethod
    def _clean(tok):
        """A token that may name a value of the current statement only ("now") is no proof
        across statements."""
        if isinstance(tok, tuple):
            if tok and tok[0] == "now":
                return None
            return None if any(isinstance(x, tuple) and FunctionTranslator._clean(x) is None for x in tok) else tok
        return tok

    def _event_dims(self, ev):
        """The size tokens of the value one assignment stores (None where unknown)."""
        t = self.var_type(ev.name)
        self.current = ev.stmt
        try:
            if ev.kind == "elt":
                i, _ = ev.index
                v = ev.payload
                if isinstance(v, ast.Attribute) and v.attr == "shape":
                    A = self.expr(v.value)
                    if is_array(A.type):
                        return self._clean(dims_of(A)[i])  # n, m = A.shape
                return None if t == INT else (None,) * t.rank
            e = self.expr(ev.payload)
        except (Unsupported, _Unknown, KeyError):
            return None if t == INT else (None,) * t.rank
        if t == INT:
            return self._clean(int_token(e)) if e.type in (INT, BOOL) else None
        d = dims_of(e) if e.type == t else (None,) * t.rank
        return tuple(self._clean(x) for x in d)

    def infer_dims(self):
        """Size tokens for every array and int variable: a fixpoint like type inference. The
        guess for a variable is what its assignments agree on (for a parameter, also its
        argument's own sizes); then every assignment is evaluated again under the guesses, and a
        dimension is dropped wherever one disagrees, until none does. At the fixpoint every
        assignment stores a value of the guessed sizes, given that every variable it reads has
        its guessed sizes, so by induction over the run, each variable always has them."""
        tracked = {}
        for name in list(self.params) + self.locals_order:
            t = self.params.get(name) or self.local_types.get(name)
            if is_array(t) or t == INT:
                tracked[name] = t
        evs = {}
        for ev in self.events:
            if ev.name not in tracked:
                continue
            if isinstance(ev.stmt, ast.AugAssign) and is_array(tracked[ev.name]):
                continue  # in place: the array keeps its shape
            evs.setdefault(ev.name, []).append(ev)

        def base(name):  # what the variable holds before any assignment
            t = tracked[name]
            if name not in self.params:
                return None
            return ("p", name) if t == INT else tuple(("p", name, k) for k in range(t.rank))

        def once(name):  # a local assigned once, outside loops: its value's sizes are fixed per call
            return (name not in self.params and self._rebinds(name, tracked[name]) == 1
                    and name not in self._in_loop and len(evs.get(name, [])) == 1)

        def fresh(name, d):
            t = tracked[name]
            if not once(name):
                return d
            if t == INT:
                return d if d is not None else ("v", name)
            return tuple(x if x is not None else ("v", name, k) for k, x in enumerate(d))

        def rank(name):
            t = tracked[name]
            return 0 if t == INT else t.rank

        self._once = {n for n in self.locals_order
                      if isinstance(self.local_types.get(n), MaskType) and self._rebinds(n, self.local_types[n]) == 1
                      and n not in self._in_loop and sum(ev.name == n for ev in self.events) == 1}
        saved = (list(self.notes), set(self._noted), self.inferring)
        self.inferring = True
        self.dims_h = {}
        try:
            # guess (any guess will do: the check below is what proves): what the assignments
            # agree on, ignoring the ones not known yet; repeated, as one guess feeds the next
            self.dims_h = {n: base(n) for n in tracked if base(n) is not None}
            for _ in range(len(tracked) + 2):
                grown = False
                for n in tracked:
                    vals = [self._event_dims(ev) for ev in evs.get(n, [])]
                    vals += [base(n)] if base(n) is not None else []
                    r = rank(n)
                    if r == 0:
                        known = {v for v in vals if v is not None}
                        g = known.pop() if len(known) == 1 else None
                    else:
                        g = tuple((lambda ks: ks.pop() if len(ks) == 1 else None)(
                            {v[k] for v in vals if v[k] is not None}) for k in range(r))
                    g = fresh(n, g)
                    if g != self.dims_h.get(n):
                        self.dims_h[n] = g
                        grown = True
                if not grown:
                    break
            for n in tracked:
                self.dims_h.setdefault(n, None if rank(n) == 0 else (None,) * rank(n))
            # then check, dropping what isn't confirmed
            changed = True
            for _ in range(2 * len(tracked) + 3):
                changed = False
                for n in tracked:
                    vals = [self._event_dims(ev) for ev in evs.get(n, [])]
                    g = self.dims_h[n]
                    if once(n):
                        new = fresh(n, vals[0])  # its one value: proven sizes, else its own
                    else:
                        vals += [base(n)] if base(n) is not None else []
                        if rank(n) == 0:
                            new = g if g is not None and all(v == g for v in vals) else None
                        else:
                            new = tuple(x if x is not None and all(v[k] == x for v in vals) else None
                                        for k, x in enumerate(g))
                    if new != g:
                        self.dims_h[n] = new
                        changed = True
                if not changed:
                    break
            if changed:
                self.dims_h = {}  # no fixpoint within the bound: prove nothing
        finally:
            self.notes, self._noted, self.inferring = saved
        self._now_ok = not any(isinstance(x, ast.NamedExpr) for x in ast.walk(self.fdef))

    def _var_dims(self, name, t):
        """Size tokens for a read of a variable: proven ones, else ones that hold within the
        current statement only (a local can't be rebound in the middle of an expression)."""
        h = getattr(self, "dims_h", {}).get(name)
        now_ok = getattr(self, "_now_ok", False) and self.current is not None
        if t == INT:
            return None, h if h is not None else (("now", name, id(self.current)) if now_ok else None)
        if isinstance(t, MaskType):
            # a mask's *values* decide how many elements a[mask] has; one assigned once,
            # outside loops (or read twice in one statement) has the same values at every read
            if name in getattr(self, "_once", set()):
                return None, ("mask", name)
            return None, (("now", "mask", name, id(self.current)) if now_ok else None)
        if not is_array(t):
            return None, None
        h = h or (None,) * t.rank
        return tuple(x if x is not None else (("now", name, k, id(self.current)) if now_ok else None)
                     for k, x in enumerate(h)), None

    def _unused(self, name: str) -> str:
        """`[[maybe_unused]] ` for a local Python assigns but never reads: the C++ variable would
        otherwise draw -Wunused-variable (the generated code is meant to compile warning-free,
        also under -Werror), and the attribute says what the Python says: this value is not used."""
        if not hasattr(self, "_read_names"):
            self._read_names = python_reads(self.fdef)
        return "" if name in self._read_names else "[[maybe_unused]] "

    def ident(self, name: str) -> str:
        """The C++ name of a local or parameter. Inside a method, one that would hide a field
        or method of the class (and trip -Wshadow) gets a trailing underscore."""
        if self.owner is None:
            return cpp_ident(name)
        return member_safe_name(name, self.owner, self.py_names)

    # --- variables -------------------------------------------------------------------
    @property
    def locals_set(self):
        return set(self.locals_order)

    def is_variable(self, name):
        if name in getattr(self, "comp_vars", {}):
            return True
        return (name in self.params or name in self.loop_vars or name in self.ignored
                or name in self._locals_cache or (name == "self" and self.has_self))

    def variable(self, node):
        n = node.id
        if n in getattr(self, "comp_vars", {}):  # inside a list comprehension
            t, ident = self.comp_vars[n]
            return Expr(ident, P_ATOM, t, simple=True)
        if n == "self" and self.has_self:
            return Expr(THIS, P_ATOM, self.owner.type, simple=True)
        if n in self.loop_vars:
            t = self.loop_types.get(n, INT)
            if t is None:  # over a list or a Vec: its element type
                t = self._loop_elem_type(n)
            if t == INT:  # a range() loop index lies between start and stop
                return Expr(self.ident(n), P_ATOM, t, simple=True, iv=self.loop_iv.get(n))
        elif n in self.params:
            t = self.params[n]
            if t is None:  # a parameter whose type is still being worked out (from the calls)
                raise _Unknown(n)
            dims, tok = self._var_dims(n, t)
            return Expr(self.ident(n), P_ATOM, t, simple=True, dims=dims, dimtok=tok)
        else:
            t = self.local_types.get(n)
            if t is None:
                if self.inferring:
                    raise _Unknown(n)
                raise Unsupported(node, f"could not infer a type for '{n}'")
            dims, tok = self._var_dims(n, t)
            return Expr(self.ident(n), P_ATOM, t, simple=True, dims=dims, dimtok=tok)
        return Expr(self.ident(n), P_ATOM, t, simple=True)

    # --- driver ----------------------------------------------------------------------
    def translate(self):
        self._locals_cache = set()
        self.scan(self.body)
        self.check_loop_vars()
        self.finish_scan()
        self.infer_types()
        self.infer_dims()
        if self.fdef.name in self.mod.errstate_funcs:
            self.note("errstate", "np.errstate(...) blocks are translated as their contents: they only "
                                  "change which floating-point warnings NumPy prints, never the values")
        self.check_aliasing()
        self.check_lists()
        self.check_random_draws()
        self._flag_int_or_float()
        self.plan_declarations()
        self.check_generator_declarations()
        self.definite_assignment()
        if self.sig.ret != VOID and not always_returns(self.body):
            raise Unsupported(self.fdef, f"{self.fdef.name}() can reach its end without returning "
                                         "(Python would return None)", "add a final return or raise")
        self.emit_block(self.body)
        self.flush_comments(self.fdef.end_lineno + 1)
        self._mark_unread_locals()
        return self.lines

    def check_lists(self):
        """A Python list is one object, however many names it has: `b = a` shares it, a function
        given a list changes the caller's, a loop sees appends made while it runs. The C++ copies
        instead, so wherever that sharing would show it is refused."""
        types = {**self.local_types, **self.params}
        lists = {n for n, t in types.items() if isinstance(t, ListType)}
        globals_ = {n for n, c in self.mod.constants.items() if isinstance(c.type, ListType)}
        if not lists and not globals_:
            return
        mutated = {}  # name -> the first statement that changes it in place

        def mark(name, st):
            mutated.setdefault(name, st)
        for st in (x for x in ast.walk(self.fdef) if isinstance(x, ast.stmt)):
            for e in self._own_exprs(st):  # the statement's own expressions, not its nested blocks'
                for n in ast.walk(e) if e is not None else ():
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                            and n.func.attr in ("append", "extend", "insert", "pop", "sort", "reverse", "clear", "remove"):
                        mark(n.func.value.id, st)
            if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                targets = st.targets if isinstance(st, (ast.Assign, ast.Delete)) else [st.target]
                for t in targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        mark(t.value.id, st)
                    elif isinstance(st, ast.AugAssign) and isinstance(t, ast.Name) and t.id in lists:
                        mark(t.id, st)  # xs += ys extends xs in place
        for name, st in mutated.items():
            if name in globals_ and name not in lists:
                raise Unsupported(st, f"this changes the module's list '{name}', which the C++ holds as a constant",
                                  "copy it first: xs = list(" + name + ")")
            if name in self.params and name in lists:
                raise Unsupported(st, f"this changes the list '{name}' passed in, which in Python changes the caller's "
                                      "list too; the C++ function gets its own copy",
                                  f"work on a copy: {name} = list({name}), and return it")
        for st in (x for x in ast.walk(self.fdef) if isinstance(x, (ast.Assign, ast.AnnAssign))):
            value = st.value
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in lists and isinstance(value, ast.Name) and value.id != t.id \
                        and (value.id in lists or value.id in globals_) and (t.id in mutated or value.id in mutated):
                    raise Unsupported(st, f"'{t.id}' and '{value.id}' are one list in Python (assignment shares it) and "
                                          f"'{t.id if t.id in mutated else value.id}' is changed, which would show "
                                          "through the other name; the C++ copies", f"{t.id} = list({value.id}) for a copy")
        for st in (x for x in ast.walk(self.fdef) if isinstance(x, ast.Return)):
            if isinstance(st.value, ast.Name) and st.value.id in self.params and st.value.id in lists:
                raise Unsupported(st, f"returning the list '{st.value.id}' passed in: Python hands back the caller's own "
                                      "list (the same object), the C++ a copy", f"return list({st.value.id})")
        for loop in (x for x in ast.walk(self.fdef) if isinstance(x, ast.For)):
            if isinstance(loop.iter, ast.Name) and loop.iter.id in mutated:
                inside = {id(x) for b in loop.body for x in ast.walk(b)}
                if id(mutated[loop.iter.id]) in inside:
                    raise Unsupported(loop, f"the loop changes '{loop.iter.id}' while going through it (Python's loop "
                                            "sees the change)", f"loop over a copy: for x in list({loop.iter.id})")

    def _flag_int_or_float(self):
        """Locals that hold an int in some assignments and a float in others, shown as text where
        they may still hold the int (`total = 0`, `for ...: total += x`, `print(total)`): Python
        prints 0 or 2.5 as the run goes. Each such local gets a flag beside it, `total_is_int`,
        kept as Python's type goes, and the text is written as an int or a float by it."""
        self.flagged = set()
        if not getattr(self, "widened", None):
            return

        def text_nodes(e):
            for n in ast.walk(e):
                if isinstance(n, ast.JoinedStr):
                    yield n
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("print", "str", "repr", "format"):
                    yield n
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format":
                    yield n
                elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and isinstance(n.left, ast.Constant) \
                        and isinstance(n.left.value, str):
                    yield n
        saved = self.current
        for st in (x for x in ast.walk(self.fdef) if isinstance(x, ast.stmt) and x is not self.fdef):
            for e in self._own_exprs(st):
                if e is None:
                    continue
                for t in text_nodes(e):
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name) and n.id in self.widened and n.id not in self.flagged:
                            self.current = st
                            if not self._surely_float(n.id):
                                self.flagged.add(n.id)
        self.current = saved

    def flag_name(self, name) -> str:
        return f"{self.ident(name)}_is_int"

    def _emit_flag(self, name, value_node, s, declare):
        """After `name = value`: whether Python now holds an int there."""
        cond = self.int_cond(value_node)
        if cond is None:
            cond = "false"
        elif cond is False:
            why = self.maybe_int(value_node) or "it depends on the run"
            raise Unsupported(s, f"'{name}' is shown as text, and this value is a Python int or a float depending "
                                 f"on the run, in a way qtrans can't follow ({why})",
                              "make it a float on every path (return 0.0, not 0), or format it with a float "
                              "spec (:.2f, :g)")
        self.emit(f"{'bool ' if declare else ''}{self.flag_name(name)} = {cond};")

    def _mark_unread_locals(self):
        """[[maybe_unused]] on a local the C++ never reads, though Python does: its only read was
        in code the translation dropped (the else of `if True:`). Writes (`x = ...`, `x += ...`)
        aren't reads, as for -Wunused-but-set-variable."""
        import re
        code = [re.sub(r"//.*", "", line) for line in self.lines]
        for name in self.locals_order:
            ident = self.ident(name)
            word = re.compile(rf"\b{re.escape(ident)}\b")
            decl = re.compile(rf"^(\s*)(?!\[\[maybe_unused\]\])(\S.*?) {re.escape(ident)}( = |\{{\}};| = )")
            write = re.compile(rf"^\s*{re.escape(ident)} (=|\+=|-=|\*=|/=) ")
            reads, at = 0, None
            for k, line in enumerate(code):
                n = len(word.findall(line))
                if not n:
                    continue
                if at is None and decl.match(line):
                    at, n = k, n - 1
                elif write.match(line):
                    n -= 1
                reads += n
            if at is not None and reads == 0:
                self.lines[at] = decl.sub(lambda m: f"{m.group(1)}[[maybe_unused]] {m.group(2)} {ident}{m.group(3)}",
                                          self.lines[at], count=1)

    # --- pass 1: scan ----------------------------------------------------------------
    def _add_event(self, name, kind, payload, stmt, index=None):
        self.events.append(_Event(name, kind, payload, stmt, index))
        if name not in self.params and name not in self.locals_order:
            self.locals_order.append(name)

    def scan(self, stmts):
        allowed = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.If, ast.While,
                   ast.For, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Raise, ast.Assert, ast.Try)
        for s in stmts:
            if not isinstance(s, allowed):
                what, hint = _STATEMENT_MESSAGES.get(type(s).__name__, (type(s).__name__, None))
                raise Unsupported(s, f"{what} are not supported in stage 1", hint)
            if isinstance(s, ast.For):
                if not isinstance(s.target, ast.Name):
                    raise Unsupported(s, "for-loops must use a single loop variable")
                if s.orelse:
                    raise Unsupported(s, "for ... else is not supported")
                if s.target.id in self.params or s.target.id == "self":
                    raise Unsupported(s, f"loop variable '{s.target.id}' shadows a parameter")
                v = s.target.id
                if v in self.loop_types and self.loop_types[v] != self._loop_type(s):
                    raise Unsupported(s, f"'{v}' is used as an int loop variable and as an array element")
                self.loop_vars.add(v)
                self.loop_types[v] = self._loop_type(s)
                lb = self._range_lower_bound(s)
                self.loop_lb[v] = lb if v not in self.loop_lb else (
                    None if lb is None or self.loop_lb[v] is None else min(lb, self.loop_lb[v]))
                self.scan(s.body)
            elif isinstance(s, ast.While):
                if s.orelse:
                    raise Unsupported(s, "while ... else is not supported")
                self.scan(s.body)
            elif isinstance(s, ast.If):
                self.scan(s.body)
                self.scan(s.orelse)
            elif isinstance(s, ast.Try):
                self.check_try(s)
                self.scan(s.body)
                for h in s.handlers:
                    self.scan(h.body)
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    self._scan_target(t, s.value, s)
            elif isinstance(s, ast.AnnAssign):
                if not isinstance(s.target, ast.Name):
                    raise Unsupported(s, "annotated assignment must target a plain name")
                t = self.mod.parse_annotation(s.annotation)
                name = s.target.id
                if name in self.params:
                    raise Unsupported(s, f"re-annotating parameter '{name}' is not supported")
                if name in self.annotated and self.annotated[name] != t:
                    raise Unsupported(s, f"'{name}' is annotated with two different types")
                self.annotated[name] = t
                if name not in self.locals_order:
                    self.locals_order.append(name)
                if s.value is not None:
                    self._add_event(name, "expr", s.value, s)
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, (ast.Subscript, ast.Attribute)):
                self._check_store_target(s.target, s)  # a[i] += x, obj.x += 1: no new variable
            elif isinstance(s, ast.AugAssign):
                if not isinstance(s.target, ast.Name):
                    raise Unsupported(s, "augmented assignment must target a name, a field or an array element")
                b = ast.copy_location(ast.BinOp(left=ast.Name(id=s.target.id, ctx=ast.Load()),
                                                op=s.op, right=s.value), s)
                self.aug_binops[id(s)] = b
                self._add_event(s.target.id, "expr", b, s)

    # --- try / except -------------------------------------------------------------------------
    def _handler_class(self, h):
        """The C++ exception class an `except` clause catches, or Unsupported."""
        if h.type is None:
            raise Unsupported(h, "a bare `except:` catches everything, including errors Python raises "
                                 "where C++ carries on with inf/NaN", "catch the specific error, e.g. "
                                 "except np.linalg.LinAlgError:")
        if h.name is not None and any(isinstance(n, ast.Name) and n.id == h.name for b in h.body for n in ast.walk(b)):
            raise Unsupported(h, f"using the caught exception ('{h.name}') is not supported")
        q = self.qualify(h.type) or (h.type.id if isinstance(h.type, ast.Name) else None)
        if q in ("numpy.linalg.LinAlgError", "numpy.linalg.linalg.LinAlgError", "numpy.linalg.exceptions.LinAlgError"):
            return "qt::linalg_error"
        if q == "ValueError":
            return "std::invalid_argument"
        why = {"ZeroDivisionError": "x / 0.0 raises it in Python but gives inf in C++",
               "ArithmeticError": "x / 0.0 and float overflow raise it in Python but give inf in C++",
               "OverflowError": "float overflow raises it in Python but gives inf in C++",
               "FloatingPointError": "NumPy raises it only under np.seterr; C++ never does",
               "Exception": "it includes errors Python raises where C++ carries on with inf/NaN"}
        src = ast.unparse(h.type)
        raise Unsupported(h, f"catching {src} is not supported" + (f": {why[src]}" if src in why else ""),
                          "supported: except np.linalg.LinAlgError, except ValueError")

    def _uses_math_domain(self, stmts, seen=None) -> bool:
        """Does running `stmts` (or a function it calls) use a math-module function that raises
        ValueError in Python where C++ returns NaN, e.g. math.sqrt(-1)?"""
        seen = set() if seen is None else seen
        for n in (m for s in stmts for m in ast.walk(s)):
            if not isinstance(n, ast.Call):
                continue
            if self.qualify(n.func) in lib.MATH_DOMAIN:
                return True
            if isinstance(n.func, ast.Name) and n.func.id in self.mod.ext_math_domain:
                return True  # a function of another module of the project that may raise it
            if isinstance(n.func, ast.Name) and self.mod.fdef_named(n.func.id) is not None and n.func.id not in seen:
                seen.add(n.func.id)
                if self._uses_math_domain(self.mod.fdef_named(n.func.id).body, seen):
                    return True
        return False

    def check_try(self, s):
        if s.finalbody:
            raise Unsupported(s, "try ... finally is not supported")
        if s.orelse:
            raise Unsupported(s, "try ... else is not supported", "move the else-block after the try statement")
        for h in s.handlers:
            if isinstance(h.type, ast.Tuple):
                raise Unsupported(h, "catching several exception types in one clause is not supported",
                                  "write one except clause per type")
            cls = self._handler_class(h)
            if cls == "std::invalid_argument" and self._uses_math_domain(s.body):
                raise Unsupported(h, "except ValueError around math.sqrt / math.log / ...: Python raises "
                                     "ValueError outside their domain, C++ returns NaN, so the handler would "
                                     "run in Python but not in C++", "test the argument before the call instead")

    def _scan_target(self, t, value, stmt):
        if isinstance(t, ast.Name) and t.id == "self" and self.has_self:
            raise Unsupported(stmt, "assigning to self is not supported")
        if isinstance(t, ast.Name):
            self._add_event(t.id, "expr", value, stmt)
        elif isinstance(t, (ast.Tuple, ast.List)):
            if not all(isinstance(e, ast.Name) for e in t.elts):
                raise Unsupported(stmt, "only unpacking into plain names is supported",
                                  "assign array elements on separate lines")
            if isinstance(value, (ast.Tuple, ast.List)):
                if len(value.elts) != len(t.elts):
                    raise Unsupported(stmt, "unpacking: left and right sides have different lengths")
                for e, v in zip(t.elts, value.elts):
                    self._add_event(e.id, "expr", v, stmt)
            else:
                for i, e in enumerate(t.elts):
                    self._add_event(e.id, "elt", value, stmt, (i, len(t.elts)))
        elif isinstance(t, (ast.Subscript, ast.Attribute)):
            self._check_store_target(t, stmt)
        else:
            raise Unsupported(stmt, f"cannot assign to {type(t).__name__}")

    def _check_store_target(self, t, stmt):
        base = t.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if not isinstance(base, ast.Name):
            raise Unsupported(stmt, "assignment into a nested subscript (a[1:][0] = ...) is not supported",
                              "index the array directly")
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Subscript):
            raise Unsupported(stmt, "assignment into a nested subscript (a[1:][0] = ...) is not supported",
                              "index the array directly")

    @staticmethod
    def _is_range(it):
        return isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"

    def _loop_type(self, s):
        lit = self._literal_items(s.iter)
        if lit is not None:
            return lit[0]
        if self._is_range(s.iter):
            return INT
        self.__dict__.setdefault("loop_iters", {})[s.target.id] = s.iter
        return None  # the element type of what it iterates (a Vec, a list): known after inference

    def _loop_elem_type(self, name):
        it = self.loop_iters[name]
        saved = self.current
        try:
            t = self.expr(it).type
        finally:
            self.current = saved
        if isinstance(t, ListType):
            if t.elem is None:
                raise _Unknown(name) if self.inferring else Unsupported(it, "the list's element type is not known")
            return t.elem
        return FLOAT

    @staticmethod
    def _literal_items(it):
        """`for x in [0.1, 0.2, 0.5]:` (a list or tuple of numbers or strings, all of one type):
        (type, C++ for each item), else None."""
        if not isinstance(it, (ast.List, ast.Tuple)) or not it.elts:
            return None
        vals = []
        for e in it.elts:
            neg = isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub)
            c = e.operand if neg else e
            if not isinstance(c, ast.Constant) or isinstance(c.value, (complex, bytes)) or c.value is None:
                return None
            if neg and not isinstance(c.value, (int, float)) or (neg and isinstance(c.value, bool)):
                return None
            vals.append(-c.value if neg else c.value)
        kinds = {type(v) for v in vals}
        if len(kinds) != 1:
            return None  # Python's loop variable would change type between iterations
        kind = kinds.pop()
        if kind is bool:
            return BOOL, ["true" if v else "false" for v in vals]
        if kind is int:
            if any(abs(v) >= 2 ** 63 for v in vals):
                return None
            return INT, [f"{v}LL" for v in vals]
        if kind is float:
            return FLOAT, [float_literal(v)[0] for v in vals]
        return STR, [cpp_string(v) for v in vals]

    def _range_lower_bound(self, s):
        """Smallest value a range() loop variable takes, if it is evident from constants."""
        it = s.iter
        if not self._is_range(it) or it.keywords:
            return None
        def const(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
                return n.value
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
                return -n.operand.value
            return None
        if len(it.args) == 1:
            return 0
        step = const(it.args[2]) if len(it.args) == 3 else 1
        if step is None:
            return None
        if step > 0:
            return const(it.args[0])
        stop = const(it.args[1])
        return None if stop is None else stop + 1

    def lower_bound(self, node):
        """Proven lower bound of an int expression: constants, range() loop variables, len(),
        and sums / differences of those."""
        if isinstance(node, ast.Name) and node.id in self.loop_vars and self.loop_types.get(node.id) == INT:
            return self.loop_lb.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
            return 0
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            a, b = self.lower_bound(node.left), None
            if isinstance(node.op, ast.Add):
                b = self.lower_bound(node.right)
                return None if a is None or b is None else a + b
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and a is not None:
                return a - node.right.value
            return None
        return super().lower_bound(node)

    def finish_scan(self):
        loads = {n.id for n in ast.walk(self.fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # `delta, _ = greeks(...)`: an underscore that is never read is thrown away (std::ignore)
        if "_" in self.locals_order and "_" not in loads and "_" not in self.loop_vars:
            self.ignored.add("_")
            self.locals_order.remove("_")
            self.events = [e for e in self.events if e.name != "_"]
        self._locals_cache = set(self.locals_order)

    # --- pass 2: loop variables ----------------------------------------------------------
    def check_loop_vars(self):
        """C++ `for (long long i = 0; i < n; ++i)` matches Python's range loop only if the
        body never assigns i (Python would ignore it; C++ would change the iteration) and
        i isn't read after the loop (Python keeps the last value, C++ scopes it away)."""
        def walk(node, active):
            if isinstance(node, ast.For):
                v = node.target.id
                walk(node.iter, active)
                if v in active:
                    raise Unsupported(node, f"nested loops reuse the loop variable '{v}'")
                for s in node.body:
                    walk(s, active | {v})
                return
            if isinstance(node, ast.Name) and node.id in self.loop_vars:
                if node.id not in active:
                    raise Unsupported(node, f"loop variable '{node.id}' is used outside its for-loop",
                                      "Python keeps its last value after the loop; copy it into "
                                      "another variable inside the loop if you need it")
                if isinstance(node.ctx, ast.Store):
                    raise Unsupported(node, f"assigning to loop variable '{node.id}' inside the loop",
                                      "Python ignores it on the next iteration; C++ would not")
            for child in ast.iter_child_nodes(node):
                walk(child, active)
        for s in self.body:
            walk(s, frozenset())

    # --- pass 3: type inference ------------------------------------------------------------
    def _list_mutations(self):
        """(list name, method, argument node, statement) for xs.append(x) / insert / extend."""
        if not hasattr(self, "_list_muts"):
            out = []
            for st in ast.walk(self.fdef):
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                    c = st.value
                    if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) \
                            and c.func.attr in ("append", "insert", "extend") and c.args:
                        out.append((c.func.value.id, c.func.attr, c.args[-1] if c.func.attr == "insert" else c.args[0], st))
            self._list_muts = out
        return self._list_muts

    def _list_element_types(self) -> bool:
        """`xs = []`, then xs.append(2.5): the element type comes from what is added."""
        changed = False
        for name, how, arg, st in self._list_mutations():
            t = self.local_types.get(name)
            if not isinstance(t, ListType) or name in self.annotated:
                continue
            self.current = st
            try:
                a = self.expr(arg).type
            except (_Unknown, Unsupported):
                continue
            elem = (a.elem if isinstance(a, ListType) else None) if how == "extend" else a
            if elem is None:
                continue
            if t.elem is None:
                if elem not in (FLOAT, INT, STR):
                    raise Unsupported(st, f"a list of {elem} values is not supported", "lists of float, int or str are")
                self.local_types[name] = ListType(elem)
                changed = True
            elif t.elem != elem:
                raise Unsupported(st, f"'{name}' gets a {elem} here and holds {t.elem} values: Python keeps each "
                                      "element's type (an int prints as 1, a float as 1.0), a std::vector has one",
                                  "make them one type: 1.0 rather than 1")
        return changed

    def event_type(self, ev):
        self.current = ev.stmt
        e = self.expr_raw(ev.payload) if ev.kind == "expr" else self.expr(ev.payload)
        t = e.type
        if ev.kind == "elt":
            i, n = ev.index
            elts = self.unpack_types(t)
            if elts is None or len(elts) != n:
                raise Unsupported(ev.stmt, f"cannot unpack a {t} value into {n} names")
            t = elts[i]
        if t == VOID:
            raise Unsupported(ev.stmt, "assigning the result of a function that returns None")
        return t

    def infer_types(self):
        """Every local gets one C++ type: the join of the types of everything assigned to
        it. Iterate to a fixpoint, because `y = x * 2` can't be typed until x is."""
        self.local_types = dict(self.annotated)
        self.inferring = True
        try:
            for _ in range(len(self.events) + 3):
                changed = False
                for ev in self.events:
                    if ev.name in self.params or ev.name in self.annotated:
                        continue
                    try:
                        t = self.event_type(ev)
                    except (_Unknown, Unsupported):
                        continue
                    old = self.local_types.get(ev.name)
                    new = join(old, t)
                    if new == NONE:
                        continue  # only None so far: its type comes from the other assignments
                    if new is None:
                        raise Unsupported(ev.stmt, f"'{ev.name}' is assigned both {old} and {t} values; "
                                                   "a C++ variable has one type", "use separate names")
                    if new != old:
                        self.local_types[ev.name] = new
                        changed = True
                changed = self._list_element_types() or changed
                if not changed:
                    break
        finally:
            self.inferring = False
        for name in self.locals_order:
            if name not in self.local_types:
                ev = next((e for e in self.events if e.name == name), None)
                if ev is None:
                    raise Unsupported(self.fdef, f"'{name}' is annotated but never assigned")
                try:
                    if all(self.event_type(e) == NONE for e in self.events if e.name == name):
                        self.local_types[name] = NONE  # only ever None: any use of it is refused later
                        continue
                except (Unsupported, _Unknown):
                    pass
                try:
                    self.event_type(ev)  # re-run outside inference mode to surface the real error
                except _Unknown as u:
                    raise Unsupported(ev.stmt, f"could not infer a type for '{name}': it depends on "
                                               f"'{u.args[0]}', whose type is never determined",
                                      f"annotate it, e.g. {name}: float = ...")
        # notes from inference are discarded: types were provisional then
        self.notes, self._noted = [], set()
        self.widened = set()  # locals that hold an int in some assignments, a float in others
        for name in self.locals_order:
            seen = set()
            for ev in self.events:
                if ev.name == name:
                    try:
                        seen.add(self.event_type(ev))
                    except Unsupported:
                        pass
            if self.local_types[name] == FLOAT and (INT in seen or BOOL in seen):
                self.widened.add(name)
                self.note(f"widen:{name}", f"'{name}' holds an int in some assignments and a float in "
                                           "others; declared double (same values)")
        # and those assigned a value that may itself be a Python int (a function returning 0 on
        # one path, another such local): which print as 0, not 0.0 (see printing.maybe_int)
        maybe = getattr(self.mod, "maybe_int", set())
        changed = True
        while changed:
            changed = False
            for ev in self.events:
                n = ev.name
                if n in self.widened or self.local_types.get(n) != FLOAT or n in self.params:
                    continue
                if ev.kind == "elt":
                    p = ev.payload
                    hit = isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and (p.func.id, ev.index[0]) in maybe
                elif isinstance(ev.stmt, ast.AugAssign):
                    hit = False  # x += v: x's own type decides (above)
                else:
                    self.current = ev.stmt
                    hit = bool(self.maybe_int(ev.payload)) if isinstance(ev.payload, ast.AST) else False
                if hit:
                    self.widened.add(n)
                    changed = True

    # --- pass 3b: aliasing ---------------------------------------------------------------------
    def _alias_sources(self, value, types):
        """Arrays `value` may share memory with: itself, a view of it (slice, row, .T), either
        branch of a conditional, or a parameter a called function hands back."""
        if isinstance(value, ast.IfExp):
            return self._alias_sources(value.body, types) | self._alias_sources(value.orelse, types)
        root = view_root(value, types, self.mod)
        if root:
            return {root}
        if isinstance(value, ast.Call) and (
                (isinstance(value.func, ast.Name) and value.func.id in self.mod.classes)
                or self.qualify(value.func) == "dataclasses.replace"):
            # Option(w=w): the object's field *is* the array passed in; replace() copies the
            # other fields by reference, so arrays inside stay shared
            return {r for a in list(value.args) + [k.value for k in value.keywords]
                    for r in [view_root(a, types, self.mod)] if r}
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                and value.func.id in self.mod.returns_alias:
            out = set()
            for i in self.mod.returns_alias[value.func.id]:
                if i < len(value.args):
                    r = view_root(value.args[i], types, self.mod)
                    if r:
                        out.add(r)
            return out
        return set()

    def check_aliasing(self):
        """In NumPy `b = a`, slices, rows and `.T` share memory with the original; in C++ the
        variable is a copy. The two agree unless one of the names is later modified in place,
        so any such pair is refused rather than translated into a silent difference."""
        types = {**self.local_types, **self.params}
        if self.has_self:
            types["self"] = self.owner.type
        arrays = {n for n, t in types.items()
                  if is_array(t) or (isinstance(t, StructType) and not self.mod.immutable(t))}
        if not arrays:
            return
        mutated = set()
        for n in ast.walk(self.fdef):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for leaf in target_leaves(t):
                        if isinstance(leaf, (ast.Subscript, ast.Attribute)):
                            mutated.add(subscript_root(leaf))
                        elif isinstance(n, ast.AugAssign) and isinstance(leaf, ast.Name) and leaf.id in arrays:
                            mutated.add(leaf.id)
            elif isinstance(n, ast.Call):
                callee = None
                if isinstance(n.func, ast.Name) and n.func.id in self.mod.sigs:
                    callee = self.mod.sigs[n.func.id]
                elif isinstance(n.func, ast.Attribute) and not self._is_module_path(n.func.value):
                    bt = _chain_type(n.func.value, types, self.mod)
                    if isinstance(bt, StructType) and bt.name in self.mod.classes:
                        m = self.mod.classes[bt.name].methods.get(n.func.attr)
                        if m is not None and m.key in self.mod.sigs:
                            callee = self.mod.sigs[m.key]
                            if m.kind == "instance" and not callee.const:
                                mutated.add(subscript_root(n.func.value))  # obj.step() changes obj
                if callee is not None:
                    pos = {p.name: i for i, p in enumerate(callee.params)}
                    pairs = list(enumerate(n.args)) + [(pos.get(k.arg, -1), k.value) for k in n.keywords]
                    for i, a in pairs:
                        if 0 <= i < len(callee.params) and callee.params[i].mode == "ref" and \
                                isinstance(a, (ast.Name, ast.Attribute, ast.Subscript)):
                            mutated.add(subscript_root(a))
        for n in ast.walk(self.fdef):
            if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            pairs = []
            for t in targets:
                if isinstance(t, ast.Name):
                    pairs.append((t.id, n.value))
                elif isinstance(t, (ast.Tuple, ast.List)) and isinstance(n.value, (ast.Tuple, ast.List)):
                    pairs += [(e.id, v) for e, v in zip(t.elts, n.value.elts) if isinstance(e, ast.Name)]
            for x, value in pairs:
                if x not in arrays:
                    continue
                for y in self._alias_sources(value, types):
                    if y != x and (x in mutated or y in mutated):
                        which = x if x in mutated else y
                        if is_array(types.get(x)) and is_array(types.get(y)):
                            raise Unsupported(n, f"'{x}' and '{y}' share memory in NumPy (assignment, slices "
                                                 f"and .T are views, not copies) and '{which}' is modified in "
                                                 "place, so a C++ copy would give different results",
                                              f"if you want an independent array, write {x} = "
                                              + (f"{ast.unparse(value)}.copy()" if isinstance(value, (ast.Name, ast.Subscript, ast.Attribute))
                                                 else f"({ast.unparse(value)}).copy()"))
                        if isinstance(types.get(y), StructType) and not isinstance(types.get(x), StructType):
                            rel = f"'{x}' is part of '{y}' in Python (an object's fields are shared, not copied)"
                        elif isinstance(types.get(x), StructType) and not isinstance(types.get(y), StructType):
                            rel = f"'{y}' is part of '{x}' in Python (the object holds that very array)"
                        else:
                            rel = f"'{x}' and '{y}' are the same object in Python (objects are shared, not copied)"
                        raise Unsupported(n, f"{rel} and '{which}' is modified, so a C++ copy would give "
                                             "different results",
                                          "pass or store a copy: x.copy() for an array, copy.deepcopy(obj) for an "
                                          "object (dataclasses.replace keeps sharing the arrays inside)")
        for n in ast.walk(self.fdef):
            if isinstance(n, ast.For) and not self._is_range(n.iter):
                src = view_root(n.iter, types)
                for b in n.body:
                    for m in ast.walk(b):
                        if isinstance(m, (ast.Assign, ast.AugAssign)):
                            targets = m.targets if isinstance(m, ast.Assign) else [m.target]
                            for t in targets:
                                for leaf in target_leaves(t):
                                    if src and (subscript_root(leaf) == src or (
                                            isinstance(m, ast.AugAssign) and getattr(leaf, "id", None) == src)):
                                        raise Unsupported(m, f"the loop modifies '{src}' while iterating over it",
                                                          "loop over range(len(...)) instead")

    def _needs_eval(self, value_node, target: str, whole_array: bool) -> bool:
        """Must the right-hand side be evaluated into a temporary before it is written?
        NumPy always computes the right side first; Eigen evaluates lazily, element by element,
        so it would read values it has already overwritten. Writing into a slice or element
        needs a temporary whenever the right side reads the same array at all; writing the
        whole array only when the right side reads it through a slice, index or transpose
        (elementwise expressions of the whole array are safe)."""
        if whole_array:
            return self._reads_view_of(value_node, target)
        return any(isinstance(n, ast.Name) and n.id == target for n in ast.walk(value_node))

    # --- pass 3c: random draws ----------------------------------------------------------------
    def _effect(self, n, types):
        """What a call changes: (variables it modifies, whether it draws random numbers).
        A draw from a generator (also one held in an object: self.rng.normal()), a method that
        isn't const, and a function given an argument it modifies all count."""
        roots, draw = set(), False
        f = n.func
        callee = None
        if isinstance(f, ast.Attribute) and not self._is_module_path(f.value):
            bt = _chain_type(f.value, types, self.mod)
            if bt == RNG:
                roots.add(subscript_root(f.value))
                draw = True
            elif isinstance(bt, ListType) and f.attr in ("append", "extend", "insert", "pop", "sort", "reverse",
                                                          "clear", "remove"):
                roots.add(subscript_root(f.value))  # xs.pop() changes xs
            elif isinstance(bt, StructType) and bt.name in self.mod.classes:
                m = self.mod.classes[bt.name].methods.get(f.attr)
                if m is not None and m.key in self.mod.sigs:
                    callee = self.mod.sigs[m.key]
                    if m.kind == "instance" and not callee.const:
                        roots.add(subscript_root(f.value))
        elif isinstance(f, ast.Name) and f.id in self.mod.sigs:
            callee = self.mod.sigs[f.id]
            if f.id in self.mod.printing:
                roots.add("<output>")  # it prints: where its output goes depends on when it runs
        if callee is not None:
            pos = {p.name: i for i, p in enumerate(callee.params)}
            pairs = list(enumerate(n.args)) + [(pos.get(k.arg, -1), k.value) for k in n.keywords]
            for i, a in pairs:
                if 0 <= i < len(callee.params) and callee.params[i].mode == "ref" and \
                        isinstance(a, (ast.Name, ast.Attribute, ast.Subscript)):
                    roots.add(subscript_root(a))
                    draw = draw or callee.params[i].type == RNG
        roots.discard(None)
        return roots, draw

    def check_random_draws(self):
        """Python evaluates left to right; C++ leaves the order of operands and arguments
        unspecified (GCC on x86 often goes right to left). So a statement may contain at most
        one call that changes something (a random draw, a method that changes its object, a
        function that changes an argument), and must not otherwise read what that call changes:
        rng.random() - rng.random(), p.step() + p.cash and push(a) + a[0] are all refused."""
        types = {**self.local_types, **self.params}
        if self.has_self:
            types["self"] = self.owner.type

        def visit(stmts):
            for s in stmts:
                self._check_order(s, types)
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None:
                    targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                    if any(isinstance(t, ast.Name) and types.get(t.id) == RNG for t in targets) and not (
                            isinstance(s.value, ast.Call) and self.qualify(s.value.func) == "numpy.random.default_rng"):
                        raise Unsupported(s, "a generator can only be created with np.random.default_rng(seed); "
                                             "`g2 = g` would share one stream in Python but copy it in C++",
                                          "pass the generator itself to the functions that need it")
                for block in (getattr(s, "body", []), getattr(s, "orelse", []),
                              *[h.body for h in getattr(s, "handlers", [])]):
                    visit(block)
        visit(self.body)

    def _check_order(self, s, types):
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            reads = [s.value] if s.value is not None else []  # the right side runs first in both languages
            exprs = reads + (list(s.targets) if isinstance(s, ast.Assign) else [s.target])
        elif isinstance(s, ast.AugAssign):
            # Python reads the target before the right side; C++17 evaluates the right side first
            reads = exprs = [s.target, s.value]
        else:
            reads = exprs = [e for e in self._own_exprs(s) if e is not None]
        effects = []
        for e in exprs:
            for n in ast.walk(e):
                if isinstance(n, ast.Call):
                    roots, draw = self._effect(n, types)
                    if roots:
                        effects.append((n, roots, draw))
        def nested(a, b):  # one call inside the other's arguments: the inner one runs first, in both
            return any(x is b for x in ast.walk(a)) or any(x is a for x in ast.walk(b))
        if len(effects) > 1 and not all(nested(effects[i][0], effects[j][0])
                                        for i in range(len(effects)) for j in range(i + 1, len(effects))):
            if all(d for _, _, d in effects):
                raise Unsupported(s, "two random draws in one statement: C++ doesn't fix which is "
                                     "evaluated first, so the draws could be used in a different order",
                                  "draw into separate variables, one per line")
            raise Unsupported(s, "two calls that change something, in one statement: Python runs them left to "
                                 "right, C++ in an unspecified order", "make each call on its own line")
        for call, roots, _ in effects:
            inside = {id(x) for x in ast.walk(call)}
            for e in reads:
                for x in ast.walk(e):
                    if isinstance(x, ast.Name) and x.id in roots and id(x) not in inside:
                        raise Unsupported(s, f"this statement calls {ast.unparse(call.func)}(...), which changes "
                                             f"'{x.id}', and also reads '{x.id}': Python does them left to right, "
                                             "C++ in an unspecified order",
                                          "make the call on its own line, into a variable")

    def check_generator_declarations(self):
        for s, names in ((s, n) for s in ast.walk(self.fdef) for n in [self.hoist_at.get(id(s), ())]):
            for n in names:
                if self.local_types.get(n) == RNG:
                    raise Unsupported(s, f"generator '{n}' must be created once, by a plain assignment at the "
                                         "top level of the function", "move rng = np.random.default_rng(seed) up")

    @staticmethod
    def _reads_view_of(node, name) -> bool:
        """Does `node` read `name` through a slice, index or transpose? Then assigning the result
        back into `name` must evaluate it first: Eigen is lazy and would read while writing."""
        for n in ast.walk(node):
            if isinstance(n, ast.Subscript) and subscript_root(n) == name:
                return True
            if isinstance(n, ast.Attribute) and n.attr == "T" and isinstance(n.value, ast.Name) and n.value.id == name:
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "transpose" \
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == name:
                return True
        return False

    # --- pass 4: declaration planning --------------------------------------------------------
    def _loads_stores(self, stmt):
        loads, stores = [], []
        skip = {id(n.target) for n in ast.walk(stmt) if isinstance(n, ast.AnnAssign) and n.value is None}
        for n in ast.walk(stmt):
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                loads.append(n.target.id)
            if isinstance(n, ast.Name) and id(n) not in skip:
                (stores if isinstance(n.ctx, ast.Store) else loads).append(n.id)
        L = self.locals_set
        return [x for x in loads if x in L], [x for x in stores if x in L]

    def _tuple_needs_tie(self, s) -> bool:
        """`a, b = b, a`: the right side reads a target, so assign through a temporary tuple."""
        t = s.targets[0]
        names = {e.id for e in t.elts}
        loads = {n.id for n in ast.walk(s.value) if isinstance(n, ast.Name)}
        return bool(names & loads)

    @staticmethod
    def _own_exprs(s):
        """The expressions a statement evaluates itself, excluding its nested blocks."""
        if isinstance(s, (ast.If, ast.While)):
            return [s.test]
        if isinstance(s, ast.For):
            return [s.iter]
        if isinstance(s, ast.Assign):
            return list(s.targets) + [s.value]
        if isinstance(s, ast.AnnAssign):
            return [s.target, s.value] if s.value is not None else []
        if isinstance(s, ast.AugAssign):
            return [s.target, s.value]
        if isinstance(s, (ast.Return, ast.Expr)):
            return [s.value] if s.value is not None else []
        if isinstance(s, ast.Raise):
            return [s.exc] if s.exc is not None else []
        if isinstance(s, ast.Assert):
            return [s.test] + ([s.msg] if s.msg is not None else [])
        return []

    def _block_tree(self):
        """Map every local to the innermost block (statement list) containing all of its uses.
        An elif chain counts as one level, because C++ can't declare between `} else if`."""
        blocks, parent, stacks = {}, {}, {}

        def visit(stmts, stack):
            blocks[id(stmts)] = stmts
            parent[id(stmts)] = stack[-1] if stack else None
            stack = stack + (id(stmts),)
            for s in stmts:
                self._visit_stmt(s, stack, visit, stacks)

        visit(self.body, ())
        owner = {}
        for v, st in stacks.items():
            prefix = st[0]
            for other in st[1:]:
                n = 0
                while n < len(prefix) and n < len(other) and prefix[n] == other[n]:
                    n += 1
                prefix = prefix[:n]
            owner[v] = prefix[-1]
        return blocks, parent, owner

    def _visit_stmt(self, s, stack, visit, stacks):
        for e in self._own_exprs(s):
            for n in ast.walk(e):
                if isinstance(n, ast.Name) and n.id in self.locals_set:
                    stacks.setdefault(n.id, []).append(stack)
        if isinstance(s, ast.If):
            visit(s.body, stack)
            orelse = s.orelse
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                self._visit_stmt(orelse[0], stack, visit, stacks)  # elif: same level
            elif orelse:
                visit(orelse, stack)
        elif isinstance(s, (ast.While, ast.For)):
            visit(s.body, stack)
        elif isinstance(s, ast.Try):
            visit(s.body, stack)
            for h in s.handlers:
                visit(h.body, stack)

    def _assigned_before_every_read(self, stmts, var) -> bool:
        """True if, running `stmts` from the top, `var` is always assigned before it is read.
        Only then may it be declared inside this block: a loop body runs afresh each
        iteration, and a read before the assignment would see the previous iteration's value
        in Python but a fresh variable in C++."""
        ok = True

        def reads(node, A):
            nonlocal ok
            if var not in A and any(isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Load)
                                    for n in ast.walk(node)):
                ok = False

        def block(body, A):
            for s in body:
                A = stmt(s, A)
                if A is None:
                    return None
            return A

        def stmt(s, A):
            if isinstance(s, ast.If):
                reads(s.test, A)
                a, b = block(s.body, set(A)), block(s.orelse, set(A))
                return b if a is None else a if b is None else a & b
            if isinstance(s, ast.Try):
                return _try_state(s, A, block)
            if isinstance(s, (ast.While, ast.For)):
                reads(s.test if isinstance(s, ast.While) else s.iter, A)
                block(s.body, set(A))
                return A
            if isinstance(s, (ast.Return, ast.Raise)):
                reads(s, A)
                return None
            if isinstance(s, (ast.Break, ast.Continue)):
                return None
            if isinstance(s, ast.AugAssign):
                reads(s.value, A)
                if isinstance(s.target, (ast.Subscript, ast.Attribute)):
                    reads(s.target, A)
                    return A
                reads(ast.Name(id=s.target.id, ctx=ast.Load()), A)
                return A | {s.target.id}
            if isinstance(s, (ast.Assign, ast.AnnAssign)):
                if s.value is None:
                    return A
                reads(s.value, A)
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                leaves = [x for t in targets for x in target_leaves(t)]
                for x in leaves:
                    if isinstance(x, ast.Subscript):
                        reads(x, A)
                return A | {x.id for x in leaves if isinstance(x, ast.Name)}
            reads(s, A)
            return A

        block(stmts, set())
        return ok

    def plan_declarations(self):
        """C++ needs a declaration before use; Python doesn't. Each local is declared in the
        innermost block that contains all its uses, provided it's assigned before it is read
        there; otherwise it moves out a level (eventually to the top of the function).
        Within its block it's declared at its first plain assignment (`double d1 = ...;`),
        or just before the first statement that touches it."""
        blocks, parent, owner = self._block_tree()
        for v in owner:
            while parent[owner[v]] is not None and not self._assigned_before_every_read(blocks[owner[v]], v):
                owner[v] = parent[owner[v]]
        order = {n: i for i, n in enumerate(self.locals_order)}
        for bid, stmts in blocks.items():
            mine = {v for v, b in owner.items() if b == bid}
            if mine:
                self._plan_block(stmts, mine, order)

    def _plan_block(self, stmts, mine, order):
        declared = set()
        for s in stmts:
            loads, stores = self._loads_stores(s)
            loads = [x for x in loads if x in mine]
            stores = [x for x in stores if x in mine]
            hoist = []
            for n in sorted(set(loads) - declared, key=order.get):
                hoist.append(n)
                declared.add(n)
            inline = []
            if isinstance(s, ast.Assign) and len(s.targets) == 1:
                t = s.targets[0]
                if isinstance(t, ast.Name):
                    if t.id in mine and t.id not in declared:
                        inline = [t.id]
                elif isinstance(t, (ast.Tuple, ast.List)):
                    names = [e.id for e in t.elts]
                    if isinstance(s.value, (ast.Tuple, ast.List)):
                        if not self._tuple_needs_tie(s):
                            inline = [n for n in names if n in mine and n not in declared]
                    else:
                        elts = self.unpack_types(self.expr_type_quiet(s.value))
                        if (elts is not None and len(set(names)) == len(names)
                                and all(n in mine and n not in declared for n in names)
                                and all(self.local_types[n] == et for n, et in zip(names, elts))):
                            inline = names
            elif isinstance(s, ast.AnnAssign) and s.value is not None \
                    and s.target.id in mine and s.target.id not in declared:
                inline = [s.target.id]
            declared.update(inline)
            for n in sorted(set(stores) - declared, key=order.get):
                hoist.append(n)
                declared.add(n)
            self.hoist_at[id(s)] = hoist
            self.inline_at[id(s)] = set(inline)

    def unpack_types(self, t):
        """The element types `a, b = value` unpacks: a tuple's, or a NamedTuple's fields."""
        if isinstance(t, TupleType):
            return list(t.elts)
        if isinstance(t, StructType) and self.mod.classes[t.name].namedtuple:
            return [f.type for f in self.mod.classes[t.name].fields]
        return None

    def expr_type_quiet(self, node):
        try:
            return self.expr(node).type
        except Unsupported:
            return None

    # --- pass 5: definite assignment ------------------------------------------------------------
    def definite_assignment(self):
        def check(node, A):
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in self.locals_set \
                        and n.id not in A:
                    self.note(f"unbound:{n.id}", f"'{n.id}' may be read before it is assigned on some "
                                                 "path (Python raises UnboundLocalError; C++ reads the "
                                                 "initial value 0)")

        def block(stmts, A):
            for s in stmts:
                A = stmt(s, A)
                if A is None:
                    return None
            return A

        def stmt(s, A):
            if isinstance(s, ast.If):
                check(s.test, A)
                a, b = block(s.body, set(A)), block(s.orelse, set(A))
                if a is None:
                    return b
                if b is None:
                    return a
                return a & b
            if isinstance(s, ast.Try):
                return _try_state(s, A, block)
            if isinstance(s, ast.While):
                check(s.test, A)
                block(s.body, set(A))
                return A
            if isinstance(s, ast.For):
                check(s.iter, A)
                block(s.body, set(A))
                return A
            if isinstance(s, (ast.Return, ast.Raise)):
                check(s, A)
                return None
            if isinstance(s, (ast.Break, ast.Continue)):
                return None
            if isinstance(s, ast.AugAssign):
                check(s.value, A)
                if isinstance(s.target, (ast.Subscript, ast.Attribute)):
                    check(s.target, A)
                    return A
                check(ast.Name(id=s.target.id, ctx=ast.Load()), A)
                return A | {s.target.id}
            if isinstance(s, (ast.Assign, ast.AnnAssign)):
                if s.value is not None:
                    check(s.value, A)
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                leaves = [x for t in targets for x in target_leaves(t)]
                for x in leaves:
                    if isinstance(x, ast.Subscript):
                        check(x, A)
                names = {x.id for x in leaves if isinstance(x, ast.Name)}
                return A | names if s.value is not None else A
            check(s, A)
            return A

        block(self.body, set())

    # --- pass 6: emission ----------------------------------------------------------------------
    def emit(self, line):
        if self._tail:
            line += self._tail
            self._tail = None
        src = getattr(self, "cur_src", None)
        if line.strip() == "}" and getattr(self, "_last_end", None):
            src = (self._last_end, self._last_end)  # a block's closing brace: with its last line
        elif src:
            self._last_end = src[1]
        self.lines.append("    " * self.depth + line + (source_mark(*src) if src else ""))

    def flush_comments(self, upto_line):
        full = self.mod.full_comments
        for ln in range(self.comment_floor + 1, upto_line):
            if ln in full:
                self.lines.append("    " * self.depth + "// " + full[ln] + source_mark(ln, ln))
        self.comment_floor = max(self.comment_floor, upto_line - 1)

    def _header_src(self, s):
        """The Python lines of a statement's own text: a compound statement's header only."""
        if isinstance(s, (ast.If, ast.While)):
            return (s.lineno, s.test.end_lineno)
        if isinstance(s, ast.For):
            return (s.lineno, s.iter.end_lineno)
        if isinstance(s, ast.Try):
            return (s.lineno, s.lineno)
        return (s.lineno, s.end_lineno)

    def _keyword_line(self, after, before, word):
        """The line of `else:` (or another keyword) between two statements, for the source map."""
        lines = self.mod.source.splitlines()
        for ln in range(after + 1, before):
            if lines[ln - 1].strip().startswith(word):
                return ln
        return after + 1

    def emit_block(self, stmts):
        for s in stmts:
            self.emit_stmt(s)

    def emit_stmt(self, s):
        self.current = s
        self.flush_comments(s.lineno)
        prev = getattr(self, "cur_src", None)
        self.cur_src = self._header_src(s)  # a declaration hoisted here belongs to this statement
        for n in self.hoist_at.get(id(s), ()):
            t = self.local_types[n]
            self.emit(f"{self._unused(n)}{t.cpp()} {self.ident(n)} = {t.zero()};" if isinstance(t, Scalar)
                      else f"{self._unused(n)}{t.cpp()} {self.ident(n)}{{}};")
            if n in getattr(self, "flagged", ()):
                self.emit(f"bool {self.flag_name(n)} = false;  // does Python hold an int in {n} here?")
        compound = isinstance(s, (ast.If, ast.While, ast.For))
        last = s.lineno if compound else s.end_lineno
        texts = [self.mod.inline_comments[ln] for ln in range(s.lineno, last + 1) if ln in self.mod.inline_comments]
        self._tail = ("  // " + "; ".join(texts)) if texts else None
        self.comment_floor = last
        getattr(self, "s_" + type(s).__name__)(s)
        self.cur_src = prev
        self._tail = None

    def _block(self, stmts):
        self.depth += 1
        self.emit_block(stmts)
        self.depth -= 1

    def check_assignable(self, target, e, node, what):
        if not assignable(target, e.type):
            hint = None
            if target == INT and e.type == FLOAT:
                hint = "Python would store a float here; annotate it as float or convert with int(...)"
            raise Unsupported(node, f"{what} is declared {target} but gets a {e.type} value", hint)

    # statements ---------------------------------------------------------------------------
    def s_Pass(self, s):
        pass

    def s_Break(self, s):
        self.emit("break;")

    def s_Continue(self, s):
        self.emit("continue;")

    def s_Expr(self, s):
        v = s.value
        if isinstance(v, ast.Constant):
            if isinstance(v.value, str):
                for line in inspect.cleandoc(v.value).splitlines():
                    self.emit("// " + line)
            return  # `...` or a bare literal does nothing
        if not isinstance(v, ast.Call):
            self.note(f"noeffect:{s.lineno}", f"line {s.lineno}: expression with no effect skipped")
            return
        if self.qualify(v.func) == "sys.exit":
            return self.exit_call(s, v)
        if isinstance(v.func, ast.Name) and v.func.id == "print" and not self.is_variable("print") \
                and "print" not in self.mod.sigs:
            line = self.print_statement(v)
            self.emit(line if line else "// print(end=\"\") writes nothing")
            return
        if ast.unparse(v.func) == "object.__setattr__":
            # the frozen-dataclass idiom for a derived field: object.__setattr__(self, "x", value)
            if not (self.sig.kind == "post_init" and len(v.args) == 3 and isinstance(v.args[0], ast.Name)
                    and v.args[0].id == "self" and isinstance(v.args[1], ast.Constant)
                    and isinstance(v.args[1].value, str)):
                raise Unsupported(s, "object.__setattr__ is supported only as object.__setattr__(self, \"field\", "
                                     "value) inside __post_init__")
            target = ast.copy_location(ast.Attribute(value=v.args[0], attr=v.args[1].value, ctx=ast.Store()), v)
            self._setattr_ok = True
            try:
                self.store_attribute(target, v.args[2], s)
            finally:
                self._setattr_ok = False
            return
        e = self.expr(v)
        self.emit(f"{e.code};")

    def exit_call(self, s, v):
        """sys.exit(): from the program's own code `return status;` out of main(), elsewhere
        std::exit(status). sys.exit(None) and sys.exit() end with status 0."""
        if v.keywords or len(v.args) > 1:
            raise Unsupported(s, "sys.exit() takes one argument")
        arg = v.args[0] if v.args else None
        code = "0"
        if arg is not None and not (isinstance(arg, ast.Constant) and arg.value is None):
            e = self.expr_raw(arg)
            if e.type == VOID:  # sys.exit(main()): main returns None
                self.emit(f"{e.code};")
            elif e.type in (INT, BOOL):
                code = str(int(e.value)) if isinstance(e.value, (int, bool)) else f"static_cast<int>({e.code})"
            else:
                raise Unsupported(arg, f"sys.exit() of a {e.type}: only an int status (or None) is supported",
                                  "print the message, then sys.exit(1)")
        if self.sig.cpp == "main" and self.sig.name == "(program)":
            self.emit(f"return {code};")
        else:
            self.emit(f"std::exit({code});")

    def var_type(self, name):
        return self.params.get(name) or self.local_types[name]

    def assign_name(self, name, value_node, s, inline_set):
        vt = self.var_type(name)
        if (is_array(vt) or (isinstance(vt, OptionalType) and is_array(vt.inner))) and \
                self.mod.is_asarray_of(value_node, name) and name not in inline_set:
            self.emit(f"// {ast.unparse(s)}: already a float64 array, so NumPy returns it unchanged")
            return
        e = self.expr_raw(value_node) if isinstance(vt, OptionalType) else self.expr(value_node)
        if e.type == VOID:
            raise Unsupported(s, "assigning the result of a function that returns None")
        what = f"parameter '{name}'" if name in self.params else f"'{name}'"
        self.check_assignable(vt, e, s, what)
        if isinstance(vt, OptionalType):
            e = self.to_optional(e, vt)
        code = self.as_double(e).code if vt == FLOAT else e.code
        if vt == RNG and name in inline_set and code.startswith("qt::Rng(") and code.endswith(")"):
            self.emit(f"qt::Rng {self.ident(name)}{code[len('qt::Rng'):]};")
            return
        if (is_array(vt) or (isinstance(vt, OptionalType) and is_array(vt.inner))) and \
                self._needs_eval(value_node, name, whole_array=True):
            code = f"({code}).eval()"  # Eigen is lazy: read all of the old value before overwriting it
        if isinstance(vt, ListType) and code.startswith(vt.cpp() + "{"):
            code = code[len(vt.cpp()):]  # std::vector<double> xs = {1.0, 2.0};
        if name in inline_set:
            self.emit(f"{self._unused(name)}{vt.cpp()} {self.ident(name)} = {code};")
        elif code == self.ident(name):  # w = w.copy() of a parameter passed by value
            self.emit(f"// {ast.unparse(s)}: {name} is already this function's own copy")
        else:
            self.emit(f"{self.ident(name)} = {code};")
        if name in getattr(self, "flagged", ()):
            self._emit_flag(name, value_node, s, declare=name in inline_set)

    def s_Assign(self, s):
        inline = self.inline_at.get(id(s), set())
        if len(s.targets) > 1:
            if not isinstance(s.value, ast.Constant):
                raise Unsupported(s, "chained assignment `a = b = expr` is only supported for constants")
            for t in s.targets:
                if not isinstance(t, ast.Name):
                    raise Unsupported(s, "chained assignment must target plain names")
                self.assign_name(t.id, s.value, s, inline)
            return
        t = s.targets[0]
        if isinstance(t, ast.Name):
            self.assign_name(t.id, s.value, s, inline)
            return
        if isinstance(t, ast.Subscript):
            self.store_subscript(t, s.value, s, "=")
            return
        if isinstance(t, ast.Attribute):
            self.store_attribute(t, s.value, s)
            return
        names = [e.id for e in t.elts]
        if isinstance(s.value, (ast.Tuple, ast.List)):
            if not self._tuple_needs_tie(s):
                for n, v in zip(names, s.value.elts):
                    if n in self.ignored:
                        self.emit(f"static_cast<void>({self.expr(v).code});")
                    else:
                        self.assign_name(n, v, s, inline)
                return
            # a, b = b, a: build the new values first, then assign (Python semantics)
            parts = []
            for n, v in zip(names, s.value.elts):
                e = self.expr(v)
                if n not in self.ignored:
                    self.check_assignable(self.var_type(n), e, s, f"'{n}'")
                parts.append(e.code)
            lhs = ", ".join("std::ignore" if n in self.ignored else self.ident(n) for n in names)
            self.emit(f"std::tie({lhs}) = std::make_tuple({', '.join(parts)});")
            return
        e = self.expr(s.value)
        elts = self.unpack_types(e.type)
        if elts is None or len(elts) != len(names):
            raise Unsupported(s, f"cannot unpack a {e.type} value into {len(names)} names")

        for n, et in zip(names, elts):
            if n not in self.ignored and not assignable(self.var_type(n), et):
                raise Unsupported(s, f"'{n}' is declared {self.var_type(n)} but gets a {et} value")
        if inline and set(names) <= inline:
            self.emit(f"auto [{', '.join(self.ident(n) for n in names)}] = {e.code};")  # works on structs too
        elif isinstance(e.type, StructType):  # a NamedTuple into existing variables, field by field
            info = self.mod.classes[e.type.name]
            tmp = "qtrans_unpacked"
            while tmp in self.all_names:
                tmp += "_"
            self.emit(f"{{ const {e.type.cpp()} {tmp} = {e.code};")
            for n, f in zip(names, info.fields):
                if n not in self.ignored:
                    self.emit(f"  {self.ident(n)} = {tmp}.{f.cpp};")
            self.emit("}")
        else:
            lhs = ", ".join("std::ignore" if n in self.ignored else self.ident(n) for n in names)
            self.emit(f"std::tie({lhs}) = {e.code};")

    def store_subscript(self, t, value_node, s, sym):
        """a[i] = x, a[i:j] = v, A[i] = row, A[:, j] = col, A[r0:r1, c0:c1] = M (and +=, -=, ...)."""
        root = subscript_root(t)
        lt = self.var_type(root) if root and (root in self.params or root in self.local_types) else None
        if isinstance(lt, ListType):
            return self.store_list_item(t, value_node, s, sym, lt)
        if isinstance(self.var_type(root) if root else None, OptionalType):
            raise Unsupported(s, f"'{root}' may be None, and storing into it is not supported",
                              f"after the None check, work on a new name: {root}_ = {root}.copy()")
        code, kind = self.array_subscript(t, target=True)
        part = self._last_target_dims  # (reading the right side may index other arrays)
        crossing = _crossing_view(t, value_node)
        if crossing:
            src, dst = ("row", "column") if crossing == "row" else ("column", "row")
            raise Unsupported(s, f"`{ast.unparse(s).strip()}` copies a {src} of '{root}' into a "
                                 f"{dst} of it, and the two share an "
                                 "element: NumPy's result then depends on the order of its copy loop (it reads "
                                 "an element it has already overwritten)",
                              f"write `{ast.unparse(value_node)}.copy()` for the usual meaning")
        v = self.expr(value_node)
        if kind == "mask":
            base, mask = code
            if is_array(v.type):
                raise Unsupported(s, "a[mask] = array is not supported; assign a scalar, or use np.where")
            self.need_scalar(v, value_node, "the value stored into an array")
            op = {"=": "set_op", "+=": "add_op", "-=": "sub_op", "*=": "mul_op", "/=": "div_op"}[sym]
            self.emit(f"qt::masked_update({base}, {mask}, {self.as_double(v).code}, qt::{op}{{}});")
            return
        if kind == "scalar" or not is_array(v.type):
            self.need_scalar(v, value_node, "the value stored into an array")
            rhs = self.as_double(v).code  # a scalar into a slice fills it, as in NumPy
        else:
            if (kind == "mat") != (v.type == MAT):
                raise Unsupported(s, f"assigning a {v.type} into a {'2-D' if kind == 'mat' else '1-D'} "
                                     "part of an array")
            rhs = f"{wrap(v, P_ATOM)}.transpose()" if kind == "row" else v.code
            if self._needs_eval(value_node, root, whole_array=False):
                rhs = _evaluated(rhs, v.type)  # NumPy computes the whole right side before storing
            if not all(same_dim(a, b) for a, b in zip(part, dims_of(v))):
                # NumPy broadcasts the right side into the part (a length-1 Vec fills it)
                op = {"=": "", "+=": ", qt::add_op{}", "-=": ", qt::sub_op{}", "*=": ", qt::mul_op{}",
                      "/=": ", qt::div_op{}"}[sym]
                self.emit(f"qt::store({code}, {rhs}{op});")
                return
        self.emit(f"{code} {sym} {rhs};")

    def store_list_item(self, t, value_node, s, sym, lt):
        """xs[i] = x (and xs[i] += x): the element keeps the list's one type, as Python's would."""
        if isinstance(t.slice, ast.Slice) or not isinstance(t.value, ast.Name):
            raise Unsupported(s, "assigning to a slice of a list is not supported", "build a new list")
        target = self.list_subscript(t, self.expr(t.value))
        v = self.expr(value_node)
        if sym != "=" and lt.elem == INT and sym == "/=":
            raise Unsupported(s, "xs[i] /= ... makes a float in a list of ints: Python keeps each element's type")
        if v.type != lt.elem and not (sym != "=" and lt.elem == FLOAT and v.type in (INT, BOOL)):
            raise Unsupported(s, f"storing a {v.type} in a list of {lt.elem}: Python keeps each element's type "
                                 "(an int prints as 1, a float as 1.0)")
        if lt.elem == FLOAT and self.maybe_int(value_node) and sym == "=":
            raise Unsupported(s, "this value may be a Python int here, in a list of floats",
                              "make it a float on every path (0.0 rather than 0)")
        if sym in ("+=", "-=", "*=") and lt.elem == INT:
            op = {"+=": "qt::iadd", "-=": "qt::isub", "*=": "qt::imul"}[sym]
            self.emit(f"{target.code} = {op}({target.code}, {self.as_type(v, INT).code});")
            return
        code = self.as_double(v).code if lt.elem == FLOAT else v.code
        self.emit(f"{target.code} {sym} {code};")

    def _field_target(self, t, s):
        """`obj.x` as an assignment target: (C++ lvalue, field)."""
        base = self.expr(t.value)
        if not isinstance(base.type, StructType):
            raise Unsupported(s, f"assigning an attribute of a {base.type} is not supported")
        info = self.class_info(base.type.name, t)
        f = info.field(t.attr)
        if f is None:
            what = "a property" if t.attr in info.methods else "not a field"
            raise Unsupported(s, f"{info.name}.{t.attr} is {what}; only fields can be assigned")
        if info.frozen and not getattr(self, "_setattr_ok", False):
            kind = "a NamedTuple" if info.namedtuple else "a frozen dataclass"
            raise Unsupported(s, f"{info.name} is {kind}: assigning {t.attr} raises an error in Python",
                              "make a changed copy with dataclasses.replace(obj, field=value)")
        return self._member(base, f.cpp), f

    def store_attribute(self, t, value_node, s):
        code, f = self._field_target(t, s)
        e = self.expr_raw(value_node) if isinstance(f.type, OptionalType) else self.expr(value_node)
        e = self.convert_to(e, f.type, s, f"field '{f.name}'")
        rhs = e.code
        if is_array(f.type) and self._needs_eval(value_node, subscript_root(t), whole_array=False):
            rhs = _evaluated(rhs, f.type)  # NumPy computes the right side before storing it
        self.emit(f"{code} = {rhs};")

    def s_AnnAssign(self, s):
        if s.value is None:
            return  # a bare `x: float` declares nothing in Python
        self.assign_name(s.target.id, s.value, s, self.inline_at.get(id(s), set()))

    def s_AugAssign(self, s):
        syms = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}
        if isinstance(s.target, ast.Attribute):
            self.aug_attribute(s, syms)
            return
        if isinstance(s.target, ast.Subscript):
            if type(s.op) not in syms:
                raise Unsupported(s, f"{type(s.op).__name__}= on an array element is not supported")
            self.store_subscript(s.target, s.value, s, syms[type(s.op)])
            return
        if isinstance(self.var_type(s.target.id), ListType):  # xs += ys: extends xs in place
            if not isinstance(s.op, ast.Add):
                raise Unsupported(s, f"{type(s.op).__name__}= on a list is not supported")
            call = ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id=s.target.id, ctx=ast.Load()),
                                                                 attr="extend", ctx=ast.Load()),
                                              args=[s.value], keywords=[]), s)
            self.emit(f"{self.expr(call).code};")
            return
        if isinstance(self.var_type(s.target.id), OptionalType) and is_array(self.var_type(s.target.id).inner):
            raise Unsupported(s, f"'{s.target.id}' may be None; in-place arithmetic on it is not supported",
                              f"work on a new name: {s.target.id}_ = {s.target.id} + ...")
        if is_array(self.var_type(s.target.id)):
            name, vt = s.target.id, self.var_type(s.target.id)
            if type(s.op) not in syms:
                raise Unsupported(s, f"{type(s.op).__name__}= on an array is not supported")
            rhs = self.expr(s.value)
            target = self.ident(name)
            if is_array(rhs.type):
                code = rhs.code
                if self._needs_eval(s.value, name, whole_array=True):
                    code = f"({code}).eval()"
                self._emit_inplace(s, target, vt, dims_of(self.variable(s.target)), rhs, code, syms)
            else:
                self.need_scalar(rhs, s.value, "array arithmetic")
                self.emit(f"{target} {syms[type(s.op)]} {self.as_double(rhs).code};")
            return
        name = s.target.id
        vt = self.var_type(name)
        b = self.aug_binops[id(s)]
        result = self.expr(b)
        self.check_assignable(vt, result, s, f"'{name}'")
        if result.type == INT and result.code.startswith("qt::i"):
            self.emit(f"{self.ident(name)} = {result.code};")  # checked: may leave 64 bits
        elif isinstance(s.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)) and result.type == vt:
            rhs = self.promote(self.expr(s.value), vt)
            sym = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}[type(s.op)]
            if isinstance(s.op, ast.Div) and vt != FLOAT:
                raise Unsupported(s, "internal: /= on a non-float variable")  # inference widens it
            self.emit(f"{self.ident(name)} {sym} {rhs.code};")
        else:
            self.emit(f"{self.ident(name)} = {result.code};")
        if name in getattr(self, "flagged", ()):
            # int op int stays an int in Python (x / y never does); a float operand makes a float
            cond = "false" if isinstance(s.op, ast.Div) else self.int_cond(s.value)
            if cond is False:
                raise Unsupported(s, f"'{name}' is shown as text where it may be an int or a float, and this value "
                                     "is one or the other depending on the run in a way qtrans can't follow",
                                  f"start it as a float ({name} = 0.0 rather than 0)")
            if cond is None or cond == "false":
                self.emit(f"{self.flag_name(name)} = false;")
            elif cond != "true":
                self.emit(f"{self.flag_name(name)} = {self.flag_name(name)} && {cond};")

    def _emit_inplace(self, s, target, tt, tdims, rhs, code, syms):
        """target op= rhs for arrays. NumPy broadcasts the right side into the target's shape
        (a length-1 Vec, a (1, m) row, ...) and never changes the target's shape. Where the
        sizes provably match, Eigen's own operator; otherwise qt::broadcast_into."""
        sym = syms[type(s.op)]
        op = {ast.Add: "add_op", ast.Sub: "sub_op", ast.Mult: "mul_op", ast.Div: "div_op"}[type(s.op)]
        rd = dims_of(rhs)
        if rhs.type == tt:
            if all(same_dim(a, b) for a, b in zip(tdims, rd)):
                self.emit(f"{target} {sym} {code};")
            else:
                self.emit(f"qt::broadcast_into({target}, {code}, qt::{op}{{}});")
        elif tt == MAT:  # a Vec across the rows
            if same_dim(tdims[1], rd[0]):
                self.emit(f"{target}.rowwise() {sym} {wrap(rhs, P_ATOM)}.transpose();")
            else:
                self.emit(f"qt::broadcast_into({target}, {wrap(rhs, P_ATOM)}.transpose(), qt::{op}{{}});")
        else:
            raise Unsupported(s, f"{type(s.op).__name__}= of a Mat into a Vec can't broadcast")

    def aug_attribute(self, s, syms):
        """obj.x += v, self.w *= 2.0: the field updated in place (numbers and arrays)."""
        code, f = self._field_target(s.target, s)
        if type(s.op) not in syms:
            raise Unsupported(s, f"{type(s.op).__name__}= on a field is not supported")
        rhs = self.expr(s.value)
        if is_array(f.type):
            if is_array(rhs.type):
                rc = rhs.code
                if self._needs_eval(s.value, subscript_root(s.target), whole_array=False):
                    rc = _evaluated(rc, rhs.type)
                self._emit_inplace(s, code, f.type, (None,) * f.type.rank, rhs, rc, syms)
            else:
                self.need_scalar(rhs, s.value, "array arithmetic")
                self.emit(f"{code} {syms[type(s.op)]} {self.as_double(rhs).code};")
            return
        cur = Expr(code, P_ATOM, f.type, simple=True)
        result = self.binop(s, s.op, cur, rhs)
        self.check_assignable(f.type, result, s, f"field '{f.name}'")
        if result.type == INT and result.code.startswith("qt::i"):
            self.emit(f"{code} = {result.code};")  # checked: may leave 64 bits
        elif result.type == f.type and isinstance(s.op, (ast.Add, ast.Sub, ast.Mult)) or \
                (isinstance(s.op, ast.Div) and f.type == FLOAT):
            self.emit(f"{code} {syms[type(s.op)]} {self.promote(rhs, f.type).code};")
        else:
            self.emit(f"{code} = {self.as_type(result, f.type).code};")

    def s_If(self, s):
        cond = self.cond(s.test)
        if isinstance(cond.value, bool) and not (len(s.orelse) == 1 and isinstance(s.orelse[0], ast.If)):
            # the test is decided by the types (A.ndim == 1 for a Mat): only one branch can run
            dead, live = (s.orelse, s.body) if cond.value else (s.body, s.orelse)
            if dead:
                self.note(f"dead:{s.lineno}", f"line {s.lineno}: `{ast.unparse(s.test)}` is always {cond.value} "
                                              "for the declared types, so the other branch is left out")
            for b in live:
                self.emit_stmt(b)
            return
        self.emit(f"if ({cond.code}) {{")
        self._block(s.body)
        orelse = s.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If):
            elif_ = orelse[0]
            self.current = elif_
            self.flush_comments(elif_.lineno)
            self.comment_floor = elif_.lineno
            c = self.cond(elif_.test)
            self.cur_src = (elif_.lineno, elif_.test.end_lineno)
            self.emit(f"}} else if ({c.code}) {{")
            self._block(elif_.body)
            orelse = elif_.orelse
        if orelse:
            ln = self._keyword_line(getattr(self, "_last_end", None) or s.lineno, orelse[0].lineno, "else")
            self.cur_src = (ln, ln)
            self.emit("} else {")
            self._block(orelse)
        self.emit("}")

    def s_Try(self, s):
        """try / except LinAlgError (or ValueError): a C++ try / catch. Variables assigned in the
        try block are declared before it, so a handler and the code after see them as Python does;
        statements that ran before the error keep their effect in both."""
        self.emit("try {")
        self._block(s.body)
        for h in s.handlers:
            self.current = h
            self.flush_comments(h.lineno)
            self.comment_floor = h.lineno
            self.cur_src = (h.lineno, h.lineno)
            self.emit(f"}} catch (const {self._handler_class(h)}&) {{  // except {ast.unparse(h.type)}")
            self._block(h.body)
        self.emit("}")

    def s_While(self, s):
        cond = self.cond(s.test)
        self.emit(f"while ({cond.code}) {{")
        self._block(s.body)
        self.emit("}")

    def s_For(self, s):
        it = s.iter
        lit = self._literal_items(it)
        if lit is not None:  # over a literal list: a range-for over the same values
            t, items = lit
            decl = "std::string" if t == STR else t.cpp()
            self.emit(f"for ({decl} {self.ident(s.target.id)} : {{{', '.join(items)}}}) {{")
            self._block(s.body)
            self.emit("}")
            return
        if not self._is_range(it):
            arr = self.expr(it)
            if isinstance(arr.type, ListType) and arr.type.elem is not None:
                decl = "const std::string&" if arr.type.elem == STR else arr.type.elem.cpp()
                self.emit(f"for ({decl} {self.ident(s.target.id)} : {arr.code}) {{")
                self._block(s.body)
                self.emit("}")
                return
            if arr.type != VEC_T:
                raise Unsupported(s, "for-loops are supported over range(...) and over a Vec",
                                  "for a Mat, loop over range(A.shape[0]) and take rows")
            src = arr.code if arr.simple else f"qt::Vec({arr.code})"
            self.emit(f"for (double {self.ident(s.target.id)} : {src}) {{")
            self._block(s.body)
            self.emit("}")
            return
        if "range" in self.mod.imports or self.is_variable("range"):
            raise Unsupported(s, "'range' is redefined in this module")
        if it.keywords or not 1 <= len(it.args) <= 3:
            raise Unsupported(s, "range() takes 1 to 3 positional arguments")
        args = [self.expr(a) for a in it.args]
        for a, n in zip(args, it.args):
            if a.type not in (INT, BOOL):
                raise Unsupported(n, f"range() arguments must be ints, got {a.type}",
                                  "Python raises TypeError for floats too")
        step = 1
        if len(args) == 3:
            step = args[2].value
            if not isinstance(step, int) or isinstance(step, bool):
                raise Unsupported(it, "range() step must be an integer constant so the loop direction is known")
            if step == 0:
                raise Unsupported(it, "range() step 0 raises ValueError in Python")
        start = args[0] if len(args) >= 2 else Expr("0", P_ATOM, INT, int_lit=True, value=0, simple=True)
        stop, stop_node = (args[1], it.args[1]) if len(args) >= 2 else (args[0], it.args[0])
        var = self.ident(s.target.id)
        # i lies in [start, stop - 1] (step > 0) or [stop + 1, start] (step < 0); unknown ends
        # are the int64 limits, so i + 1 (or i - 1 going down) can never overflow
        (s0, s1), (e0, e1) = iv_of(start), iv_of(stop)
        self.loop_iv[s.target.id] = (s0, e1 - 1) if step > 0 else (e0 + 1, s1)
        # Python evaluates range() once. If the bound could change during the loop (it calls
        # a function, or reads a variable the body assigns), evaluate it once up front.
        body_stores = {n.id for b in s.body for n in ast.walk(b) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        body_stores |= {n.target.id for b in s.body for n in ast.walk(b) if isinstance(n, ast.AugAssign)}
        risky = any(isinstance(n, ast.Call) or (isinstance(n, ast.Name) and n.id in body_stores)
                    for n in ast.walk(stop_node))
        stop_code = wrap(stop, P_REL - 1)
        if risky:
            end = f"{var}_end"
            while end in self.all_names:
                end += "_"
            self.all_names.add(end)
            self.emit(f"const long long {end} = {self.as_type(stop, INT, False).code};")
            stop_code = end
        cmp = "<" if step > 0 else ">"
        inc = {1: f"++{var}", -1: f"--{var}"}.get(step, f"{var} += {step}" if step > 0 else f"{var} -= {-step}")
        self.emit(f"for (long long {var} = {start.code}; {var} {cmp} {stop_code}; {inc}) {{")
        self._block(s.body)
        self.emit("}")

    def s_Return(self, s):
        ret = self.sig.ret
        none = s.value is None or (isinstance(s.value, ast.Constant) and s.value.value is None)
        if none and ret == VOID:
            self.emit("return;")
            return
        if isinstance(ret, OptionalType):
            e = Expr("std::nullopt", P_ATOM, NONE) if s.value is None else self.expr_raw(s.value)
            self.check_assignable(ret, e, s, "the return value")
            self.emit(f"return {self.to_optional(e, ret).code};")
            return
        if s.value is None:
            raise Unsupported(s, f"bare `return` gives None, but the function returns {ret}",
                              f"annotate the return as `-> {ret} | None` if None is a real result")
        if ret == VOID:
            raise Unsupported(s, "returning a value from a function annotated -> None")
        maybe = self.mod.__dict__.setdefault("maybe_int", set())
        if isinstance(s.value, ast.Tuple):
            if not isinstance(ret, TupleType) or len(ret.elts) != len(s.value.elts):
                raise Unsupported(s, f"returns {len(s.value.elts)} values but is annotated -> {ret}")
            parts = []
            for k, (node, et) in enumerate(zip(s.value.elts, ret.elts)):
                e = self.expr(node)
                self.check_assignable(et, e, s, "the return value")
                if et == FLOAT and (e.type in (INT, BOOL) or self.maybe_int(node)):
                    maybe.add((self.fdef.name, k))  # a Python int on this path
                parts.append(self.as_double(e).code if et == FLOAT else e.code)
            self.emit(f"return {{{', '.join(parts)}}};")
            return
        e = self.expr(s.value)
        self.check_assignable(ret, e, s, "the return value")
        if ret == FLOAT and (e.type in (INT, BOOL) or self.maybe_int(s.value)):
            maybe.add((self.fdef.name, None))  # `return 0` from a float function: a Python int here
        code = self.as_double(e).code if ret == FLOAT else e.code
        self.emit(f"return {code};")

    def s_Raise(self, s):
        exc = s.exc
        if exc is None:
            raise Unsupported(s, "bare `raise` (re-raise) is not supported")
        if isinstance(exc, ast.Call):
            cls_node, args = exc.func, exc.args
            if exc.keywords:
                raise Unsupported(s, "keyword arguments to exceptions are not supported")
        else:
            cls_node, args = exc, []
        cls = ast.unparse(cls_node)
        cpp_cls = lib.EXCEPTIONS.get(cls)
        if cpp_cls is None:
            raise Unsupported(s, f"raising {cls} is not supported",
                              "use ValueError, ZeroDivisionError, ArithmeticError, OverflowError or RuntimeError")
        if not args:
            msg = cpp_string(cls)
        elif len(args) == 1:
            e = self.text_value(args[0])  # a message: a str, an f-string, str(x), ...
            msg = e.code
        else:
            raise Unsupported(s, "an exception with several arguments is not supported", "pass one message")
        self.emit(f"throw {cpp_cls}({msg});")

    def s_Assert(self, s):
        cond = self.truth(self.expr(s.test), s)
        if s.msg is not None and isinstance(s.msg, ast.Constant) and isinstance(s.msg.value, str):
            msg = s.msg.value
        else:
            msg = "assertion failed: " + ast.unparse(s.test)
        self.note("assert", "assert statements always run in C++; `python -O` strips them")
        self.emit(f"if (!{wrap(cond, P_ATOM)}) throw std::logic_error({cpp_string(msg)});")
