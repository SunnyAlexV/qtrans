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
import inspect

from . import library as lib
from .errors import Unsupported
from .expressions import (P_ATOM, P_REL, P_UNARY, Expr, ExprTranslator, _Unknown, cpp_ident, wrap)
from .types import VEC as VEC_T
from .types import RNG
from .types import BOOL, FLOAT, INT, MAT, VOID, ArrayType, MaskType, Scalar, TupleType, assignable, is_array, join

_STATEMENT_MESSAGES = {
    "FunctionDef": ("nested functions", "move the function to module level"),
    "AsyncFunctionDef": ("async functions", None),
    "ClassDef": ("classes inside functions", "classes and dataclasses arrive in stage 4"),
    "Try": ("try/except blocks", "catching exceptions inside translated functions isn't supported yet; raising is"),
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
    """`a[i]`, `A[i, j]`, `a[1:3]` -> 'a' (the array variable being indexed)."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def view_root(node, types):
    """If `node` evaluates to an existing array or a view of one (NumPy shares memory for
    `b = a`, slices, rows, and `.T`), return that array's name; otherwise None (a new array)."""
    if isinstance(node, ast.Name):
        return node.id if is_array(types.get(node.id)) else None
    if isinstance(node, ast.Attribute) and node.attr == "T":
        return view_root(node.value, types)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "transpose" \
            and not node.args:
        return view_root(node.func.value, types)
    if isinstance(node, ast.Subscript):
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        base_t = types.get(node.value.id) if isinstance(node.value, ast.Name) else None
        is_view = any(isinstance(p, ast.Slice) for p in parts) or (base_t == MAT and len(parts) == 1)
        if is_view:
            return view_root(node.value, types)
    return None


def is_docstring(stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def cpp_string(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        c = chr(b)
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif 32 <= b < 127:
            out.append(c)
        else:
            out.append(f"\\{b:03o}")  # octal escapes: always exactly 3 digits, no ambiguity
    return '"' + "".join(out) + '"'


def _contains_break(stmts) -> bool:
    for s in stmts:
        if isinstance(s, ast.Break):
            return True
        if isinstance(s, ast.If) and (_contains_break(s.body) or _contains_break(s.orelse)):
            return True
    return False


def always_returns(stmts) -> bool:
    """Does every path through `stmts` end in return or raise?"""
    for s in stmts:
        if isinstance(s, (ast.Return, ast.Raise)):
            return True
        if isinstance(s, ast.If) and always_returns(s.body) and always_returns(s.orelse):
            return True
        if isinstance(s, ast.While) and isinstance(s.test, ast.Constant) and s.test.value is True \
                and not _contains_break(s.body):
            return True
    return False


class _Event:
    """One assignment to a local variable: name <- payload."""
    __slots__ = ("name", "kind", "payload", "index", "stmt")

    def __init__(self, name, kind, payload, stmt, index=None):
        self.name, self.kind, self.payload, self.stmt, self.index = name, kind, payload, stmt, index


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

    # --- variables -------------------------------------------------------------------
    @property
    def locals_set(self):
        return set(self.locals_order)

    def is_variable(self, name):
        return (name in self.params or name in self.loop_vars or name in self.ignored
                or name in self._locals_cache)

    def variable(self, node):
        n = node.id
        if n in self.loop_vars:
            t = self.loop_types.get(n, INT)
        elif n in self.params:
            t = self.params[n]
        else:
            t = self.local_types.get(n)
            if t is None:
                if self.inferring:
                    raise _Unknown(n)
                raise Unsupported(node, f"could not infer a type for '{n}'")
        return Expr(cpp_ident(n), P_ATOM, t, simple=True)

    # --- driver ----------------------------------------------------------------------
    def translate(self):
        self._locals_cache = set()
        self.scan(self.body)
        self.check_loop_vars()
        self.finish_scan()
        self.infer_types()
        self.check_aliasing()
        self.check_random_draws()
        self.plan_declarations()
        self.check_generator_declarations()
        self.definite_assignment()
        if self.sig.ret != VOID and not always_returns(self.body):
            raise Unsupported(self.fdef, f"{self.fdef.name}() can reach its end without returning "
                                         "(Python would return None)", "add a final return or raise")
        self.emit_block(self.body)
        self.flush_comments(self.fdef.end_lineno + 1)
        return self.lines

    # --- pass 1: scan ----------------------------------------------------------------
    def _add_event(self, name, kind, payload, stmt, index=None):
        self.events.append(_Event(name, kind, payload, stmt, index))
        if name not in self.params and name not in self.locals_order:
            self.locals_order.append(name)

    def scan(self, stmts):
        allowed = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.If, ast.While,
                   ast.For, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Raise, ast.Assert)
        for s in stmts:
            if not isinstance(s, allowed):
                what, hint = _STATEMENT_MESSAGES.get(type(s).__name__, (type(s).__name__, None))
                raise Unsupported(s, f"{what} are not supported in stage 1", hint)
            if isinstance(s, ast.For):
                if not isinstance(s.target, ast.Name):
                    raise Unsupported(s, "for-loops must use a single loop variable")
                if s.orelse:
                    raise Unsupported(s, "for ... else is not supported")
                if s.target.id in self.params:
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
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Subscript):
                self._check_store_target(s.target, s)  # a[i] += x: updates an element, no new variable
            elif isinstance(s, ast.AugAssign):
                if not isinstance(s.target, ast.Name):
                    raise Unsupported(s, "augmented assignment must target a name or an array element",
                                      "attribute assignment arrives in stage 4")
                b = ast.copy_location(ast.BinOp(left=ast.Name(id=s.target.id, ctx=ast.Load()),
                                                op=s.op, right=s.value), s)
                self.aug_binops[id(s)] = b
                self._add_event(s.target.id, "expr", b, s)

    def _scan_target(self, t, value, stmt):
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
        elif isinstance(t, ast.Subscript):
            self._check_store_target(t, stmt)
        elif isinstance(t, ast.Attribute):
            raise Unsupported(stmt, "attribute assignment is not supported in stage 1",
                              "objects arrive in stage 4")
        else:
            raise Unsupported(stmt, f"cannot assign to {type(t).__name__}")

    def _check_store_target(self, t, stmt):
        if not isinstance(t.value, ast.Name):
            raise Unsupported(stmt, "assignment into a nested subscript (a[1:][0] = ...) is not supported",
                              "index the array directly")

    @staticmethod
    def _is_range(it):
        return isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"

    def _loop_type(self, s):
        return INT if self._is_range(s.iter) else FLOAT

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
    def event_type(self, ev):
        self.current = ev.stmt
        e = self.expr(ev.payload)
        t = e.type
        if ev.kind == "elt":
            i, n = ev.index
            if not isinstance(t, TupleType) or len(t.elts) != n:
                raise Unsupported(ev.stmt, f"cannot unpack a {t} value into {n} names")
            t = t.elts[i]
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
                    if new is None:
                        raise Unsupported(ev.stmt, f"'{ev.name}' is assigned both {old} and {t} values; "
                                                   "a C++ variable has one type", "use separate names")
                    if new != old:
                        self.local_types[ev.name] = new
                        changed = True
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
                    self.event_type(ev)  # re-run outside inference mode to surface the real error
                except _Unknown as u:
                    raise Unsupported(ev.stmt, f"could not infer a type for '{name}': it depends on "
                                               f"'{u.args[0]}', whose type is never determined",
                                      f"annotate it, e.g. {name}: float = ...")
        # notes from inference are discarded: types were provisional then
        self.notes, self._noted = [], set()
        for name in self.locals_order:
            seen = set()
            for ev in self.events:
                if ev.name == name:
                    try:
                        seen.add(self.event_type(ev))
                    except Unsupported:
                        pass
            if self.local_types[name] == FLOAT and (INT in seen or BOOL in seen):
                self.note(f"widen:{name}", f"'{name}' holds an int in some assignments and a float in "
                                           "others; declared double (same values)")

    # --- pass 3b: aliasing ---------------------------------------------------------------------
    def _alias_sources(self, value, types):
        """Arrays `value` may share memory with: itself, a view of it (slice, row, .T), either
        branch of a conditional, or a parameter a called function hands back."""
        if isinstance(value, ast.IfExp):
            return self._alias_sources(value.body, types) | self._alias_sources(value.orelse, types)
        root = view_root(value, types)
        if root:
            return {root}
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                and value.func.id in self.mod.returns_alias:
            out = set()
            for i in self.mod.returns_alias[value.func.id]:
                if i < len(value.args):
                    r = view_root(value.args[i], types)
                    if r:
                        out.add(r)
            return out
        return set()

    def check_aliasing(self):
        """In NumPy `b = a`, slices, rows and `.T` share memory with the original; in C++ the
        variable is a copy. The two agree unless one of the names is later modified in place,
        so any such pair is refused rather than translated into a silent difference."""
        types = {**self.local_types, **self.params}
        arrays = {n for n, t in types.items() if is_array(t)}
        if not arrays:
            return
        mutated = set()
        for n in ast.walk(self.fdef):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for leaf in target_leaves(t):
                        if isinstance(leaf, ast.Subscript):
                            mutated.add(subscript_root(leaf))
                        elif isinstance(n, ast.AugAssign) and isinstance(leaf, ast.Name) and leaf.id in arrays:
                            mutated.add(leaf.id)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.mod.sigs:
                callee = self.mod.sigs[n.func.id]
                pos = {p.name: i for i, p in enumerate(callee.params)}
                pairs = list(enumerate(n.args)) + [(pos.get(k.arg, -1), k.value) for k in n.keywords]
                for i, a in pairs:
                    if isinstance(a, ast.Name) and a.id in arrays and 0 <= i < len(callee.params) \
                            and callee.params[i].mode == "ref":
                        mutated.add(a.id)
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
                        raise Unsupported(n, f"'{x}' and '{y}' share memory in NumPy (assignment, slices "
                                             f"and .T are views, not copies) and '{which}' is modified in "
                                             "place, so a C++ copy would give different results",
                                          f"if you want an independent array, write {x} = ....copy()")
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
    def _draws(self, node, types) -> int:
        """How many times evaluating `node` advances a generator: rng.method(...) calls, and
        calls that pass a generator to another function."""
        count = 0
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and types.get(f.value.id) == RNG:
                count += 1
            elif isinstance(f, ast.Name) and any(isinstance(a, ast.Name) and types.get(a.id) == RNG
                                                 for a in list(n.args) + [k.value for k in n.keywords]):
                count += 1
        return count

    def check_random_draws(self):
        """Python evaluates left to right; C++ leaves the order of operands and arguments
        unspecified. With two draws in one expression (rng.random() - rng.random()), C++ could
        take them in the other order and compute a different number, so that is refused."""
        types = {**self.local_types, **self.params}
        if RNG not in types.values():
            return

        def own_exprs(s):
            return [e for e in self._own_exprs(s) if e is not None]

        def visit(stmts):
            for s in stmts:
                if sum(self._draws(e, types) for e in own_exprs(s)) > 1:
                    raise Unsupported(s, "two random draws in one statement: C++ doesn't fix which is "
                                         "evaluated first, so the draws could be used in a different order",
                                      "draw into separate variables, one per line")
                if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None:
                    targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                    if any(isinstance(t, ast.Name) and types.get(t.id) == RNG for t in targets) and not (
                            isinstance(s.value, ast.Call) and self.qualify(s.value.func) == "numpy.random.default_rng"):
                        raise Unsupported(s, "a generator can only be created with np.random.default_rng(seed); "
                                             "`g2 = g` would share one stream in Python but copy it in C++",
                                          "pass the generator itself to the functions that need it")
                for block in (getattr(s, "body", []), getattr(s, "orelse", [])):
                    visit(block)
        visit(self.body)

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
                if isinstance(s.target, ast.Subscript):
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
                        vt = self.expr_type_quiet(s.value)
                        if (isinstance(vt, TupleType) and len(set(names)) == len(names)
                                and all(n in mine and n not in declared for n in names)
                                and all(self.local_types[n] == et for n, et in zip(names, vt.elts))):
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
                if isinstance(s.target, ast.Subscript):
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
        self.lines.append("    " * self.depth + line)

    def flush_comments(self, upto_line):
        full = self.mod.full_comments
        for ln in range(self.comment_floor + 1, upto_line):
            if ln in full:
                self.lines.append("    " * self.depth + "// " + full[ln])
        self.comment_floor = max(self.comment_floor, upto_line - 1)

    def emit_block(self, stmts):
        for s in stmts:
            self.emit_stmt(s)

    def emit_stmt(self, s):
        self.current = s
        self.flush_comments(s.lineno)
        for n in self.hoist_at.get(id(s), ()):
            t = self.local_types[n]
            self.emit(f"{t.cpp()} {cpp_ident(n)} = {t.zero()};" if isinstance(t, Scalar)
                      else f"{t.cpp()} {cpp_ident(n)}{{}};")
        compound = isinstance(s, (ast.If, ast.While, ast.For))
        last = s.lineno if compound else s.end_lineno
        texts = [self.mod.inline_comments[ln] for ln in range(s.lineno, last + 1) if ln in self.mod.inline_comments]
        self._tail = ("  // " + "; ".join(texts)) if texts else None
        self.comment_floor = last
        getattr(self, "s_" + type(s).__name__)(s)
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
        e = self.expr(v)
        self.emit(f"{e.code};")

    def var_type(self, name):
        return self.params.get(name) or self.local_types[name]

    def assign_name(self, name, value_node, s, inline_set):
        e = self.expr(value_node)
        if e.type == VOID:
            raise Unsupported(s, "assigning the result of a function that returns None")
        vt = self.var_type(name)
        what = f"parameter '{name}'" if name in self.params else f"'{name}'"
        self.check_assignable(vt, e, s, what)
        code = self.as_double(e).code if vt == FLOAT else e.code
        if vt == RNG and name in inline_set and code.startswith("qt::Rng(") and code.endswith(")"):
            self.emit(f"qt::Rng {cpp_ident(name)}{code[len('qt::Rng'):]};")
            return
        if is_array(vt) and self._needs_eval(value_node, name, whole_array=True):
            code = f"({code}).eval()"  # Eigen is lazy: read all of the old value before overwriting it
        if name in inline_set:
            self.emit(f"{vt.cpp()} {cpp_ident(name)} = {code};")
        else:
            self.emit(f"{cpp_ident(name)} = {code};")

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
            lhs = ", ".join("std::ignore" if n in self.ignored else cpp_ident(n) for n in names)
            self.emit(f"std::tie({lhs}) = std::make_tuple({', '.join(parts)});")
            return
        e = self.expr(s.value)
        if not isinstance(e.type, TupleType) or len(e.type.elts) != len(names):
            raise Unsupported(s, f"cannot unpack a {e.type} value into {len(names)} names")
        for n, et in zip(names, e.type.elts):
            if n not in self.ignored and not assignable(self.var_type(n), et):
                raise Unsupported(s, f"'{n}' is declared {self.var_type(n)} but gets a {et} value")
        if inline and set(names) <= inline:
            self.emit(f"auto [{', '.join(cpp_ident(n) for n in names)}] = {e.code};")
        else:
            lhs = ", ".join("std::ignore" if n in self.ignored else cpp_ident(n) for n in names)
            self.emit(f"std::tie({lhs}) = {e.code};")

    def store_subscript(self, t, value_node, s, sym):
        """a[i] = x, a[i:j] = v, A[i] = row, A[:, j] = col, A[r0:r1, c0:c1] = M (and +=, -=, ...)."""
        code, kind = self.array_subscript(t, target=True)
        root = subscript_root(t)
        v = self.expr(value_node)
        if kind == "scalar" or not is_array(v.type):
            self.need_scalar(v, value_node, "the value stored into an array")
            rhs = self.as_double(v).code  # a scalar into a slice fills it, as in NumPy
        else:
            if (kind == "mat") != (v.type == MAT):
                raise Unsupported(s, f"assigning a {v.type} into a {'2-D' if kind == 'mat' else '1-D'} "
                                     "part of an array")
            rhs = f"{wrap(v, P_ATOM)}.transpose()" if kind == "row" else v.code
            if self._needs_eval(value_node, root, whole_array=False):
                rhs = f"({rhs}).eval()"  # NumPy computes the whole right side before storing
        self.emit(f"{code} {sym} {rhs};")

    def s_AnnAssign(self, s):
        if s.value is None:
            return  # a bare `x: float` declares nothing in Python
        self.assign_name(s.target.id, s.value, s, self.inline_at.get(id(s), set()))

    def s_AugAssign(self, s):
        syms = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}
        if isinstance(s.target, ast.Subscript):
            if type(s.op) not in syms:
                raise Unsupported(s, f"{type(s.op).__name__}= on an array element is not supported")
            self.store_subscript(s.target, s.value, s, syms[type(s.op)])
            return
        if is_array(self.var_type(s.target.id)):
            name, vt = s.target.id, self.var_type(s.target.id)
            if type(s.op) not in syms:
                raise Unsupported(s, f"{type(s.op).__name__}= on an array is not supported")
            rhs = self.expr(s.value)
            target = cpp_ident(name)
            if is_array(rhs.type):
                code = rhs.code
                if self._needs_eval(s.value, name, whole_array=True):
                    code = f"({code}).eval()"
                if rhs.type == vt:
                    self.emit(f"{target} {syms[type(s.op)]} {code};")
                elif vt == MAT:
                    self.emit(f"{target}.rowwise() {syms[type(s.op)]} {wrap(rhs, P_ATOM)}.transpose();")
                else:
                    raise Unsupported(s, f"{type(s.op).__name__}= of a Mat into a Vec can't broadcast")
            else:
                self.need_scalar(rhs, s.value, "array arithmetic")
                self.emit(f"{target} {syms[type(s.op)]} {self.as_double(rhs).code};")
            return
        name = s.target.id
        vt = self.var_type(name)
        b = self.aug_binops[id(s)]
        result = self.expr(b)
        self.check_assignable(vt, result, s, f"'{name}'")
        if isinstance(s.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)) and result.type == vt:
            rhs = self.promote(self.expr(s.value), vt)
            sym = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}[type(s.op)]
            if isinstance(s.op, ast.Div) and vt != FLOAT:
                raise Unsupported(s, "internal: /= on a non-float variable")  # inference widens it
            self.emit(f"{cpp_ident(name)} {sym} {rhs.code};")
        else:
            self.emit(f"{cpp_ident(name)} = {result.code};")

    def s_If(self, s):
        cond = self.truth(self.expr(s.test), s)
        self.emit(f"if ({cond.code}) {{")
        self._block(s.body)
        orelse = s.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If):
            elif_ = orelse[0]
            self.current = elif_
            self.flush_comments(elif_.lineno)
            self.comment_floor = elif_.lineno
            c = self.truth(self.expr(elif_.test), elif_)
            self.emit(f"}} else if ({c.code}) {{")
            self._block(elif_.body)
            orelse = elif_.orelse
        if orelse:
            self.emit("} else {")
            self._block(orelse)
        self.emit("}")

    def s_While(self, s):
        cond = self.truth(self.expr(s.test), s)
        self.emit(f"while ({cond.code}) {{")
        self._block(s.body)
        self.emit("}")

    def s_For(self, s):
        it = s.iter
        if not self._is_range(it):
            arr = self.expr(it)
            if arr.type != VEC_T:
                raise Unsupported(s, "for-loops are supported over range(...) and over a Vec",
                                  "for a Mat, loop over range(A.shape[0]) and take rows")
            src = arr.code if arr.simple else f"qt::Vec({arr.code})"
            self.emit(f"for (double {cpp_ident(s.target.id)} : {src}) {{")
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
        var = cpp_ident(s.target.id)
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
            self.emit(f"const long long {end} = {self.as_type(stop, INT).code};")
            stop_code = end
        cmp = "<" if step > 0 else ">"
        inc = {1: f"++{var}", -1: f"--{var}"}.get(step, f"{var} += {step}" if step > 0 else f"{var} -= {-step}")
        self.emit(f"for (long long {var} = {start.code}; {var} {cmp} {stop_code}; {inc}) {{")
        self._block(s.body)
        self.emit("}")

    def s_Return(self, s):
        ret = self.sig.ret
        if s.value is None:
            if ret != VOID:
                raise Unsupported(s, f"bare `return` gives None, but the function returns {ret}")
            self.emit("return;")
            return
        if ret == VOID:
            raise Unsupported(s, "returning a value from a function annotated -> None")
        if isinstance(s.value, ast.Tuple):
            if not isinstance(ret, TupleType) or len(ret.elts) != len(s.value.elts):
                raise Unsupported(s, f"returns {len(s.value.elts)} values but is annotated -> {ret}")
            parts = []
            for node, et in zip(s.value.elts, ret.elts):
                e = self.expr(node)
                self.check_assignable(et, e, s, "the return value")
                parts.append(self.as_double(e).code if et == FLOAT else e.code)
            self.emit(f"return {{{', '.join(parts)}}};")
            return
        e = self.expr(s.value)
        self.check_assignable(ret, e, s, "the return value")
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
            msg = cls
        elif len(args) == 1 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            msg = args[0].value
        else:
            raise Unsupported(s, "exception messages must be plain string literals",
                              "f-strings and formatting aren't supported in stage 1")
        self.emit(f"throw {cpp_cls}({cpp_string(msg)});")

    def s_Assert(self, s):
        cond = self.truth(self.expr(s.test), s)
        if s.msg is not None and isinstance(s.msg, ast.Constant) and isinstance(s.msg.value, str):
            msg = s.msg.value
        else:
            msg = "assertion failed: " + ast.unparse(s.test)
        self.note("assert", "assert statements always run in C++; `python -O` strips them")
        self.emit(f"if (!{wrap(cond, P_ATOM)}) throw std::logic_error({cpp_string(msg)});")
