"""Random typed Python functions in the subset qtrans translates: the fuzzer's generator.

Every expression is built for a requested type (float, int, bool, Vec, Mat) from productions
that give that type, so each program is well-typed Python by construction. What it may still
do at run time (divide by zero, index out of range, take the log of a negative number, mix
shapes NumPy won't broadcast) is the point: the C++ must do what Python does there too, or
differ only in the ways the verifier knows (IEEE-754 inf/NaN where Python raises).

Kept small on purpose: loop bounds are small literals, len(), .shape or an int parameter (the
verifier draws those from -10 to 100), array sizes come from the same, and nesting is limited,
so a program runs in milliseconds and its arrays stay tiny.
"""
from __future__ import annotations

import random

F, I, B, V, M = "float", "int", "bool", "Vec", "Mat"
HEADER = "import math\nimport numpy as np\nfrom qtrans.typing import Mat, Vec\n"

_FLOATS = ["0.0", "1.0", "2.0", "0.5", "-1.5", "3.0", "1e-3", "100.0", "-0.0", "0.25"]
_INTS = ["0", "1", "2", "3", "-1", "5"]
_MATH1 = ["sqrt", "exp", "log", "sin", "cos", "tanh", "fabs", "atan", "log1p", "expm1"]
_NP1 = ["abs", "exp", "sqrt", "log", "sin", "cos", "tanh", "floor", "cumsum"]
_REDUCE = ["sum", "mean", "max", "min", "std", "var"]


class _NoVar(Exception):
    """A production needs a variable of a type there is none of: choose another."""


class Gen:
    def __init__(self, rng: random.Random, max_depth: int = 3, max_block: int = 4, max_nest: int = 2):
        self.r = rng
        self.max_depth, self.max_block, self.max_nest = max_depth, max_block, max_nest
        self.known = {}  # functions written earlier in this module: name -> (param types, return type)

    def new_module(self):
        self.known = {}

    def _call(self, ret, d):
        """A call of an earlier function of this module that returns `ret`. Arrays go in as
        variables where there are some, so a callee that changes its argument in place changes
        the caller's array (which the C++ must do too, through a reference)."""
        fs = [n for n, (_, r) in self.known.items() if r == ret]
        if not fs:
            raise _NoVar(ret)
        n = self._pick(fs)
        args, taken = [], set()
        for t in self.known[n][0]:
            # arrays as variables, each passed once: qtrans needs a variable for a parameter
            # changed in place, and refuses the same array passed twice (the refusals are tested
            # elsewhere; here they'd only waste the callers)
            free = [v for v in self._vars(t) if v not in taken] if t in (V, M) else []
            if free and self.r.random() < 0.9:
                v = self._pick(free)
                taken.add(v)
                args.append(v)
            else:
                args.append(self.expr(t, d))
        return f"{n}({', '.join(args)})"

    # --- names and scopes ------------------------------------------------------------------
    def _vars(self, t):
        return [n for scope in self.scopes for n, vt in scope.items() if vt == t]

    def _fresh(self, t):
        self.count += 1
        return f"{'xnbvm'[[F, I, B, V, M].index(t)]}{self.count}"

    def _pick(self, xs):
        return xs[self.r.randrange(len(xs))]

    def _w(self, options):
        """Weighted choice: [(weight, value), ...]."""
        total = sum(w for w, _ in options)
        x = self.r.uniform(0, total)
        for w, v in options:
            x -= w
            if x <= 0:
                return v
        return options[-1][1]

    # --- expressions ----------------------------------------------------------------------------
    def leaf(self, t):
        vs = self._vars(t) + (self.loop_vars if t == I else [])
        if vs and self.r.random() < 0.75:
            return self._pick(vs)
        if t == F:
            return self._pick(_FLOATS)
        if t == I:
            return self._pick(_INTS)
        if t == B:
            return self._pick(["True", "False"])
        if t == V:
            return self._pick(["np.ones(3)", "np.array([1.0, -2.0, 0.5])", "np.zeros(2)", "np.array([4.0])"])
        return self._pick(["np.eye(2)", "np.ones((2, 3))", "np.array([[1.0, 2.0], [3.0, 4.5]])"])

    def size(self):
        """An int to size an array or bound a loop: small, so programs stay fast."""
        opts = [(3, self._pick(_INTS[:4] + ["4"]))]
        for v in self._vars(V):
            opts.append((2, f"len({v})"))
        for m in self._vars(M):
            opts.append((1, f"{m}.shape[{self.r.randrange(2)}]"))
        for n in [v for v in self._vars(I) if v.startswith("p")]:
            opts.append((1, n))
        return self._w(opts)

    def expr(self, t, d=0):
        if d >= self.max_depth or self.r.random() < 0.2:
            return self.leaf(t)
        for _ in range(6):
            try:
                return getattr(self, f"_e_{t.lower()}")(d + 1)
            except _NoVar:
                continue
        return self.leaf(t)

    def _e_float(self, d):
        e = lambda t=F: self.expr(t, d)  # noqa: E731
        opts = [
            (6, lambda: f"({e()} {self._pick(['+', '-', '*', '/'])} {e()})"),
            (2, lambda: f"({e()} {self._pick(['+', '-', '*'])} {e(I)})"),
            (1, lambda: f"(-{e()})"),
            (3, lambda: f"math.{self._pick(_MATH1)}({e()})"),
            (1, lambda: f"abs({e()})"),
            (1, lambda: f"float({e(I)})"),
            (1, lambda: f"({e()} ** {self._pick(['2', '0.5', '3', '-1'])})"),
            (1, lambda: f"{self._pick(['min', 'max'])}({e()}, {e()})"),
            (1, lambda: f"({e()} if {e(B)} else {e()})"),
            (3, lambda: f"float(np.{self._pick(_REDUCE)}({e(self._pick([V, V, M]))}))"),
            (2, lambda: f"float({e(V)} @ {e(V)})"),
            (2, lambda: f"{self._named(V)}[{e(I)}]"),
            (1, lambda: f"{self._named(M)}[{e(I)}, {e(I)}]"),
            (2, lambda: self._call(F, d)),
            (1, lambda: f"float(np.count_nonzero({self._mask(d)}))"),
        ]
        return self._w(opts)()

    def _mask(self, d):
        """A boolean array: a comparison, or two combined with & / | / ~."""
        cmp = self._pick(["<", "<=", ">", ">=", "!="])
        m = f"({self.expr(V, d)} {cmp} {self.expr(F, d)})"
        k = self.r.random()
        if k < 0.25:
            m = f"({m} {self._pick(['&', '|'])} ({self.expr(V, d)} > {self.expr(F, d)}))"
        elif k < 0.35:
            m = f"(~{m})"
        return m

    def _e_int(self, d):
        e = lambda t=I: self.expr(t, d)  # noqa: E731
        opts = [
            (5, lambda: f"({e()} {self._pick(['+', '-', '*'])} {e()})"),
            (2, lambda: f"({e()} {self._pick(['//', '%'])} {e()})"),
            (2, lambda: f"len({e(V)})"),
            (1, lambda: f"{e(M)}.shape[{self.r.randrange(2)}]"),
            (1, lambda: f"int({e(F)})"),
            (1, lambda: f"abs({e()})"),
            (1, lambda: f"({e()} ** {self._pick(['2', '3'])})"),
            (1, lambda: f"{self._pick(['min', 'max'])}({e()}, {e()})"),
            (1, lambda: f"({e()} if {e(B)} else {e()})"),
            (1, lambda: self._call(I, d)),
        ]
        return self._w(opts)()

    def _e_bool(self, d):
        e = lambda t=B: self.expr(t, d)  # noqa: E731
        cmp = lambda: self._pick(["<", "<=", ">", ">=", "==", "!="])  # noqa: E731
        def pair(t):  # x < x is always False: a C++ compiler warns, and it says nothing
            a, b = self.expr(t, d), self.expr(t, d)
            if b == a:
                b = self.leaf(t)
            return (a, b) if b != a else (a, "0.5" if t == F else "7")
        opts = [
            (4, lambda: "({0} {2} {1})".format(*pair(F), cmp())),
            (2, lambda: "({0} {2} {1})".format(*pair(I), cmp())),
            (1, lambda: f"({e()} {self._pick(['and', 'or'])} {e()})"),
            (1, lambda: f"(not {e()})"),
            (1, lambda: f"bool(np.{self._pick(['any', 'all'])}({self.expr(V, d)} {cmp()} {self.expr(F, d)}))"),
            (1, lambda: f"math.isfinite({self.expr(F, d)})"),
        ]
        return self._w(opts)()

    def _named(self, t):
        """A variable of type t (qtrans refuses indexing or slicing an expression)."""
        vs = self._vars(t)
        if not vs:
            raise _NoVar(t)
        return self._pick(vs)

    def _array_rhs(self, t):
        """The right side of an array assignment: a bare name would alias it (b = a shares
        memory in NumPy), which qtrans refuses when either is later changed in place."""
        rhs = self.expr(t)
        view = rhs.isidentifier() or (rhs.endswith("]") and not rhs.endswith("])"))  # a, a[1:], A[i]
        return f"{rhs}.copy()" if view else rhs

    def _slice(self):
        return self._pick(["1:", ":-1", "::2", "::-1", "1:3", ":2", "-2:"])

    def _e_vec(self, d):
        e = lambda t=V: self.expr(t, d)  # noqa: E731
        op = lambda: self._pick(["+", "-", "*", "/"])  # noqa: E731
        opts = [
            (5, lambda: f"({e()} {op()} {e()})"),
            (3, lambda: f"({e()} {op()} {self.expr(F, d)})"),
            (1, lambda: f"({self.expr(F, d)} {op()} {e()})"),
            (1, lambda: f"(-{e()})"),
            (3, lambda: f"np.{self._pick(_NP1)}({e()})"),
            (2, lambda: f"{self._named(V)}[{self._slice()}]"),
            (2, lambda: f"({e(M)} @ {e()})"),
            (1, lambda: f"({e()} @ {e(M)})"),
            (1, lambda: f"{self._named(M)}[{self.expr(I, d)}]"),
            (1, lambda: f"{self._named(M)}[:, {self.expr(I, d)}]"),
            (2, lambda: f"np.where({e()} > {self.expr(F, d)}, {e()}, {self._pick([self.expr(F, d), e()])})"),
            (2, lambda: f"np.{self._pick(['maximum', 'minimum'])}({e()}, {self._pick([self.expr(F, d), e()])})"),
            (1, lambda: f"np.clip({e()}, {self._pick(['-1.0', '0.0'])}, {self._pick(['1.0', '2.5'])})"),
            (1, lambda: f"{e(M)}.{self._pick(['mean', 'sum'])}(axis={self.r.randrange(2)})"),
            (1, lambda: f"np.{self._pick(['zeros', 'ones'])}({self.size()})"),
            (1, lambda: f"np.full({self.size()}, {self.expr(F, d)})"),
            (1, lambda: f"np.linspace({self.expr(F, d)}, {self.expr(F, d)}, {self.size()})"),
            (1, lambda: f"{self._named(V)}.copy()"),
            (1, lambda: f"({e()} ** 2)"),
            (2, lambda: self._call(V, d)),
            (2, lambda: f"{self._named(V)}[{self._mask(d)}]"),
        ]
        return self._w(opts)()

    def _e_mat(self, d):
        e = lambda t=M: self.expr(t, d)  # noqa: E731
        op = lambda: self._pick(["+", "-", "*", "/"])  # noqa: E731
        opts = [
            (4, lambda: f"({e()} {op()} {e()})"),
            (2, lambda: f"({e()} {op()} {self.expr(F, d)})"),
            (2, lambda: f"({e()} {op()} {self.expr(V, d)})"),
            (2, lambda: f"{e()}.T"),
            (2, lambda: f"({e()} @ {e()})"),
            (1, lambda: f"np.outer({self.expr(V, d)}, {self.expr(V, d)})"),
            (1, lambda: f"np.diag({self.expr(V, d)})"),
            (1, lambda: f"np.eye({self._pick(['1', '2', '3'])})"),
            (1, lambda: f"np.zeros(({self.size()}, {self.size()}))"),
            (1, lambda: f"np.linalg.inv({e()})"),
            (1, lambda: f"np.linalg.solve({e()}, {e()})"),
            (1, lambda: f"np.{self._pick(['abs', 'exp', 'sqrt'])}({e()})"),
            (1, lambda: f"{self._named(M)}[{self._pick(['1:', ':-1', '1:3', ':2', '-2:'])}]"),  # rows: no step
            (1, lambda: f"np.where({e()} > {self.expr(F, d)}, {e()}, {self.expr(F, d)})"),
            (1, lambda: self._call(M, d)),
        ]
        return self._w(opts)()

    # --- statements -----------------------------------------------------------------------------
    def block(self, nest, ret):
        out = []
        for _ in range(self.r.randint(1, self.max_block)):
            out += self.stmt(nest, ret)
        return out

    def stmt(self, nest, ret):
        kinds = [(4, "new"), (2, "rebind"), (2, "aug"), (1, "store"), (1, "call")]
        if nest < self.max_nest:
            kinds += [(2, "if"), (2, "for")]
        k = self._w(kinds)
        if k == "new":
            t = self._w([(3, F), (1, I), (3, V), (1, M), (1, B)])
            rhs = self._array_rhs(t) if t in (V, M) else self.expr(t)
            name = self._fresh(t)
            self.scopes[-1][name] = t
            return [f"{name} = {rhs}"]
        if k == "rebind":
            t = self._w([(3, F), (1, I), (3, V), (1, M)])
            vs = [v for v in self._vars(t) if not (t in (V, M) and v.startswith("p"))]  # params: in place only
            if not vs:
                return self.stmt(nest, ret)
            return [f"{self._pick(vs)} = {self._array_rhs(t) if t in (V, M) else self.expr(t)}"]
        if k == "aug":
            t = self._w([(3, F), (1, I), (3, V), (1, M)])
            vs = self._vars(t)
            if not vs:
                return self.stmt(nest, ret)
            ops = ["+=", "-=", "*="] if t == I else ["+=", "-=", "*=", "/="]
            rhs_t = t if t in (F, I) else self._pick([t, F, V] if t == M else [t, F])
            return [f"{self._pick(vs)} {self._pick(ops)} {self.expr(rhs_t)}"]
        if k == "call":  # for its effect on the arrays passed in
            t = self._pick([F, V, M, I])
            try:
                return [self._call(t, 1)]
            except _NoVar:
                return self.stmt(nest, ret)
        if k == "store":
            vs, ms = self._vars(V), self._vars(M)
            if not vs and not ms:
                return self.stmt(nest, ret)
            if vs and (not ms or self.r.random() < 0.6):
                v = self._pick(vs)
                return [self._pick([f"{v}[{self.expr(I)}] = {self.expr(F)}",
                                    f"{v}[{self._slice()}] = {self._pick([self.expr(F), self.expr(V)])}",
                                    f"{v}[{v} {self._pick(['<', '>'])} {self.expr(F)}] = {self.expr(F)}"])]
            m = self._pick(ms)
            return [self._pick([f"{m}[{self.expr(I)}, {self.expr(I)}] = {self.expr(F)}",
                                f"{m}[{self.expr(I)}] = {self.expr(V)}"])]
        if k == "if":
            cond = self.expr(B)
            body = self._scoped(lambda: self.block(nest + 1, ret))
            if self.r.random() < 0.3:
                body.append(f"return {self.expr(ret)}")
            lines = [f"if {cond}:"] + ["    " + x for x in body]
            if self.r.random() < 0.5:
                other = self._scoped(lambda: self.block(nest + 1, ret))
                lines += ["else:"] + ["    " + x for x in other]
            return lines
        # for
        self.count += 1
        i = f"i{self.count}"
        bound = self.size()
        self.loop_vars.append(i)
        body = self._scoped(lambda: self.block(nest + 1, ret))
        self.loop_vars.remove(i)
        return [f"for {i} in range({bound}):"] + ["    " + x for x in body]

    def _scoped(self, f):
        self.scopes.append({})
        try:
            return f()
        finally:
            self.scopes.pop()

    # --- a function -----------------------------------------------------------------------------
    def function(self, name: str) -> str:
        self.count, self.loop_vars = 0, []
        params = [(f"p{k}", self._w([(3, F), (2, I), (3, V), (2, M), (1, B)])) for k in range(self.r.randint(1, 4))]
        ret = self._w([(4, F), (3, V), (1, M), (1, I)])
        self.scopes = [dict(params)]
        body = self.block(0, ret) + [f"return {self.expr(ret)}"]
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        self.known[name] = ([t for _, t in params], ret)
        return "\n".join([f"def {name}({sig}) -> {ret}:"] + ["    " + x for x in body]) + "\n"


def module(rng: random.Random, names) -> str:
    g = Gen(rng)
    return HEADER + "\n\n" + "\n\n".join(g.function(n) for n in names)


if __name__ == "__main__":
    import sys
    print(module(random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0), ["f0", "f1", "f2"]))
