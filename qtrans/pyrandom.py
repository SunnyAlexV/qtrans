"""Python's random module (0.14), and math.fsum.

    random.seed(42)                 qt::py::random_module().seed(42LL)
    random.gauss(0.0, 0.2)          qt::py::random_module().gauss(0.0, 0.2)
    random.choice(names)            qt::py::random_module().choice(names)
    math.fsum(xs)                   qt::py::fsum(xs)

qtrans_pyrandom.hpp reproduces CPython's generator and Lib/random.py exactly, so the numbers are
Python's, in Python's order. The module functions share one hidden generator, as in Python:

* a call that draws is an effect, like rng.normal(): one per statement (C++ doesn't fix the order
  of a statement's parts), and a function that draws, itself or through what it calls, counts as
  drawing wherever it is called (Translator.py_random_users);
* a program must call random.seed(...) before anything draws: Python seeds the module from the
  operating system when it is imported, so the numbers would change every run;
* `qtrans verify` seeds both sides the same way before each call of a function that draws, and
  compares the generator after the call (its next 32 bits and gauss()'s saved value).

Arguments are bound as the running Python binds them: the defaults are read from its own
functions (gauss() has defaults from 3.11, expovariate() from 3.12).
"""
from __future__ import annotations

import ast
import inspect
import random as _random

from .core import P_ATOM, Expr
from .errors import Unsupported
from .types import BOOL, FLOAT, INT, VOID, ListType, is_array

R = "qt::py::random_module()"

# name -> (parameters as C++ sees them, result) ; "list" / "elem" follow the population's type
_SPECS = {
    "random": ((), FLOAT),
    "seed": (("a",), VOID),
    "getrandbits": (("k",), INT),
    "randrange": (("start", "stop", "step"), INT),
    "randint": (("a", "b"), INT),
    "uniform": (("a", "b"), FLOAT),
    "triangular": (("low", "high", "mode"), FLOAT),
    "gauss": (("mu", "sigma"), FLOAT),
    "normalvariate": (("mu", "sigma"), FLOAT),
    "lognormvariate": (("mu", "sigma"), FLOAT),
    "expovariate": (("lambd",), FLOAT),
    "choice": (("seq",), "elem"),
    "shuffle": (("x",), VOID),
    "sample": (("population", "k"), "list"),
    "choices": (("population", "weights", "k"), "list"),
}
NOT_YET = {"betavariate", "gammavariate", "vonmisesvariate", "paretovariate", "weibullvariate", "binomialvariate",
           "randbytes", "getstate", "setstate"}
DRAWS = set(_SPECS) - {"seed"}


def _python_params(name):
    """{parameter: default (inspect.Parameter.empty if none)} of the running Python's
    random.<name>, in order, without *args / keyword-only ones (which are refused if given)."""
    sig = inspect.signature(getattr(_random, name))
    return {p.name: p.default for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) or p.name in ("k", "counts", "cum_weights")}


class PyRandomOps:
    """Mixin for ExprTranslator: random.<f>(...) and math.fsum(...)."""

    def py_random_call(self, node, qual):
        name = qual.split(".", 1)[1]
        src = ast.unparse(node.func)
        if name == "Random" or name == "SystemRandom":
            raise Unsupported(node, f"{src}(...) objects are not supported yet",
                              "seed the module's generator and call its functions: random.seed(42); random.gauss(0.0, 1.0)")
        if name in NOT_YET:
            raise Unsupported(node, f"{src}() is not supported yet",
                              "supported: random, seed, getrandbits, randrange, randint, uniform, triangular, gauss, "
                              "normalvariate, lognormvariate, expovariate, choice, shuffle, sample, choices")
        if name not in _SPECS or not callable(getattr(_random, name, None)):
            raise Unsupported(node, f"{src} is not a function of the random module that qtrans knows")
        got = self._bind_random(node, name, src)
        return getattr(self, f"_rnd_{name}", self._rnd_plain)(node, name, got, src)

    def _bind_random(self, node, name, src):
        params = _python_params(name)
        names = list(params)
        given = {}
        positional = [n for n in names if n not in ("k", "counts", "cum_weights")] if name in ("choices", "sample") \
            else names
        if name == "sample":
            positional = ["population", "k"]
        for i, a in enumerate(node.args):
            if isinstance(a, ast.Starred) or i >= len(positional):
                raise Unsupported(node, f"{src}() takes at most {len(positional)} positional arguments here")
            given[positional[i]] = a
        for k in node.keywords:
            if k.arg is None or k.arg not in params:
                raise Unsupported(node, f"{src}() has no parameter '{k.arg}' (Python raises TypeError)")
            if k.arg in given:
                raise Unsupported(node, f"{src}() got two values for '{k.arg}'")
            given[k.arg] = k.value
        for p in ("counts", "cum_weights", "version", "random"):
            if p in given and not (isinstance(given[p], ast.Constant) and given[p].value is None):
                raise Unsupported(node, f"{src}({p}=...) is not supported yet")
        out = {}
        for n in names:
            if n in given:
                out[n] = given[n]
            elif params[n] is inspect.Parameter.empty:
                if name == "seed" and n == "a":
                    break
                raise Unsupported(node, f"{src}() needs '{n}' in this Python ({_python_version()}), which gives it "
                                        "no default (Python raises TypeError)")
            else:
                out[n] = ast.copy_location(ast.Constant(params[n]), node)
        return out

    # --- the functions ----------------------------------------------------------------------
    def _num(self, node, t, what):
        e = self.expr(node)
        self.need_scalar(e, node, what)
        if t == INT:
            if e.type not in (INT, BOOL):
                raise Unsupported(node, f"{what} must be an int (Python raises TypeError for a {e.type})")
            return self.int_arg(e)
        return self.as_double(e)

    def _rnd_plain(self, node, name, got, src):
        args, ret = _SPECS[name]
        t = INT if ret == INT else FLOAT
        codes = [self._num(got[a], t, f"argument '{a}' of {src}()").code for a in args if a in got]
        return Expr(f"{R}.{name}({', '.join(codes)})", P_ATOM, ret)

    def _rnd_seed(self, node, name, got, src):
        if "a" not in got or (isinstance(got["a"], ast.Constant) and got["a"].value is None):
            raise Unsupported(node, f"{src}() without a seed: Python seeds from the operating system, so the "
                                    "numbers differ on every run", "give it an int: random.seed(42)")
        e = self.expr(got["a"])
        if e.type not in (INT, BOOL):
            raise Unsupported(got["a"], f"random.seed() of a {e.type}: only int seeds are supported "
                                        "(Python hashes other kinds its own way)", "seed with an int")
        return Expr(f"{R}.seed({self.int_arg(e).code})", P_ATOM, VOID)

    def _rnd_getrandbits(self, node, name, got, src):
        k = got["k"]
        if not (isinstance(k, ast.Constant) and type(k.value) is int and 0 <= k.value <= 63):
            raise Unsupported(node, "getrandbits(k) needs k as a number from 0 to 63 here (a long long holds "
                                    "the result)")
        return Expr(f"{R}.getrandbits({k.value})", P_ATOM, INT)

    def _rnd_randrange(self, node, name, got, src):
        start = self._num(got["start"], INT, f"argument 'start' of {src}()")
        stop_n, step_n = got.get("stop"), got.get("step")
        if stop_n is None or (isinstance(stop_n, ast.Constant) and stop_n.value is None):
            if not (isinstance(step_n, ast.Constant) and step_n.value == 1):
                raise Unsupported(node, f"{src}() with a step but no stop (Python raises TypeError)")
            return Expr(f"{R}.randrange({start.code})", P_ATOM, INT)
        stop = self._num(stop_n, INT, f"argument 'stop' of {src}()")
        if isinstance(step_n, ast.Constant) and step_n.value == 1:
            return Expr(f"{R}.randrange({start.code}, {stop.code})", P_ATOM, INT)
        step = self._num(step_n, INT, f"argument 'step' of {src}()")
        return Expr(f"{R}.randrange({start.code}, {stop.code}, {step.code})", P_ATOM, INT)

    def _rnd_triangular(self, node, name, got, src):
        low = self._num(got["low"], FLOAT, f"argument 'low' of {src}()")
        high = self._num(got["high"], FLOAT, f"argument 'high' of {src}()")
        mode = got.get("mode")
        if mode is None or (isinstance(mode, ast.Constant) and mode.value is None):
            return Expr(f"{R}.triangular({low.code}, {high.code})", P_ATOM, FLOAT)
        m = self._num(mode, FLOAT, f"argument 'mode' of {src}()")
        return Expr(f"{R}.triangular({low.code}, {high.code}, {m.code})", P_ATOM, FLOAT)

    def _population(self, node, src):
        xs = self.expr(node)
        if not isinstance(xs.type, ListType) or xs.type.elem is None:
            raise Unsupported(node, f"{src}() of a {xs.type}: a list is supported here",
                              "make it a list: list(...) or [..]")
        return xs

    def _rnd_choice(self, node, name, got, src):
        xs = self._population(got["seq"], src)
        return Expr(f"{R}.choice({xs.code})", P_ATOM, xs.type.elem)

    def _rnd_shuffle(self, node, name, got, src):
        x = got["x"]
        if not isinstance(x, ast.Name):
            raise Unsupported(node, f"{src}() of an expression: Python shuffles that list and drops it",
                              "shuffle a list variable")
        xs = self._population(x, src)
        return Expr(f"{R}.shuffle({xs.code})", P_ATOM, VOID)

    def _rnd_sample(self, node, name, got, src):
        xs = self._population(got["population"], src)
        k = self._num(got["k"], INT, f"argument 'k' of {src}()")
        return Expr(f"{R}.sample({xs.code}, {k.code})", P_ATOM, xs.type)

    def _rnd_choices(self, node, name, got, src):
        xs = self._population(got["population"], src)
        k = self._num(got["k"], INT, f"argument 'k' of {src}()")
        w = got.get("weights")
        if w is None or (isinstance(w, ast.Constant) and w.value is None):
            return Expr(f"{R}.choices({xs.code}, {k.code})", P_ATOM, xs.type)
        we = self.expr(w)
        if not (isinstance(we.type, ListType) and we.type.elem in (FLOAT, INT)):
            raise Unsupported(w, f"weights must be a list of floats or ints here, got {we.type}")
        return Expr(f"{R}.choices({xs.code}, {we.code}, {k.code})", P_ATOM, xs.type)

    # --- math.fsum --------------------------------------------------------------------------
    def fsum_call(self, node):
        if len(node.args) != 1 or node.keywords:
            raise Unsupported(node, "math.fsum() takes one iterable")
        a = node.args[0]
        if isinstance(a, ast.GeneratorExp):  # fsum(f(x) for x in xs): the values, then the sum
            e = self.e_ListComp(a)
        else:
            e = self.expr(a)
        t = e.type
        if is_array(t) and t.rank == 1:
            return Expr(f"qt::py::fsum({e.code})", P_ATOM, FLOAT)
        if isinstance(t, ListType) and t.elem in (FLOAT, INT, BOOL):
            return Expr(f"qt::py::fsum({e.code})", P_ATOM, FLOAT)
        raise Unsupported(node, f"math.fsum() of a {t}: a list, a Vec, or a generator of numbers is supported")


def _python_version():
    import sys
    return ".".join(map(str, sys.version_info[:2]))
