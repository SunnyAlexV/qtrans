# The qtrans specification

What qtrans translates, what the C++ it writes computes, and what it refuses. This is the
contract; the README explains how and why. Every example marked `spec` below is checked by
`tests/test_spec.py`: it is translated (or refused) and the C++ (or the reason) must contain the
quoted text, and the ones marked `program` are compiled and run, and must print what Python
prints. So this document can't silently drift from what qtrans does.

## 1. The guarantee

For a function or program qtrans translates, the C++ is meant to do what the Python does, on
every input. That is a design rule, applied construct by construct below, and checked by
differential testing (`qtrans verify` runs both on the same inputs and compares them, section
14) and by the test suite; it is not a formal proof. What it covers:

1. **The same result.** Bit for bit wherever the operations are exact in IEEE-754 double
   arithmetic (`+ - * /`, `sqrt`, comparisons, conversions) or where qtrans ports the exact
   algorithm Python or NumPy uses (`//` and `%` on floats, `round`, `sum`, `math.fsum`,
   `math.gamma`, `np.sum`'s pairwise summation, `np.mean`, `np.var`, `np.quantile`, `np.cumsum`,
   NumPy's and Python's random generators, text formatting). Where both sides call a C library
   function (`exp`, `log`, `sin`, `pow`, ...), they call the same one, so on one machine the
   result is the same; NumPy's and SciPy's own vectorised `exp`, `pow` and `erf`, and BLAS /
   LAPACK against Eigen, agree to rounding, and `qtrans verify` measures by how much.
2. **The same errors.** The same exception class (`ValueError`, `IndexError`, `KeyError`,
   `ZeroDivisionError`, `OverflowError`, `numpy.linalg.LinAlgError`, your own classes) with the
   same message, where Python raises one; the float errors of Python's numbers included
   (section 4.2), unless a function is marked `# qtrans: ieee`.
3. **The same output.** What a program prints, character for character, and its exit status.
4. **The same sharing.** Where Python shares an object (an array passed to a function, an object
   of a class hierarchy, a generator), a change made through one name is seen through the
   other, in the C++ too; where C++ would copy instead, and the copy could show, qtrans refuses.

Where it can't keep these, **it refuses**: the function is not translated, and the report gives
the line and the reason, usually with a hint. A function calling a refused function is refused
too. There are no silent approximations.

The exceptions to 1 and 2 are listed in section 15 ("Differences kept by design"), each with
its reason, and the report names them next to the functions they affect. `qtrans verify`
classifies every case it sees as `match`, `ill-cond`, `known` (one of those differences),
`edge` or `mismatch` (section 14), and counts each group separately, so a `known` case is
never counted as a match.

**What the C++ needs:** C++17; IEEE-754 doubles; 64-bit `long long`; no fast-math (every
generated header refuses to compile under `-ffast-math` or `/fp:fast`); floating-point
contraction off (`-ffp-contract=off`, `/fp:precise`), so `a * b + c` is not fused into one
rounding. GCC, Clang, Apple Clang and MSVC are tested.

## 2. Types

| Python | C++ |
|---|---|
| `float` | `double` |
| `int` | `long long` (checked: see 4.1) |
| `bool` | `bool` |
| `str` | `std::string` (UTF-8; indexes count characters) |
| `Vec`, `Mat` (`from qtrans.typing import Vec, Mat`) | `qt::Vec` (`Eigen::ArrayXd`), `qt::Mat` (row-major) |
| `IVec` (1-D int64 array) | `qt::IVec` |
| `np.random.Generator` | `qt::Rng` (PCG64) |
| `tuple[float, int, ...]` | `std::tuple<...>` |
| `list[float]`, `list[int]`, `list[str]`, `list[tuple[...]]`, `list[Payoff]` | `std::vector<...>` |
| `dict[str, float]` (str or int keys; float, int or str values) | `qt::py::dict<...>` (insertion order) |
| `X \| None`, `Optional[X]` for a number or array | `std::optional<X>` |
| a `@dataclass`, `NamedTuple` or plain class | a `struct` |
| a class of a hierarchy (with a base or subclasses) | `std::shared_ptr<...>` to a `struct` |
| `-> None` | `void` |

Parameters and returns are annotated, or typed from how the file calls the function (one C++
version per kind of call where calls differ). Locals are inferred: each gets the join of what is
assigned to it (`bool < int < float`; two subclasses join to their common base).

```python spec translated "double f(double x, long long n)"
def f(x: float, n: int) -> float:
    return x * n
```

```python spec translated "std::shared_ptr<Payoff> p;"
class Payoff:
    def __init__(self, k: float):
        self.k = k

class Call(Payoff):
    pass

class Put(Payoff):
    pass

def f(up: bool) -> float:
    if up:
        p = Call(1.0)
    else:
        p = Put(2.0)
    return p.k
```

Refused: a variable holding two unrelated types; `np.ndarray` without a rank; lists of lists;
lists holding several types (`[1, 2.5]` keeps each element's type in Python).

```python spec refused "a C++ variable has one type"
def f(b: bool) -> float:
    if b:
        x = 1.0
    else:
        x = "one"
    return 0.0
```

```python spec refused "a list holding several types"
def f() -> float:
    xs = [1, 2.5]
    return xs[0]
```

## 3. Statements

Assignment (also tuple unpacking, swaps, `a[i] = x`, slices of arrays, fields), augmented
assignment, `if` / `elif` / `else`, `while`, `for` over `range(...)`, a list, a `Vec`, an `IVec`,
a dict (`d`, `.keys()`, `.values()`, `.items()`), a string, a literal list, `enumerate` and `zip`,
`break`, `continue`, `else` on loops, `return`, `raise`, `assert`, `try` / `except` / `else` /
`finally`, `del d[k]`, `with np.errstate(...)` (only its body: it changes NumPy's warnings, not
values), `pass`, docstrings, comments (carried into the C++).

Refused: `global` / `nonlocal` rebinding, `yield` (generators), `async`, `match`, `lambda`
outside `key=` and `default_factory=`, nested functions, `with` other than `np.errstate`.

```python spec refused "generators (yield) are not supported"
def f(n: int):
    for i in range(n):
        yield i
```

```python spec refused "async functions are not supported"
async def f(x: float) -> float:
    return x
```

A function must return on every path if it returns a value (Python would return `None`):

```python spec refused "can reach its end without returning"
def f(x: float) -> float:
    if x > 0:
        return x
```

## 4. Numbers

### 4.1 int

An int is a `long long`: Python's int rules hold within 64 bits (about ±9.2 × 10¹⁸), and there
is no arbitrary-precision int. Arithmetic that could leave that range is checked (`qt::iadd`,
`qt::imul`, ...) and raises `OverflowError` where Python's unlimited int would go on, rather
than wrap silently; arithmetic qtrans can show stays inside 64 bits is plain C++. An int that is
surely beyond 64 bits whatever the input (`2 ** 100`) is refused. `//` and `%` round toward −∞,
as Python's do, and raise `ZeroDivisionError` for a zero divisor; `/` of two ints is a float;
`**` with a constant exponent is exact (a negative exponent gives a float).

```python spec translated "qt::floordiv(a, b)" "qt::mod(a, b)"
def f(a: int, b: int) -> int:
    return a // b + a % b
```

```python spec translated "qt::py::truediv(a, b)"
def f(a: int, b: int) -> float:
    return a / b
```

An int exponent that isn't a constant is refused: the result is an int or a float depending on
its sign at run time.

```python spec refused "non-constant exponent"
def f(a: int, b: int) -> float:
    return a ** b
```

### 4.2 float

`//` and `%` on floats are CPython's (`float_floor_div`, `float_mod`); `round(x)` rounds half to
even, `round(x, n)` rounds the exact decimal value, as Python does; `int(x)` raises `ValueError`
for NaN and `OverflowError` for infinities and values past 64 bits.

`x ** y` and `math.pow` are the C library's `pow`, as CPython calls it. A C++ compiler turns
`std::pow(x, 2.0)` into `x * x` and `std::pow(x, -1.0)` into `1.0 / x`, which round differently
for about 1 value in 1,000 (glibc), so qtrans calls `qt::pow`, which keeps the call
(`qt::py::power` is `qt::pow` plus Python's errors, below):

```python spec translated "qt::py::power(s, 2.0)"
def f(s: float) -> float:
    return s ** 2
```

`np.power(x, 2)` is NumPy's square (its fast path), so there it is `x * x`:

```python spec translated "x * x"
import numpy as np

def f(x: float) -> float:
    return np.power(x, 2)
```

**Float errors.** Python raises on a float division by zero (`ZeroDivisionError`), an
overflowing `**`, `math.exp` or `math.cosh` (`OverflowError`) and a math domain error
(`math.sqrt(-1)`, `math.log(0.0)`: `ValueError`), where a C++ double gives inf or NaN. Since
1.0 the C++ raises them as Python does, with the running Python's message, everywhere: each
check is a compare that is almost never taken (1% to 18% per call on scalar pricing functions,
`tools/float_error_cost.py`; nothing for NumPy arrays, which don't raise).

```python spec translated "qt::py::truediv(a, b)" "qt::py::math_log(x)"
import math

def f(a: float, b: float, x: float) -> float:
    return a / b + math.log(x)
```

NumPy's numbers don't raise: `np.float64(1.0) / 0.0` is inf, with a warning, in Python too, so
there the C++ keeps IEEE-754's result. A value that is a NumPy number on some runs and a Python
float on others (`max(np.exp(x), 0.0)`) makes Python raise on some runs and not on others; the
C++ then gives NumPy's inf or NaN, and the report says so next to the function (inside a `try`
that may catch the error, where the two would take different branches, it is refused).

`# qtrans: ieee` on a `def` (or on a line of its own at the top of the file, for every
function of the module) gives the checks up: the function keeps IEEE-754's inf and NaN, and
`qtrans verify` reports the inputs where that shows as `known`, never as a match. Inside a `try`
that may catch one of these errors the hint has no effect.

```python spec translated "return a / b;"
def f(a: float, b: float) -> float:  # qtrans: ieee
    return a / b
```

### 4.3 Evaluation order

Python evaluates left to right; C++ leaves the order of operands and arguments unspecified. So a
statement may contain at most one call that changes something (a random draw, a method that
changes its object, a function that changes an argument or an object of a hierarchy, a function
that prints), and must not otherwise read what that call changes:

```python spec refused "two random draws in one statement"
import numpy as np

def f(rng: np.random.Generator) -> float:
    return rng.random() - rng.random()
```

A `print()` line is the exception: its values are worked out in order (a braced list, which
C++ does evaluate left to right).

Where two parts of an expression can raise, the order decides which error Python raises:
`math.log(S / K) / (sigma * math.sqrt(T))` raises `ZeroDivisionError` for `S = K = 0.0` and
`T < 0`, and the C++ would raise `ValueError` if it worked out the right side first. So when
both operands of an operator on numbers, or several arguments of a call to a function, method
or class of the file (and the object a method is called on), can raise or change something (a
call that modifies an argument, prints or draws a random number), the earlier ones are worked out
first, into temporaries; and in `xs[k] += v` the element is found (IndexError, KeyError) before
`v` is worked out, as Python does:

```python spec translated "const double left = qt::py::math_log(qt::py::truediv(S, K));"
import math

def d1(S: float, K: float, sigma: float, T: float) -> float:
    return math.log(S / K) / (sigma * math.sqrt(T))
```

A library function's arguments (`max(...)`, `math.*`, `np.*`) are not reordered: where one of
them changes something and another may raise, it is refused (the state after the error would
differ), with a hint to work the first out on its own line. Which error comes out is not yet
kept for the arguments of library functions (`max(a / b, c / d)`), for array or object
arguments, nor between a number and an array (`math.log(c) * (A @ x)`, where a temporary would
outlive Eigen's lazy array expression): there, when two parts raise at once, which of the two
errors comes out may differ (section 15).

## 5. Text

`str` values, `+`, `*`, `==`, ordering (by code point), `in`, indexing and slicing by character,
`len`, and the methods `split`, `strip` / `lstrip` / `rstrip`, `replace`, `find` / `index` /
`count`, `startswith` / `endswith`, `upper`, `lower`, `capitalize`, `title`, `swapcase`,
`casefold`, the `is...` tests, `zfill`, `ljust` / `rjust` / `center`, `join`; `float(s)` and
`int(s)` read text as Python does. Case mapping and character classes are exact for all of
Unicode: `qtrans_unicode.hpp` is written from the running Python, character by character.

```python spec translated "qt::py::upper"
def f(s: str) -> str:
    return s.strip().upper()
```

Printing: `print` (`sep`, `end`, `file=sys.stderr`, `flush`), f-strings, `str`, `repr`,
`format`, `"...".format`, `%` formatting. Floats show as Python's `repr` (the shortest decimal
that reads back as the same double); format specs follow CPython's code, ported; arrays print as
NumPy prints them; NumPy scalars' `repr` names their type (`np.float64(0.5)`), as NumPy 2 does.

```python spec program
import numpy as np

x = 0.1 + 0.2
v = np.array([0.5, 1e-9])
print(x, f"{x:.3f}", "%5.1e" % x, round(2.675, 2), repr(np.max(v)))
print(np.arange(5), v, [1.5, 2.0], {"a": 1})
```

Not supported, with the reason: `rsplit`, `partition`, `rfind`, `splitlines`, `expandtabs`,
`translate`, `encode`.

## 6. Functions

Positional and keyword arguments, defaults (spelled out in the C++ where a later argument is
given), recursion, calls in any order (declarations come first). How a parameter is passed
follows what Python shares:

| Parameter | C++ |
|---|---|
| a number, `bool` | by value |
| `str` | `const std::string&` (by value if rebound) |
| an array the function only reads | `const qt::Vec&` |
| an array the function changes in place | `qt::Vec&`: the caller sees the change, as in Python |
| an array rebound (`x = x * 2`) | by value |
| an array both changed in place and rebound | refused: the caller would see the first change, not the rest |
| a generator | `qt::Rng&`: drawing advances the caller's |
| a list or dict | `const std::vector<...>&`; changing one passed in is refused |
| an object of a class hierarchy | `const std::shared_ptr<...>&`: the object is shared |

```python spec translated "void scale(qt::Vec& v, double k)"
from qtrans.typing import Vec

def scale(v: Vec, k: float) -> None:
    v *= k
```

```python spec refused "both modified in place and reassigned"
from qtrans.typing import Vec

def f(v: Vec) -> Vec:
    v[0] = 1.0
    v = v * 2.0
    return v
```

Refused: `*args`, `**kwargs`, keyword-only parameters, decorators (other than `@property`,
`@staticmethod`, `@classmethod`, `@abstractmethod`, `@dataclass`), default arguments that are
arrays or objects (Python creates them once and shares them between calls).

## 7. Classes

### 7.1 Dataclasses, NamedTuples and plain classes

`@dataclass` (with `frozen`, `eq`, `field(default=..., default_factory=lambda: ..., init=False,
compare=False, repr=False)`), `NamedTuple`, and plain classes whose fields come from `__init__`.
`__post_init__`, `@property`, `@staticmethod`, `@classmethod` alternate constructors, `__repr__`
and `__str__`, `==` field by field (a dataclass), `dataclasses.replace`, `copy.deepcopy`.
A method that doesn't change `self` is `const`.

```python spec translated "struct Option {" "double price() const;"
from dataclasses import dataclass

@dataclass(frozen=True)
class Option:
    S: float
    K: float

    def price(self) -> float:
        return max(self.S - self.K, 0.0)
```

Objects not in a hierarchy are C++ values (copied on assignment, as structs are). They agree
with Python while nothing changes a shared object; where two names for one object exist and one
is then changed, it is refused:

```python spec refused "are the same object in Python"
from dataclasses import dataclass

@dataclass
class P:
    x: float

def f(p: P) -> float:
    q = p
    q.x = 2.0
    return p.x
```

Refused: class attributes with values, property setters, defining `__eq__`, `__add__` and the
other operators, `==` on an object holding an array (Python raises), a plain class's `==`
(identity, outside a hierarchy), printing a plain class without `__repr__` (Python shows its
address), array defaults without `default_factory`.

### 7.2 Class hierarchies

One base class from the same file (plain or `@dataclass`, the same kind as the subclass), any
depth, abstract bases (`ABC`, `@abstractmethod`). Objects of a hierarchy are held by
`std::shared_ptr`, so they are shared exactly as Python shares them, and a method a subclass
redefines is `virtual` / `override` (`= 0` when abstract). `super().__init__(...)` (first in
`__init__`), `super().method(...)`, an inherited `__init__`, `isinstance`, `type(x).__name__`,
`is`, `==` between plain objects (identity), `list[Base]` holding any subclass, printing through
each class's own `__repr__` or dataclass repr.

```python spec translated "virtual double value(double s) const = 0;" "return std::min(Call::value(s), cap);"
from abc import ABC, abstractmethod

class Payoff(ABC):
    def __init__(self, k: float):
        self.k = k

    @abstractmethod
    def value(self, s: float) -> float:
        ...

class Call(Payoff):
    def value(self, s: float) -> float:
        return max(s - self.k, 0.0)

class Capped(Call):
    def __init__(self, k: float, cap: float):
        super().__init__(k)
        self.cap = cap

    def value(self, s: float) -> float:
        return min(super().value(s), self.cap)
```

Refused, with the reason: two bases; a base from another module; mixing `@dataclass` and plain
classes; an override with other parameter names, types or defaults (C++ takes them from the
variable's class, Python from the object's); creating an object with abstract methods left;
`self` passed on as a value; `@classmethod` in a hierarchy; `==` on dataclass objects of a
hierarchy; `dataclasses.replace` or `copy` of one; a field only a subclass has, read through the
base.

```python spec refused "a C++ override must keep them"
class A:
    def __init__(self):
        self.x = 1.0

    def f(self, y: float) -> float:
        return y

class B(A):
    def f(self, y: int) -> float:
        return 1.0
```

## 8. Lists, tuples and dicts

Lists (section 2's element types): literals, `[]` typed by what is appended, `append`,
`extend`, `insert`, `pop`, `index`, `count`, `copy`, `sort` (also `key=`, `reverse=`),
`reverse`, `clear`, `len`, `sum` (CPython's: compensated from 3.12), `min`, `max`, `sorted`,
`list(range(n))`, indexing and slicing with Python's rules (negative indexes, `IndexError`), `+`,
`*`, `in`, comprehensions, loops. Tuples: literals, indexing by constant, unpacking, returning
several values. Dicts keep insertion order: `d[k]` (`KeyError` when absent), `d[k] = v`, `in`,
`len`, `get(k, default)`, `pop`, `del`, `setdefault`, `update`, `copy`, `clear`, `==`, loops,
comprehensions, `sum(d.values())`, `sorted(d)`.

A list or dict is one object in Python, however many names it has. C++ copies, so wherever the
sharing would show it is refused: `b = a` where either is then changed, a function changing one
passed to it, a loop changing what it goes through.

```python spec refused "the C++ copies"
def f() -> float:
    a = [1.0, 2.0]
    b = a
    b.append(3.0)
    return a[0] + len(a)
```

## 9. Exceptions

`raise` (with or without a message, `from`, bare to re-raise), `try` / `except` (several classes
in one clause, `as e`, `str(e)`, `type(e).__name__`) / `else` / `finally` (run on every way out:
the end, `return`, `break`, `continue`, an exception), your own exception classes derived from the
built-in ones, `assert`. The C++ exception classes have Python's names and family (`qt::ValueError`,
`qt::LookupError`, `qt::KeyError`, ...), and messages are the running Python's (Python 3.14
changed many).

```python spec translated "struct PricingError : qt::ValueError" "catch (const qt::ValueError& e)"
class PricingError(ValueError):
    pass

def f(x: float) -> float:
    try:
        if x < 0:
            raise PricingError("negative")
        return x
    except ValueError as e:
        return -1.0
```

Refused: a bare `except:` and `except BaseException` (they catch what C++ can't raise the same
way), `return` / `break` / `continue` in `finally`, catching an error raised in another module
of the project, assigning to the exception variable.

```python spec refused "bare `except:`"
def f(x: float) -> float:
    try:
        return 1.0 / x
    except:
        return 0.0
```

## 10. Arrays

`Vec` and `Mat` are NumPy float64 arrays; `IVec` an int64 array. Creation (`zeros`, `ones`,
`full`, `eye`, `linspace`, `arange`, `array`, `*_like`, `copy`), elementwise arithmetic and
functions with NumPy's broadcasting, reductions (`sum`, `mean`, `std`, `var`, `max`, `min`,
`argmax`, `argmin`, `cumsum`, `cumprod`, `quantile`, `percentile`, `median`, with `axis`,
`ddof`, `keepdims`), indexing and slicing (views, as NumPy's; negative indexes), masks
(`x[mask]`, `x[mask] = c`), shapes (`v[:, None]`, `ravel`, `stack`s), linear algebra (`@`, `dot`,
`outer`, `.T`, `einsum`, `np.linalg.solve / inv / cholesky / det / eigh / eigvalsh / pinv`).
NumPy's algorithms are ported where the result depends on them (pairwise summation, `var`,
`quantile`, `linspace`, `arange`, `cumsum`): bit-identical.

Arrays are shared in Python (`b = a`, slices, `.T`, function arguments). The analysis that
decides how parameters are passed (section 6) also refuses any two names for one array where one
is changed in place afterwards:

```python spec refused "share memory"
from qtrans.typing import Vec

def f(a: Vec) -> float:
    b = a[1:]
    b[0] = 5.0
    return float(a[1])
```

Int arrays (`IVec`): `np.arange` of ints, `np.zeros(n, dtype=int)`, `np.full(n, 3)`,
`np.array([1, 2, 3])`, `x.astype(int)`, `rng.integers(..., size=n)`. Their `+ - * **` and `sum`
wrap at 2⁶⁴ as NumPy's do (a negative exponent raises `ValueError`); `//` and `%` give 0 for a
zero divisor; `/`, a float operand and the float functions make float64; `x[idx]` gathers:

```python spec translated "qt::take(x, idx)" "rng.integers(0,"
import numpy as np
from qtrans.typing import Vec

def boot(x: Vec, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=len(x))
    return float(x[idx].mean())
```

```python spec program
import numpy as np

k = np.arange(-3, 4)
print(k, k // 2, k % 2, k ** 2, k * 0.5, k.sum(), repr(k[0]))
```

Refused: other dtypes, 3-D arrays, 2-D int arrays, `x[idx] = ...`, fancy indexing of a `Mat` by
two index arrays, NumPy's legacy global random functions (`np.random.normal(...)`), arrays in
`np.random.default_rng()` without a seed.

## 11. Random numbers

**NumPy's generator.** `np.random.default_rng(seed)` is `qt::Rng(seed)`: SeedSequence, PCG64 and
NumPy's conversions ported, so the same seed gives the same stream, bit for bit. `random`,
`standard_normal`, `normal`, `uniform` (scalar, `n` or `(r, c)` sizes) and `integers` (Lemire's
method, the half word NumPy keeps for 32-bit draws included). A generator passed to a function
is shared (drawing advances the caller's); `g2 = g` is refused (it would copy a stream Python
shares).

**Python's random module.** `random.seed(int)`, `random`, `getrandbits`, `randrange`, `randint`,
`uniform`, `triangular`, `gauss`, `normalvariate`, `lognormvariate`, `expovariate`, `choice`,
`shuffle`, `sample`, `choices` on `qt::py::random_module()`: CPython's Mersenne Twister, seeded as
CPython seeds it, and each function as `Lib/random.py` computes it. The defaults and messages are
the running Python's.

```python spec program
import random

random.seed(2026)
print(random.random(), random.randint(1, 6), random.gauss(0.0, 1.0))
xs = list(range(10))
random.shuffle(xs)
print(xs, random.choice(xs), random.sample(xs, 3))
```

A program must seed before it draws (Python seeds from the operating system at import, so the
numbers would change every run):

```python spec refused "before the program calls random.seed"
import random

print(random.random())
```

Refused: `random.Random(...)` objects, `betavariate`, `gammavariate`, `vonmisesvariate`,
`paretovariate`, `weibullvariate`, `binomialvariate`, `getstate` / `setstate`, seeds that aren't
ints.

## 12. Library functions

`math.*` (with Python's errors inside a guarding `try`), `math.fsum` (CPython's algorithm),
`math.gamma` / `lgamma` (CPython's own code, not the C library's), scalar NumPy functions,
`scipy.stats.norm.cdf / pdf / ppf / sf` (with `loc` / `scale`), `scipy.special.erf / erfc / ndtr /
ndtri`, the builtins `abs`, `min`, `max`, `float`, `int`, `bool`, `round`, `pow`, `len`, `sum`,
`sorted`, `any`, `all`, `isinstance` (objects of a hierarchy), `print`, `str`, `repr`, `format`.

```python spec translated "qt::py::fsum(xs)"
import math

def f(xs: list[float]) -> float:
    return math.fsum(xs)
```

Refused: `map`, `filter` (write a comprehension), `divmod`, `complex`, `eval`, `open` and files,
pandas.

## 13. Modules, projects and programs

A module becomes a header (`m.hpp`) with its functions in `namespace m`. Several modules that
import each other (`from m import f`, `import m as M`) are translated together, in dependency
order, with `#include`s between them. A file with code that runs (top-level statements, or an
`if __name__ == "__main__":` block) is a program: that code becomes `int main()`, an uncaught
exception ends it with exit status 1 and its message on stderr, `sys.exit(n)` with `n`.

Your own C++ for a function (an overrides file) replaces the translation, and `qtrans verify`
checks it against the Python like any other.

## 14. What `qtrans verify` checks

For every translated function: the hand-picked cases in `QTRANS_CASES` and 200 random ones
(arrays of mixed shapes, empty ones and ones that broadcast, objects of every concrete class, the
random module seeded the same way on both sides), each run by Python and by the compiled C++,
comparing the return value, every array or object the call changes, the state of every generator
it uses, and what it prints. Values travel as hex floats, so nothing is rounded on the way. Each
case is:

- **match**: within `rtol=1e-10`, `atol=1e-13` (arrays normwise), or the same exception class
  and message;
- **ill-cond**: further apart, but within 100× of how far Python's own answer moves when the
  inputs are nudged by 4 ULPs (the problem itself doesn't fix the answer more finely);
- **known**: Python raised a float error IEEE-754 doesn't raise, outside a guarding `try`, or a
  Python float went complex, or an int left 64 bits (section 15);
- **edge**: linear algebra on inf, NaN or near-overflow input, where BLAS / LAPACK and Eigen add
  in different orders;
- **mismatch**: anything else. A correct translation has none.

For a program, its output and exit status must be identical.

## 15. Differences kept by design

Each is named in the per-module report next to the functions it affects.

- **Functions marked `# qtrans: ieee`**: `x / 0.0`, overflow in `**` and `math.exp`,
  `math.log(-1.0)` give inf or NaN, where Python raises (you asked for it, section 4.2).
- **A value that may be a NumPy number or a Python float, depending on the run**: Python raises
  on some runs and gives inf / NaN on others; the C++ gives NumPy's inf or NaN (section 4.2).
- **A float that goes complex**: `(-8.0) ** (1/3)` is a complex number in Python and NaN in C++,
  so what follows differs (`round()` of it raises `TypeError` in Python, `ValueError` in C++; and
  `(-1e308) ** 2.5`, too big even as a complex number, raises `OverflowError` in Python).
- **ints past 64 bits** raise `OverflowError` in C++; Python's int grows. (NumPy's int64 arrays
  wrap, in both.)
- **The last bits** where NumPy or SciPy use their own `exp`, `pow`, `erf` or BLAS / LAPACK.
- **Which of two errors is raised** when two arguments of a library function (`max(a / b, c / d)`),
  two array or object arguments, or a number and an array operand (`math.log(c) * (A @ x)`)
  raise at once: C++ evaluates those in an unspecified order. Both reject the input; `qtrans
  verify` counts such a case as `known` (order) whenever both sides raise an error that rejects
  the input. (Operators on numbers and calls of the file's own functions follow Python's order,
  section 4.3.)
- **`assert`** always runs in C++; `python -O` strips it.
- **Objects whose fields are the same NaN**: Python's dataclass `==` treats one float object as
  equal to itself even when it is NaN; C++ compares values.
- **LAPACK on inf**: `eigh` / `pinv` results on inputs holding inf are not reproduced (LAPACK's
  own are erratic).
- **Optional arrays** (`P: Mat | None`) are read and rebound but not changed in place.
- **Bindings** (`qtrans bind`) leave out class hierarchies and functions that use the random
  module, saying so.

## 16. Versions

This document describes qtrans 1.0. From 1.0 on, a change to what it says is a change of major
version (2.0), except for additions: what 1.x refuses, a later 1.x may translate. It is checked on Python 3.10 to 3.14, NumPy 2, and Eigen 3.4
and 5.0, on Linux (GCC, Clang), macOS (Apple Clang) and Windows (MinGW g++, MSVC for the
bindings).
