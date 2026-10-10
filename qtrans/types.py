"""The type system of the stage-1 subset.

Python decides types at run time; C++ fixes them at compile time. Stage 1 only
admits the types quant pricing code actually uses for scalars:

    Python   C++
    bool     bool
    int      long long   (64-bit; Python ints are unbounded, see README)
    float    double      (both are IEEE-754 binary64, so arithmetic agrees bit for bit)
    tuple[...]  std::tuple<...>   (for functions returning several values, e.g. Greeks)
    Vec      qt::Vec     (Eigen::ArrayXd; 1-D float64 array)            stage 2
    Mat      qt::Mat     (row-major Eigen::ArrayXXd; 2-D float64 array) stage 2
    None     void        (return type only)
    X | None std::optional<X>  (X a number or array: optional arguments, "no result" returns)
    @dataclass / NamedTuple   a C++ struct with a constructor, methods and properties (stage 5)

Numeric types form a ladder bool < int < float. A variable that is assigned an int
in one place and a float in another is widened to float, exactly the way Python's
own arithmetic would promote the value.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scalar:
    name: str  # "bool", "int" or "float"

    def cpp(self) -> str:
        return {"bool": "bool", "int": "long long", "float": "double"}[self.name]

    def zero(self) -> str:
        return {"bool": "false", "int": "0", "float": "0.0"}[self.name]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TupleType:
    elts: tuple

    def cpp(self) -> str:
        return "std::tuple<" + ", ".join(e.cpp() for e in self.elts) + ">"

    def zero(self) -> str:
        return "{}"

    def __str__(self) -> str:
        return "tuple[" + ", ".join(str(e) for e in self.elts) + "]"


@dataclass(frozen=True)
class VoidType:
    def cpp(self) -> str:
        return "void"

    def __str__(self) -> str:
        return "None"


@dataclass(frozen=True)
class ArrayType:
    """A NumPy float64 array of fixed rank: Vec (1-D) or Mat (2-D)."""
    rank: int

    def cpp(self) -> str:
        return "qt::Vec" if self.rank == 1 else "qt::Mat"

    def zero(self) -> str:
        return "{}"

    def __str__(self) -> str:
        return "Vec" if self.rank == 1 else "Mat"


@dataclass(frozen=True)
class MaskType:
    """The bool array from an elementwise comparison (a > 0): used by np.where, .any(), .all(),
    np.sum / count, & | ~. A comparison always makes a new array, so storing one is safe."""
    rank: int

    def cpp(self) -> str:
        return "qt::Mask" if self.rank == 1 else "qt::Mask2"

    def zero(self) -> str:
        return "{}"

    def __str__(self) -> str:
        return "bool array"


@dataclass(frozen=True)
class RngType:
    """np.random.Generator (PCG64), reproduced exactly by qt::Rng."""

    def cpp(self) -> str:
        return "qt::Rng"

    def __str__(self) -> str:
        return "Generator"


@dataclass(frozen=True)
class StrType:
    """A Python str that is compared, passed and returned (as std::string). Python compares
    strings by code point and std::string by byte; for UTF-8 the two orders agree."""

    def cpp(self) -> str:
        return "std::string"

    def zero(self) -> str:
        return "{}"

    def __str__(self) -> str:
        return "str"


@dataclass(frozen=True)
class StructType:
    """An instance of a translated @dataclass or NamedTuple (stage 5): a C++ struct. Details
    (fields, methods, frozen) live in the module's ClassInfo, looked up by name."""
    name: str
    cpp_name: str

    def cpp(self) -> str:
        return self.cpp_name

    def zero(self) -> str:
        return "{}"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class NoneValueType:
    """The value None, before it is stored anywhere (it then becomes part of an OptionalType)."""

    def cpp(self) -> str:
        return "std::nullopt_t"

    def zero(self) -> str:
        return "std::nullopt"

    def __str__(self) -> str:
        return "None"


@dataclass(frozen=True)
class OptionalType:
    """`X | None`: a number or array that may be absent, as std::optional<X>. Using the value
    where Python needs a number checks it first (Python raises TypeError on None + 1)."""
    inner: object

    def cpp(self) -> str:
        return f"std::optional<{self.inner.cpp()}>"

    def zero(self) -> str:
        return "std::nullopt"

    def __str__(self) -> str:
        return f"{self.inner} | None"


BOOL = Scalar("bool")
INT = Scalar("int")
FLOAT = Scalar("float")
VOID = VoidType()
VEC = ArrayType(1)
MAT = ArrayType(2)
RNG = RngType()
NONE = NoneValueType()
STR = StrType()

_RANK = {"bool": 0, "int": 1, "float": 2}


def is_scalar(t) -> bool:
    return isinstance(t, Scalar)


def is_array(t) -> bool:
    return isinstance(t, ArrayType)


def join(a, b):
    """Smallest type that can hold values of both a and b, or None if there is none."""
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        return a
    if a == NONE or b == NONE or isinstance(a, OptionalType) or isinstance(b, OptionalType):
        # x = None in one place and x = 1.5 in another: an optional float
        ia = a.inner if isinstance(a, OptionalType) else a
        ib = b.inner if isinstance(b, OptionalType) else b
        inner = ib if ia == NONE else ia if ib == NONE else join(ia, ib)
        return OptionalType(inner) if isinstance(inner, (Scalar, ArrayType)) else None
    if isinstance(a, Scalar) and isinstance(b, Scalar):
        return a if _RANK[a.name] >= _RANK[b.name] else b
    if isinstance(a, TupleType) and isinstance(b, TupleType) and len(a.elts) == len(b.elts):
        elts = tuple(join(x, y) for x, y in zip(a.elts, b.elts))
        return None if None in elts else TupleType(elts)
    return None


def assignable(target, value) -> bool:
    """Can a value of type `value` be stored in a C++ variable of type `target`
    without changing the number Python would have computed?"""
    if target == value:
        return True
    if isinstance(target, OptionalType):
        if value == NONE:
            return True
        v = value.inner if isinstance(value, OptionalType) else value
        return assignable(target.inner, v)
    if isinstance(target, Scalar) and isinstance(value, Scalar):
        return _RANK[value.name] <= _RANK[target.name]
    if isinstance(target, (ArrayType, RngType)) or isinstance(value, (ArrayType, RngType)):
        return False
    if isinstance(target, TupleType) and isinstance(value, TupleType):
        return len(target.elts) == len(value.elts) and all(
            assignable(t, v) for t, v in zip(target.elts, value.elts)
        )
    return False
