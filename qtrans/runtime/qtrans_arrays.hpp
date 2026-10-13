// qtrans_arrays.hpp: array support for generated code (stage 2), built on Eigen.
//
// NumPy arrays become Eigen *Arrays*, not Matrices, because Eigen Arrays are elementwise
// by default, as NumPy is:   a * b  multiplies element by element in both.
// Matrix products are explicit (qt::matmul and friends), as `@` is in Python.
//
//   Vec = 1-D float64 array  -> Eigen::ArrayXd
//   Mat = 2-D float64 array  -> Eigen::Array<double, Dynamic, Dynamic, RowMajor>
//                               (row-major, so memory order and flat reductions match NumPy)
//
// Where NumPy's exact algorithm is known (pairwise summation, mean, var, quantile, linspace,
// arange, cumsum) it is ported line for line, so results are bit-identical. Where NumPy calls
// BLAS/LAPACK or its own SIMD maths (matmul, solve, exp, ...), results agree to rounding.
//
// Shape and index errors throw instead of crashing: NumPy raises ValueError / IndexError,
// and Eigen's internal assertions are redirected to exceptions below. Include this header
// before any other Eigen header so the redirection takes effect.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "qtrans_runtime.hpp"

namespace qt {
struct shape_error : std::invalid_argument {   // NumPy: ValueError (shapes), IndexError (bounds)
    using std::invalid_argument::invalid_argument;
};
using index_error = IndexError;  // NumPy's IndexError
QT_USER_EXCEPTION(LinAlgError, ValueError)  // numpy.linalg.LinAlgError, a ValueError
using linalg_error = LinAlgError;
}  // namespace qt

#ifndef eigen_assert
#define eigen_assert(x) \
    do { if (!(x)) throw ::qt::shape_error("array shape or index mismatch: " #x); } while (false)
#endif
#include <Eigen/Dense>

namespace qt {

using Vec = Eigen::ArrayXd;
using Mat = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Mask = Eigen::Array<bool, Eigen::Dynamic, 1>;  // a stored comparison: tail = pnl <= -var
using Mask2 = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IVec = Eigen::Array<long long, Eigen::Dynamic, 1>;  // a 1-D int64 array (0.14)
using Index = Eigen::Index;
inline constexpr std::nullopt_t none = std::nullopt;

// The plain type (Vec or Mat) an array expression evaluates to.
template <class D>
using plain_t = std::conditional_t<D::ColsAtCompileTime == 1, Vec, Mat>;

// =====================================================================================
// Construction
// =====================================================================================
inline void check_dim(long long n) {
    if (n < 0) throw shape_error("negative dimensions are not allowed");
}

inline Vec vec(std::initializer_list<double> values) {  // np.array([a, b, c])
    Vec out(static_cast<Index>(values.size()));
    Index i = 0;
    for (double v : values) out(i++) = v;
    return out;
}

inline Mat mat(std::initializer_list<std::initializer_list<double>> rows) {  // np.array([[..], [..]])
    const Index r = static_cast<Index>(rows.size());
    const Index c = r ? static_cast<Index>(rows.begin()->size()) : 0;
    Mat out(r, c);
    Index i = 0;
    for (const auto& row : rows) {
        if (static_cast<Index>(row.size()) != c) throw shape_error("inhomogeneous rows in np.array");
        Index j = 0;
        for (double v : row) out(i, j++) = v;
        ++i;
    }
    return out;
}

inline Vec zeros(long long n) { check_dim(n); return Vec::Zero(n); }
inline Mat zeros(long long r, long long c) { check_dim(r); check_dim(c); return Mat::Zero(r, c); }
inline Vec ones(long long n) { check_dim(n); return Vec::Ones(n); }
inline Mat ones(long long r, long long c) { check_dim(r); check_dim(c); return Mat::Ones(r, c); }
inline Vec full(long long n, double v) { check_dim(n); return Vec::Constant(n, v); }
inline Mat full(long long r, long long c, double v) { check_dim(r); check_dim(c); return Mat::Constant(r, c, v); }

inline Mat eye(long long n) {
    Mat out = zeros(n, n);
    for (Index i = 0; i < n; ++i) out(i, i) = 1.0;
    return out;
}

// np.linspace(start, stop, num): y_i = i * step + start, last element set to stop exactly.
inline Vec linspace(double start, double stop, long long num = 50) {
    if (num < 0) throw shape_error("Number of samples must be non-negative");
    Vec y(num);
    const long long div = num - 1;
    const double delta = stop - start;
    if (div > 0) {
        const double step = delta / static_cast<double>(div);
        for (Index i = 0; i < num; ++i) {
            y(i) = step == 0.0 ? static_cast<double>(i) / static_cast<double>(div) * delta + start
                               : static_cast<double>(i) * step + start;
        }
        y(num - 1) = stop;
    } else if (num == 1) {
        y(0) = 0.0 * delta + start;
    }
    return y;
}

// np.arange(start, stop, step): length ceil((stop - start) / step); NumPy fills element i
// as start + i * (second - first), where second = start + step.
inline Vec arange(double start, double stop, double step = 1.0) {
    if (step == 0.0) throw ZeroDivisionError("division by zero");  // NumPy's own message
    const double len = std::ceil((stop - start) / step);
    if (std::isnan(len)) throw shape_error("arange: cannot compute length");
    const Index n = len > 0 ? static_cast<Index>(len) : 0;
    Vec y(n);
    if (n > 0) y(0) = start;
    if (n > 1) {
        y(1) = start + step;
        const double delta = y(1) - y(0);
        for (Index i = 2; i < n; ++i) y(i) = start + static_cast<double>(i) * delta;
    }
    return y;
}
inline Vec arange(double stop) { return arange(0.0, stop, 1.0); }

// =====================================================================================
// Indexing and slicing with Python's rules
// =====================================================================================
// a[i]: negative i counts from the end; out of range raises IndexError.
inline Index idx(long long i, Index n) {
    const long long j = i < 0 ? i + n : i;
    if (j < 0 || j >= n) throw index_error("index out of range");
    return static_cast<Index>(j);
}

// A[i] and A[:, j]: Python's index (negative counts from the end), checked before Eigen
// forms the row or column view. Eigen computes the view's address before its own bounds check,
// which is undefined behaviour out of range (the last column of an n x 0 matrix, say).
template <class D> auto row(Eigen::DenseBase<D>& a, long long i) { return a.row(idx(i, a.rows())); }
template <class D> auto row(const Eigen::DenseBase<D>& a, long long i) { return a.row(idx(i, a.rows())); }
template <class D> auto col(Eigen::DenseBase<D>& a, long long j) { return a.col(idx(j, a.cols())); }
template <class D> auto col(const Eigen::DenseBase<D>& a, long long j) { return a.col(idx(j, a.cols())); }

// a[start:stop:step]: CPython's PySlice_Unpack + PySlice_AdjustIndices, so negative and
// out-of-range bounds are clipped exactly as Python clips them (a[1:100] is fine).
struct Span { Index first, size, step; };

inline Span span(Index n, std::optional<long long> start, std::optional<long long> stop, long long step) {
    if (step == 0) throw shape_error("slice step cannot be zero");
    const long long big = std::numeric_limits<long long>::max();
    const long long small = std::numeric_limits<long long>::min();
    long long lo = start ? *start : (step < 0 ? big : 0);
    long long hi = stop ? *stop : (step < 0 ? small : big);
    if (lo < 0) { lo += n; if (lo < 0) lo = step < 0 ? -1 : 0; }
    else if (lo >= n) lo = step < 0 ? n - 1 : n;
    if (hi < 0) { hi += n; if (hi < 0) hi = step < 0 ? -1 : 0; }
    else if (hi >= n) hi = step < 0 ? n - 1 : n;
    long long len = 0;
    if (step < 0) { if (hi < lo) len = (lo - hi - 1) / (-step) + 1; }
    else if (lo < hi) len = (hi - lo - 1) / step + 1;
    return {static_cast<Index>(len ? lo : 0), static_cast<Index>(len), static_cast<Index>(step)};
}

// A view into a (readable and assignable), like a NumPy slice: a strided Map over a's own
// memory (a negative step is a negative stride), so Eigen can read it in place anywhere,
// products included. (Eigen's own a(seqN(...)) view claims direct access but has no .data().)
template <class A>
auto slice(A& a, std::optional<long long> start, std::optional<long long> stop, long long step = 1) {
    using P = typename std::remove_const_t<A>::PlainObject;
    using View = std::conditional_t<std::is_const_v<A>, Eigen::Map<const P, 0, Eigen::InnerStride<>>,
                                    Eigen::Map<P, 0, Eigen::InnerStride<>>>;
    const Span s = span(a.size(), start, stop, step);
    const Index inner = a.innerStride();
    return View(a.data() + s.first * inner, s.size, Eigen::InnerStride<>(s.step * inner));
}

// A[r0:r1, c0:c1]: a 2-D view.
template <class A>
auto block(A& m, std::optional<long long> r0, std::optional<long long> r1,
           std::optional<long long> c0, std::optional<long long> c1) {
    const Span r = span(m.rows(), r0, r1, 1);
    const Span c = span(m.cols(), c0, c1, 1);
    return m(Eigen::seqN(r.first, r.size), Eigen::seqN(c.first, c.size));
}

// =====================================================================================
// Elementwise functions. These call the C library, as Python's math module and stage-1
// code do. NumPy uses its own SIMD versions of exp/log/pow, which can differ in the last bit.
// (sqrt, abs and + - * / are exact in both, so they are left to Eigen.)
// =====================================================================================
#define QT_UNARY(NAME, FN)                                                         \
    template <class D>                                                             \
    auto NAME(const Eigen::ArrayBase<D>& a) {                                      \
        return a.unaryExpr([](double x) { return FN(static_cast<double>(x)); });   \
    }
QT_UNARY(exp, std::exp)
QT_UNARY(log, std::log)
QT_UNARY(log10, std::log10)
QT_UNARY(log2, std::log2)
QT_UNARY(log1p, std::log1p)
QT_UNARY(expm1, std::expm1)
QT_UNARY(sin, std::sin)
QT_UNARY(cos, std::cos)
QT_UNARY(tan, std::tan)
QT_UNARY(sinh, std::sinh)
QT_UNARY(cosh, std::cosh)
QT_UNARY(tanh, std::tanh)
QT_UNARY(arcsin, std::asin)
QT_UNARY(arccos, std::acos)
QT_UNARY(arctan, std::atan)
QT_UNARY(floor, std::floor)
QT_UNARY(ceil, std::ceil)
QT_UNARY(trunc, std::trunc)
QT_UNARY(rint, std::nearbyint)
QT_UNARY(norm_cdf, ::qt::norm_cdf)
QT_UNARY(norm_sf, ::qt::norm_sf)
QT_UNARY(norm_pdf, ::qt::norm_pdf)
QT_UNARY(norm_ppf, ::qt::norm_ppf)
QT_UNARY(np_sign, ::qt::np_sign)
#undef QT_UNARY

template <class D>
auto norm_cdf(const Eigen::ArrayBase<D>& a, double loc, double scale) {
    return a.unaryExpr([loc, scale](double x) { return ::qt::norm_cdf(x, loc, scale); });
}
template <class D>
auto norm_pdf(const Eigen::ArrayBase<D>& a, double loc, double scale) {
    return a.unaryExpr([loc, scale](double x) { return ::qt::norm_pdf(x, loc, scale); });
}

// =====================================================================================
// NumPy broadcasting. Two shapes combine when each dimension is equal or 1 in one of them;
// the 1 is stretched (a length-1 Vec against a length-n one, a (1, m) row against an (n, m)
// matrix, ...). The translator writes plain Eigen (`a - b`) where it can show the sizes are
// equal, and these helpers otherwise. Equal sizes take Eigen's own operation, which rounds
// exactly as the scalar one does, so the results are the same bits either way.
// =====================================================================================
struct add_op { double operator()(double x, double y) const { return x + y; } };
struct sub_op { double operator()(double x, double y) const { return x - y; } };
struct mul_op { double operator()(double x, double y) const { return x * y; } };
struct div_op { double operator()(double x, double y) const { return x / y; } };
struct pow_op { double operator()(double x, double y) const { return std::pow(x, y); } };
struct lt_op { bool operator()(double x, double y) const { return x < y; } };
struct le_op { bool operator()(double x, double y) const { return x <= y; } };
struct gt_op { bool operator()(double x, double y) const { return x > y; } };
struct ge_op { bool operator()(double x, double y) const { return x >= y; } };
struct eq_op { bool operator()(double x, double y) const { return x == y; } };
struct ne_op { bool operator()(double x, double y) const { return x != y; } };
struct and_op { bool operator()(bool x, bool y) const { return x && y; } };
struct or_op { bool operator()(bool x, bool y) const { return x || y; } };

inline Index broadcast_dim(Index a, Index b) {
    if (a == b || b == 1) return a;
    if (a == 1) return b;
    throw shape_error("operands could not be broadcast together");
}

template <class F, class D1, class D2>
auto broadcast(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b, F f) {
    using S = decltype(f(std::declval<typename D1::Scalar>(), std::declval<typename D2::Scalar>()));
    if constexpr (D1::ColsAtCompileTime == 1 && D2::ColsAtCompileTime == 1) {
        // two 1-D arrays
        using Out = Eigen::Array<S, Eigen::Dynamic, 1>;
        if (a.size() == b.size()) {
            if constexpr (std::is_same_v<F, add_op>) return Out(a + b);
            else if constexpr (std::is_same_v<F, sub_op>) return Out(a - b);
            else if constexpr (std::is_same_v<F, mul_op>) return Out(a * b);
            else if constexpr (std::is_same_v<F, div_op>) return Out(a / b);
            else return Out(a.binaryExpr(b, f));
        }
        if (b.size() == 1) {
            const auto y = b(0);
            return Out(a.unaryExpr([&](typename D1::Scalar x) { return f(x, y); }));
        }
        if (a.size() == 1) {
            const auto x = a(0);
            return Out(b.unaryExpr([&](typename D2::Scalar y) { return f(x, y); }));
        }
        throw shape_error("operands could not be broadcast together");
    } else {
        // 2-D (a Vec on either side has been turned into a (1, n) row by the caller)
        using Out = Eigen::Array<S, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        if (a.rows() == b.rows() && a.cols() == b.cols()) {
            if constexpr (std::is_same_v<F, add_op>) return Out(a + b);
            else if constexpr (std::is_same_v<F, sub_op>) return Out(a - b);
            else if constexpr (std::is_same_v<F, mul_op>) return Out(a * b);
            else if constexpr (std::is_same_v<F, div_op>) return Out(a / b);
            else return Out(a.binaryExpr(b, f));
        }
        using P1 = Eigen::Array<typename D1::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using P2 = Eigen::Array<typename D2::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        const P1 x = a;
        const P2 y = b;
        const Index r = broadcast_dim(x.rows(), y.rows()), c = broadcast_dim(x.cols(), y.cols());
        Out out(r, c);
        for (Index i = 0; i < r; ++i)
            for (Index j = 0; j < c; ++j)
                out(i, j) = f(x(x.rows() == 1 ? 0 : i, x.cols() == 1 ? 0 : j),
                              y(y.rows() == 1 ? 0 : i, y.cols() == 1 ? 0 : j));
        return out;
    }
}

// A op= B: B is broadcast to A's shape; A's shape never changes (NumPy raises
// "non-broadcastable output operand" otherwise).
// (The caller evaluates b first when it reads a itself, as in S += S.T.)
template <class F, class D>
void broadcast_into(Vec& a, const Eigen::ArrayBase<D>& b, F f) {
    static_assert(D::ColsAtCompileTime == 1, "a 1-D array gets a 1-D right side");
    if (b.size() == a.size()) {
        if constexpr (std::is_same_v<F, add_op>) a += b;
        else if constexpr (std::is_same_v<F, sub_op>) a -= b;
        else if constexpr (std::is_same_v<F, mul_op>) a *= b;
        else if constexpr (std::is_same_v<F, div_op>) a /= b;
        else a = a.binaryExpr(b, f);
        return;
    }
    if (b.size() != 1)
        throw shape_error(a.size() == 1 ? "non-broadcastable output operand"
                                        : "operands could not be broadcast together");
    const double y = b(0);
    for (Index i = 0; i < a.size(); ++i) a(i) = f(a(i), y);
}

template <class F, class D>
void broadcast_into(Mat& a, const Eigen::ArrayBase<D>& b, F f) {
    if (b.rows() == a.rows() && b.cols() == a.cols()) {
        if constexpr (std::is_same_v<F, add_op>) a += b;
        else if constexpr (std::is_same_v<F, sub_op>) a -= b;
        else if constexpr (std::is_same_v<F, mul_op>) a *= b;
        else if constexpr (std::is_same_v<F, div_op>) a /= b;
        else a = a.binaryExpr(b, f);
        return;
    }
    const Mat y = b;
    if ((y.rows() != a.rows() && y.rows() != 1) || (y.cols() != a.cols() && y.cols() != 1))
        throw shape_error("non-broadcastable output operand");
    for (Index i = 0; i < a.rows(); ++i)
        for (Index j = 0; j < a.cols(); ++j)
            a(i, j) = f(a(i, j), y(y.rows() == 1 ? 0 : i, y.cols() == 1 ? 0 : j));
}

// v[i:j] = w, A[i] = w, A[:, j] = w, A[r0:r1, c0:c1] = B (and +=, ...): the right side is
// broadcast to the part's shape, which never changes. dst is an Eigen block (a view).
struct set_op { double operator()(double, double y) const { return y; } };
template <class Dst, class Src, class F = set_op>
void store(Dst&& dst, const Eigen::ArrayBase<Src>& src, F f = {}) {
    if (src.rows() == dst.rows() && src.cols() == dst.cols()) {
        if constexpr (std::is_same_v<F, set_op>) dst = src;
        else if constexpr (std::is_same_v<F, add_op>) dst += src;
        else if constexpr (std::is_same_v<F, sub_op>) dst -= src;
        else if constexpr (std::is_same_v<F, mul_op>) dst *= src;
        else if constexpr (std::is_same_v<F, div_op>) dst /= src;
        else dst = dst.binaryExpr(src, f);
        return;
    }
    if (src.size() == 1) {  // 1 element (any shape of 1s) fills the part
        const double y = src(0, 0);
        for (Index i = 0; i < dst.rows(); ++i)
            for (Index j = 0; j < dst.cols(); ++j) dst(i, j) = f(dst(i, j), y);
        return;
    }
    // a (1, m) row into each row of an (n, m) part, an (n, 1) column into each column
    const Mat y = src;
    const bool rows_ok = y.rows() == dst.rows() || y.rows() == 1, cols_ok = y.cols() == dst.cols() || y.cols() == 1;
    if (!rows_ok || !cols_ok) {
        // a 1-D part (a row or column view) against a 1-D right side of another length
        throw shape_error("could not broadcast input array into the shape of the destination");
    }
    for (Index i = 0; i < dst.rows(); ++i)
        for (Index j = 0; j < dst.cols(); ++j)
            dst(i, j) = f(dst(i, j), y(y.rows() == 1 ? 0 : i, y.cols() == 1 ? 0 : j));
}

template <class D>
auto pow(const Eigen::ArrayBase<D>& a, double p) {
    return a.unaryExpr([p](double x) { return std::pow(x, p); });
}
template <class D>
auto pow(double b, const Eigen::ArrayBase<D>& a) {
    return a.unaryExpr([b](double x) { return std::pow(b, x); });
}
template <class D1, class D2>
auto pow(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return broadcast(a, b, pow_op{});
}

// np.maximum / np.minimum propagate NaN from either side (Eigen's max/min don't).
struct max_op { double operator()(double x, double y) const { return np_maximum(x, y); } };
struct min_op { double operator()(double x, double y) const { return np_minimum(x, y); } };
template <class D1, class D2>
auto maximum(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return broadcast(a, b, max_op{});
}
template <class D>
auto maximum(const Eigen::ArrayBase<D>& a, double b) {
    return a.unaryExpr([b](double x) { return np_maximum(x, b); });
}
template <class D>
auto maximum(double a, const Eigen::ArrayBase<D>& b) {  // not maximum(b, a): ties return the 2nd
    return b.unaryExpr([a](double y) { return np_maximum(a, y); });
}
template <class D1, class D2>
auto minimum(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return broadcast(a, b, min_op{});
}
template <class D>
auto minimum(const Eigen::ArrayBase<D>& a, double b) {
    return a.unaryExpr([b](double x) { return np_minimum(x, b); });
}
template <class D>
auto minimum(double a, const Eigen::ArrayBase<D>& b) {
    return b.unaryExpr([a](double y) { return np_minimum(a, y); });
}

// np.clip(a, lo, hi) with scalar bounds (one-sided clips are np.maximum / np.minimum).
template <class D>
auto clip(const Eigen::ArrayBase<D>& a, double lo, double hi) {
    return a.unaryExpr([lo, hi](double x) { return np_clip(x, lo, hi); });
}

// np.where(cond, x, y) with arrays or scalars on either branch, broadcast together.
inline double at(double v, Index, Index) { return v; }
template <class D>
auto at(const Eigen::ArrayBase<D>& a, Index i, Index j) {
    return a(a.rows() == 1 ? 0 : i, a.cols() == 1 ? 0 : j);
}
inline Index rows_of(double) { return 1; }
inline Index cols_of(double) { return 1; }
template <class D> Index rows_of(const Eigen::ArrayBase<D>& a) { return a.rows(); }
template <class D> Index cols_of(const Eigen::ArrayBase<D>& a) { return a.cols(); }

template <class C, class X, class Y>
auto where(const Eigen::ArrayBase<C>& cond, const X& x, const Y& y) {
    const Index r = broadcast_dim(broadcast_dim(cond.rows(), rows_of(x)), rows_of(y));
    const Index c = broadcast_dim(broadcast_dim(cond.cols(), cols_of(x)), cols_of(y));
    plain_t<C> out(r, c);
    for (Index i = 0; i < r; ++i)
        for (Index j = 0; j < c; ++j)
            out(i, j) = at(cond, i, j) ? at(x, i, j) : at(y, i, j);
    return out;
}

// =====================================================================================
// Reductions: NumPy's algorithms, ported so results are bit-identical
// =====================================================================================
// NumPy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src, PW_BLOCKSIZE 128):
// fewer than 8 values are added left to right; up to 128 use eight running partial sums;
// longer runs are split in two, recursively.
inline double pairwise(const double* a, Index n, Index stride) {
    if (n < 8) {
        double res = 0.0;
        for (Index i = 0; i < n; ++i) res += a[i * stride];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k * stride];
        Index i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[(i + k) * stride];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i * stride];
        return res;
    }
    Index n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2, stride) + pairwise(a + n2 * stride, n - n2, stride);
}

// A stored Vec / Mat is read where it is; an expression is evaluated once, in row-major order.
template <class D>
decltype(auto) evaluated(const Eigen::ArrayBase<D>& a) {
    if constexpr (std::is_same_v<D, Vec> || std::is_same_v<D, Mat>) return (a.derived());
    else return plain_t<D>(a);
}

// np.sum(a): pairwise over all elements in row-major (C) order.
template <class D>
double sum(const Eigen::ArrayBase<D>& a) {
    const auto& p = evaluated(a);
    return pairwise(p.data(), p.size(), 1);
}

// np.sum(A, axis=0): NumPy adds rows one after another (column-wise sequential), except for
// a single-column matrix, which it treats as contiguous and sums pairwise.
// np.sum(A, axis=1): pairwise along each row.
template <class D>
Vec sum(const Eigen::ArrayBase<D>& a, int axis) {
    const auto& m = evaluated(a);
    if (axis == 0 || axis == -2) {
        Vec out(m.cols());
        if (m.cols() == 1) {
            out(0) = pairwise(m.data(), m.rows(), 1);
        } else {
            out = Vec::Zero(m.cols());
            for (Index i = 0; i < m.rows(); ++i) out += m.row(i).transpose();
        }
        return out;
    }
    if (axis == 1 || axis == -1) {
        Vec out(m.rows());
        for (Index i = 0; i < m.rows(); ++i) out(i) = pairwise(m.data() + i * m.cols(), m.cols(), 1);
        return out;
    }
    throw shape_error("axis out of bounds for a 2-D array");
}

template <class D>
double mean(const Eigen::ArrayBase<D>& a) {
    return sum(a) / static_cast<double>(a.size());
}

template <class D>
Vec mean(const Eigen::ArrayBase<D>& a, int axis) {
    const Index count = (axis == 0 || axis == -2) ? a.rows() : a.cols();
    return sum(a, axis) / static_cast<double>(count);
}

// np.var(a, ddof): mean, squared deviations, pairwise sum, divide by max(n - ddof, 0).
template <class D>
double var(const Eigen::ArrayBase<D>& a, long long ddof = 0) {
    const plain_t<D> p = a;
    const double m = mean(p);
    const plain_t<D> dev = (p - m) * (p - m);
    const long long dof = std::max<long long>(static_cast<long long>(p.size()) - ddof, 0);
    return sum(dev) / static_cast<double>(dof);
}

template <class D>
double std_(const Eigen::ArrayBase<D>& a, long long ddof = 0) {
    return std::sqrt(var(a, ddof));
}

// np.var(A, axis, ddof): NumPy's _var with an axis. The mean is sum(A, axis) / count (the
// same reduction order as np.sum over that axis), then the squared deviations are summed
// over the same axis and divided by max(count - ddof, 0).
template <class D>
Vec var(const Eigen::ArrayBase<D>& a, int axis, long long ddof) {
    const Mat m = a;
    const bool down = axis == 0 || axis == -2;
    if (!down && axis != 1 && axis != -1) throw shape_error("axis out of bounds for a 2-D array");
    const Vec mu = mean(m, axis);
    Mat dev = down ? Mat(m.rowwise() - mu.transpose()) : Mat(m.colwise() - mu);
    dev = dev * dev;
    const long long count = static_cast<long long>(down ? m.rows() : m.cols());
    const long long dof = std::max<long long>(count - ddof, 0);
    return sum(dev, axis) / static_cast<double>(dof);
}

template <class D>
Vec std_(const Eigen::ArrayBase<D>& a, int axis, long long ddof) {
    return var(a, axis, ddof).sqrt();
}

template <class D>
Vec cumsum(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    Vec out(p.size());
    double s = 0.0;
    for (Index i = 0; i < p.size(); ++i) out(i) = s += p.data()[i];
    return out;
}

template <class D>
Vec cumprod(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    Vec out(p.size());
    double s = 1.0;
    for (Index i = 0; i < p.size(); ++i) out(i) = s *= p.data()[i];
    return out;
}

// np.cumsum(A, axis) / np.cumprod(A, axis): running totals down each column (axis 0) or
// along each row (axis 1), one addition per step, as NumPy's accumulate does.
template <bool Prod, class D>
Mat accumulate(const Eigen::ArrayBase<D>& a, int axis) {
    Mat m = a;
    if (axis == 0 || axis == -2) {
        for (Index i = 1; i < m.rows(); ++i)
            for (Index j = 0; j < m.cols(); ++j)
                m(i, j) = Prod ? m(i - 1, j) * m(i, j) : m(i - 1, j) + m(i, j);
    } else if (axis == 1 || axis == -1) {
        for (Index i = 0; i < m.rows(); ++i)
            for (Index j = 1; j < m.cols(); ++j)
                m(i, j) = Prod ? m(i, j - 1) * m(i, j) : m(i, j - 1) + m(i, j);
    } else {
        throw shape_error("axis out of bounds for a 2-D array");
    }
    return m;
}
template <class D>
Mat cumsum(const Eigen::ArrayBase<D>& a, int axis) { return accumulate<false>(a, axis); }
template <class D>
Mat cumprod(const Eigen::ArrayBase<D>& a, int axis) { return accumulate<true>(a, axis); }

// np.max / np.min: NaN anywhere gives NaN; an empty array raises.
template <class D>
double max(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    if (p.size() == 0) throw shape_error("zero-size array to reduction operation maximum which has no identity");
    double m = p.data()[0];
    for (Index i = 0; i < p.size(); ++i) {
        const double x = p.data()[i];
        if (std::isnan(x)) return x;
        if (x > m) m = x;
    }
    return m;
}

template <class D>
double min(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    if (p.size() == 0) throw shape_error("zero-size array to reduction operation minimum which has no identity");
    double m = p.data()[0];
    for (Index i = 0; i < p.size(); ++i) {
        const double x = p.data()[i];
        if (std::isnan(x)) return x;
        if (x < m) m = x;
    }
    return m;
}

// np.argmax / np.argmin: first occurrence; the first NaN wins if there is one.
template <class D>
long long argmax(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    if (p.size() == 0) throw shape_error("attempt to get argmax of an empty sequence");
    Index best = 0;
    for (Index i = 0; i < p.size(); ++i) {
        const double x = p.data()[i];
        if (std::isnan(x)) return i;
        if (x > p.data()[best]) best = i;
    }
    return best;
}

template <class D>
long long argmin(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    if (p.size() == 0) throw shape_error("attempt to get argmin of an empty sequence");
    Index best = 0;
    for (Index i = 0; i < p.size(); ++i) {
        const double x = p.data()[i];
        if (std::isnan(x)) return i;
        if (x < p.data()[best]) best = i;
    }
    return best;
}

// np.sort: ascending, NaN last.
template <class D>
Vec sort(const Eigen::ArrayBase<D>& a) {
    Vec out = a;
    std::sort(out.data(), out.data() + out.size(), [](double x, double y) {
        return x < y || (std::isnan(y) && !std::isnan(x));
    });
    return out;
}

// The k-th smallest value and the one after it, in linear time (std::nth_element), as
// NumPy's partition does: the same order statistics a full sort would give.
inline void order_stats(Vec& v, Index k, double& kth, double& next) {
    double* d = v.data();
    std::nth_element(d, d + k, d + v.size());
    kth = d[k];
    next = k + 1 < v.size() ? *std::min_element(d + k + 1, d + v.size()) : kth;
}

template <class D>
bool has_nan(const Eigen::ArrayBase<D>& a) {
    return a.isNaN().any();
}

// np.quantile(a, q), default 'linear' method: index (n - 1) * q, then NumPy's _lerp,
// which interpolates from whichever end is nearer.
template <class D>
double quantile(const Eigen::ArrayBase<D>& a, double q) {
    if (!(q >= 0.0 && q <= 1.0)) throw ValueError("Quantiles must be in the range [0, 1]");
    Vec v = a;
    const Index n = v.size();
    if (n == 0) throw index_error("index -1 is out of bounds for axis 0 with size 0");
    if (has_nan(v)) return nan;
    const double vi = static_cast<double>(n - 1) * q;
    const bool top = vi >= static_cast<double>(n - 1);
    const Index lo = top ? n - 1 : static_cast<Index>(std::floor(vi));
    double x, y;
    order_stats(v, lo, x, y);
    if (top) y = x;
    const double g = vi - std::floor(vi);
    const double d = y - x;
    return g >= 0.5 ? y - d * (1.0 - g) : x + d * g;
}

template <class D>
double percentile(const Eigen::ArrayBase<D>& a, double p) { return quantile(a, p / 100.0); }

template <class D>
double median(const Eigen::ArrayBase<D>& a) {
    Vec v = a;
    const Index n = v.size();
    if (n == 0 || has_nan(v)) return nan;
    double x, y;
    if (n % 2) {
        order_stats(v, n / 2, x, y);
        return x;
    }
    order_stats(v, n / 2 - 1, x, y);
    return (0.0 + x + y) / 2.0;  // NumPy: mean of the two middle values
}

// =====================================================================================
// Products. NumPy hands these to BLAS; results agree with Eigen's to rounding.
// =====================================================================================
// A product operand Eigen can read in place (a stored array, a contiguous block) is used where
// it is; anything else is evaluated first. Eigen 3.4 can't multiply a strided view
// (qt::slice with a step, v[::2]) directly: its product kernels ask for the view's memory.
// (Eigen 3.4 flags such a view as directly accessible yet gives it no .data(), so it is
// recognised by type as well.)
template <class D> struct is_indexed_view : std::false_type {};
template <class X, class R, class C> struct is_indexed_view<Eigen::IndexedView<X, R, C>> : std::true_type {};

template <class D>
decltype(auto) operand(const Eigen::ArrayBase<D>& a) {
    if constexpr (bool(Eigen::internal::traits<D>::Flags & Eigen::DirectAccessBit) && !is_indexed_view<D>::value)
        return a.derived();
    else
        return plain_t<D>(a);
}

template <class D1, class D2>
double dot(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {  // Vec @ Vec
    if (a.size() != b.size()) throw shape_error("matmul: mismatch in core dimension");
    return operand(a).matrix().dot(operand(b).matrix());
}

template <class D1, class D2>
Vec matvec(const Eigen::ArrayBase<D1>& m, const Eigen::ArrayBase<D2>& v) {  // Mat @ Vec
    if (m.cols() != v.size()) throw shape_error("matmul: mismatch in core dimension");
    return (operand(m).matrix() * operand(v).matrix()).array();
}

template <class D1, class D2>
Vec vecmat(const Eigen::ArrayBase<D1>& v, const Eigen::ArrayBase<D2>& m) {  // Vec @ Mat
    if (v.size() != m.rows()) throw shape_error("matmul: mismatch in core dimension");
    return (operand(v).matrix().transpose() * operand(m).matrix()).transpose().array();
}

template <class D1, class D2>
Mat matmul(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {  // Mat @ Mat
    if (a.cols() != b.rows()) throw shape_error("matmul: mismatch in core dimension");
    return (operand(a).matrix() * operand(b).matrix()).array();
}

template <class D1, class D2>
Mat outer(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return (operand(a).matrix() * operand(b).matrix().transpose()).array();
}

template <class D>
Mat diag(const Eigen::ArrayBase<D>& v) {  // np.diag(vector)
    Mat out = Mat::Zero(v.size(), v.size());
    for (Index i = 0; i < v.size(); ++i) out(i, i) = v(i);
    return out;
}

template <class D>
Vec diagonal(const Eigen::ArrayBase<D>& m) {  // np.diag(matrix)
    return m.matrix().diagonal().array();
}

// np.trace(A): NumPy sums the diagonal (a strided view) with the same pairwise summation.
template <class D>
double trace(const Eigen::ArrayBase<D>& a) {
    const Mat m = a;
    return pairwise(m.data(), std::min(m.rows(), m.cols()), m.cols() + 1);
}

// =====================================================================================
// Shapes: new axes, keepdims, ravel, stacking, broadcasting between two 2-D arrays
// =====================================================================================
template <class D>
Mat as_row(const Eigen::ArrayBase<D>& v) {  // v[None, :], or a keepdims=True reduction over axis 0
    const Vec p = v;
    return Mat(p.transpose());
}
template <class D>
Mat as_col(const Eigen::ArrayBase<D>& v) {  // v[:, None], or a keepdims=True reduction over axis 1
    const Vec p = v;
    Mat out(p.size(), 1);
    for (Index i = 0; i < p.size(); ++i) out(i, 0) = p(i);
    return out;
}

// a.ravel() / a.flatten(): row-major (C) order, as NumPy's default.
template <class D>
Vec ravel(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    return Eigen::Map<const Vec>(p.data(), p.size());
}

// np.concatenate / np.hstack of 1-D arrays.
inline Vec concat(std::initializer_list<Vec> parts) {
    Index n = 0;
    for (const Vec& v : parts) n += v.size();
    Vec out(n);
    Index k = 0;
    for (const Vec& v : parts) { out.segment(k, v.size()) = v; k += v.size(); }
    return out;
}

// Stack 2-D blocks on top of each other (np.vstack, np.concatenate(axis=0)) or side by side
// (np.hstack / np.column_stack of 2-D blocks, np.concatenate(axis=1)).
inline Mat vcat(std::initializer_list<Mat> parts) {
    if (parts.size() == 0) throw shape_error("need at least one array to concatenate");
    const Index c = parts.begin()->cols();
    Index r = 0;
    for (const Mat& m : parts) {
        if (m.cols() != c) throw shape_error("all the input array dimensions except for the "
                                             "concatenation axis must match exactly");
        r += m.rows();
    }
    Mat out(r, c);
    Index k = 0;
    for (const Mat& m : parts) { out.middleRows(k, m.rows()) = m; k += m.rows(); }
    return out;
}
inline Mat hcat(std::initializer_list<Mat> parts) {
    if (parts.size() == 0) throw shape_error("need at least one array to concatenate");
    const Index r = parts.begin()->rows();
    Index c = 0;
    for (const Mat& m : parts) {
        if (m.rows() != r) throw shape_error("all the input array dimensions except for the "
                                             "concatenation axis must match exactly");
        c += m.cols();
    }
    Mat out(r, c);
    Index k = 0;
    for (const Mat& m : parts) { out.middleCols(k, m.cols()) = m; k += m.cols(); }
    return out;
}


// =====================================================================================
// Boolean-mask indexing: a[mask] picks the elements where mask is true, in row-major order
// (a copy, as in NumPy); A[row_mask] picks whole rows.
// =====================================================================================
template <class D, class M>
Vec select(const Eigen::ArrayBase<D>& a, const Eigen::ArrayBase<M>& mask) {
    const plain_t<D> p = a;
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m = mask;
    if (m.size() == 0) return Vec(0);  // NumPy: an empty boolean index is an empty index, whatever a's shape
    if (p.rows() != m.rows() || p.cols() != m.cols())
        throw index_error("boolean index did not match indexed array");
    Vec out(m.count());
    Index k = 0;
    for (Index i = 0; i < p.size(); ++i)
        if (m.data()[i]) out(k++) = p.data()[i];
    return out;
}

template <class D, class M>
Mat select_rows(const Eigen::ArrayBase<D>& a, const Eigen::ArrayBase<M>& mask) {
    const Mat p = a;
    const Mask m = mask;
    if (m.size() == 0) return Mat(0, p.cols());  // an empty boolean index selects no rows
    if (p.rows() != m.size()) throw index_error("boolean index did not match indexed array");
    Mat out(m.count(), p.cols());
    Index k = 0;
    for (Index i = 0; i < p.rows(); ++i)
        if (m(i)) out.row(k++) = p.row(i);
    return out;
}

// a[mask] = v and a[mask] += v with a scalar v: only the selected elements change.
template <class A, class M, class F>
void masked_update(A& a, const Eigen::ArrayBase<M>& mask, double v, F f) {
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m = mask;
    if (m.size() == 0) return;  // an empty boolean index selects nothing to change
    if (a.rows() != m.rows() || a.cols() != m.cols())
        throw index_error("boolean index did not match indexed array");
    for (Index i = 0; i < a.rows(); ++i)
        for (Index j = 0; j < a.cols(); ++j)
            if (m(i, j)) a(i, j) = f(a(i, j), v);
}

// =====================================================================================
// The linear-algebra audit (compiled in only with -DQTRANS_LINALG_AUDIT, as `qtrans verify`
// does). Each numpy.linalg operation measures, on its own inputs and outputs, the quantity
// numerical analysis bounds for it, and divides by that bound:
//   ratio <= 1: the result is what a backward-stable algorithm guarantees (the exact answer
//               for a matrix within rounding of the input), whether or not it equals NumPy's;
//   ratio > 1:  it is not; the verifier reports that as a failure.
// LU (solve, inv, det) and Cholesky have rigorous componentwise bounds (Higham, Accuracy and
// Stability of Numerical Algorithms, 2nd ed., Theorems 9.3, 9.4 and 10.3), evaluated for the
// very matrices at hand, plus the rounding of computing the residual itself, so their ratio
// can't exceed 1 unless the code is wrong. eigh and the SVD of pinv only have normwise bounds
// with an unspecified modest constant p(n); qtrans uses p(n) = 10 n (see tools/linalg_check.py
// for the worst ratio measured on thousands of constructed matrices, condition numbers up to
// 1e16). Gradual underflow adds an absolute error the model above leaves out: a term of
// n^2 * DBL_MIN per bound covers it. Non-finite inputs or results are not audited.
// =====================================================================================
#ifdef QTRANS_LINALG_AUDIT
namespace audit {
struct State {
    long count = 0;        // operations audited in this call
    double worst = 0.0;    // largest ratio
    const char* op = "";   // the operation it came from
    bool enabled = true;   // off while the verifier times a function
};
inline State& state() { static State s; return s; }
inline void reset() { const bool on = state().enabled; state() = State{}; state().enabled = on; }
inline void record(const char* op, double measured, double bound) {
    if (!(std::isfinite(measured) && std::isfinite(bound))) return;
    State& st = state();
    ++st.count;
    const double r = bound > 0 ? measured / bound : (measured > 0 ? std::numeric_limits<double>::infinity() : 0.0);
    if (r > st.worst || st.op[0] == '\0') { st.worst = std::max(st.worst, r); st.op = op; }
}
inline double gamma(double k) {  // gamma_k = k u / (1 - k u), u = 2^-53
    const double u = std::numeric_limits<double>::epsilon() / 2;
    return k * u / (1 - k * u);
}
inline double tiny(Index n) { return double(n) * double(n) * std::numeric_limits<double>::min(); }
inline bool finite(const Eigen::MatrixXd& m) { return m.allFinite(); }

// |PA - LU| <= gamma_n |L||U| (Theorem 9.3); computing PA - LU adds gamma_n |L||U| + u |A|
inline void lu(const Eigen::MatrixXd& a, const Eigen::PartialPivLU<Eigen::MatrixXd>& f) {
    const Index n = a.rows();
    if (n == 0 || !finite(a) || !finite(f.matrixLU())) return;
    const Eigen::MatrixXd L = f.matrixLU().triangularView<Eigen::UnitLower>();
    const Eigen::MatrixXd U = f.matrixLU().triangularView<Eigen::Upper>();
    const Eigen::MatrixXd pa = f.permutationP() * a;
    const double r = (pa - L * U).norm();
    const double lu_abs = (L.cwiseAbs() * U.cwiseAbs()).norm();
    record("LU (solve, inv, det)", r, 2 * gamma(double(n)) * lu_abs + gamma(1) * a.norm() + tiny(n));
}
// solves A X = B by LU: (A + dA_j) x_j = b_j with |dA_j| <= gamma_3n |L||U| (Theorem 9.4), so
// |B - AX| <= gamma_3n |L||U||X|; computing B - AX adds gamma_{n+1} (|B| + |A||X|)
inline void lu_solve(const char* op, const Eigen::MatrixXd& a, const Eigen::PartialPivLU<Eigen::MatrixXd>& f,
                     const Eigen::MatrixXd& b, const Eigen::MatrixXd& x) {
    const Index n = a.rows();
    if (n == 0 || x.size() == 0 || !finite(a) || !finite(b) || !finite(x)) return;
    const Eigen::MatrixXd L = f.matrixLU().triangularView<Eigen::UnitLower>();
    const Eigen::MatrixXd U = f.matrixLU().triangularView<Eigen::Upper>();
    const Eigen::MatrixXd lux = L.cwiseAbs() * (U.cwiseAbs() * x.cwiseAbs());
    const Eigen::MatrixXd ax = a.cwiseAbs() * x.cwiseAbs();
    const double r = (b - a * x).norm();
    record(op, r, gamma(3.0 * double(n)) * lux.norm() + gamma(double(n) + 1.0) * (b.cwiseAbs() + ax).norm() + tiny(n) * (1 + x.norm()));
}
// A = L L^T + dA with |dA| <= gamma_{n+1} |L||L^T| (Theorem 10.3); computing the residual
// adds gamma_n |L||L^T| + u |A|
inline void cholesky(const Eigen::MatrixXd& a, const Eigen::MatrixXd& l) {
    const Index n = a.rows();
    if (n == 0 || !finite(a) || !finite(l)) return;
    const Eigen::MatrixXd s = a.selfadjointView<Eigen::Lower>();
    const double r = (s - l * l.transpose()).norm();
    const double ll = (l.cwiseAbs() * l.cwiseAbs().transpose()).norm();
    record("cholesky", r, (gamma(double(n) + 1.0) + gamma(double(n))) * ll + gamma(1) * s.norm() + tiny(n));
}
// a backward-stable symmetric eigensolver: A V - V diag(w) and V^T V - I are p(n) u ||A|| and
// p(n) u, with p(n) a modest polynomial; qtrans takes p(n) = 10 n
inline double pn(Index n) { return 10.0 * double(std::max<Index>(n, 1)) * std::numeric_limits<double>::epsilon() / 2; }
inline void eigh(const Eigen::MatrixXd& a, const Eigen::VectorXd& w, const Eigen::MatrixXd& v) {
    const Index n = a.rows();
    if (n == 0 || !finite(a) || !w.allFinite() || !finite(v)) return;
    const Eigen::MatrixXd s = a.selfadjointView<Eigen::Lower>();
    const double an = s.norm();
    record("eigh: A V - V diag(w)", (s * v - v * w.asDiagonal()).norm(), pn(n) * an + tiny(n));
    record("eigh: V^T V - I", (v.transpose() * v - Eigen::MatrixXd::Identity(n, n)).norm(), pn(n));
    for (Index i = 1; i < n; ++i)
        if (w(i) < w(i - 1)) record("eigh: ascending eigenvalues", 1.0, 0.0);
}
// the pseudo-inverse X: the four Penrose conditions (detail::penrose_ratio, defined with pinv)
inline void pinv(const Eigen::MatrixXd& a, const Eigen::MatrixXd& x, double dropped = 0.0);
}  // namespace audit
#define QTRANS_AUDIT(call) do { if (::qt::audit::state().enabled) ::qt::audit::call; } while (false)
#define QTRANS_HAVE_AUDIT 1
#else
#define QTRANS_AUDIT(call) ((void)0)
#endif

// =====================================================================================
// numpy.linalg. LU with partial pivoting is the algorithm LAPACK's gesv uses; like LAPACK,
// an exactly zero pivot means "Singular matrix".
// =====================================================================================
template <class D>
Eigen::PartialPivLU<Eigen::MatrixXd> lu_checked(const Eigen::ArrayBase<D>& a) {
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    const Eigen::MatrixXd m = a.matrix();
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(m);
    for (Index i = 0; i < m.rows(); ++i)
        if (lu.matrixLU()(i, i) == 0.0) throw linalg_error("Singular matrix");
    return lu;
}

template <class DA, class DB>
plain_t<DB> solve(const Eigen::ArrayBase<DA>& a, const Eigen::ArrayBase<DB>& b) {
    // NumPy checks that A is square first (LinAlgError, which `except LinAlgError` catches),
    // then that b fits (ValueError)
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    if (b.rows() != a.rows()) throw shape_error("solve: right-hand side has the wrong number of rows");
    const auto lu = lu_checked(a);  // singular raises even with an empty right side, as in NumPy
    if (b.size() == 0) return plain_t<DB>(b.rows(), b.cols());  // Eigen 3.4's triangular solve reads x(0, 0) of an empty x
    const Eigen::MatrixXd bm = b.matrix();
    const Eigen::MatrixXd x = lu.solve(bm);
    QTRANS_AUDIT(lu(a.matrix(), lu));
    QTRANS_AUDIT(lu_solve("solve", a.matrix(), lu, bm, x));
    return x.array();
}

template <class D>
Mat inv(const Eigen::ArrayBase<D>& a) {
    const auto lu = lu_checked(a);
    const Eigen::MatrixXd x = lu.inverse();
    QTRANS_AUDIT(lu(a.matrix(), lu));
    QTRANS_AUDIT(lu_solve("inv", a.matrix(), lu, Eigen::MatrixXd::Identity(a.rows(), a.rows()), x));
    return x.array();
}

template <class D>
double det(const Eigen::ArrayBase<D>& a) {
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    const Eigen::MatrixXd m = a.matrix();
    const Eigen::PartialPivLU<Eigen::MatrixXd> lu(m);
    QTRANS_AUDIT(lu(m, lu));  // det is the product of U's diagonal: the factorization is what to audit
    return lu.determinant();
}

// Lower-triangular L with A = L L^T, read from the lower triangle as LAPACK's potrf does.
template <class D>
Mat cholesky(const Eigen::ArrayBase<D>& a) {
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    Eigen::LLT<Eigen::MatrixXd> llt{Eigen::MatrixXd(a.matrix())};
    if (llt.info() != Eigen::Success) throw linalg_error("Matrix is not positive definite");
    const Eigen::MatrixXd l = llt.matrixL();
    QTRANS_AUDIT(cholesky(a.matrix(), l));
    return l.array();
}

// np.linalg.eigh(A): eigenvalues in ascending order and eigenvectors as columns, read from the
// lower triangle as LAPACK's syevd does (NumPy's default UPLO='L'). Each eigenvector is only
// defined up to its sign (and, for a repeated eigenvalue, up to a rotation within its
// eigenspace), and LAPACK and Eigen may choose differently: results built from V diag(f(w)) V^T
// agree; code that reads individual eigenvector entries may not. NaN in the lower triangle
// gives NaN everywhere, as LAPACK does.
template <class D>
std::tuple<Vec, Mat> eigh(const Eigen::ArrayBase<D>& a) {
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    const Eigen::MatrixXd m = a.matrix();
    const Index n = m.rows();
    for (Index j = 0; j < n; ++j)
        for (Index i = j; i < n; ++i)
            if (std::isnan(m(i, j))) return {Vec::Constant(n, nan), Mat::Constant(n, n, nan)};
    if (n == 0) return {Vec(0), Mat(0, 0)};
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);  // uses the lower triangle
    if (es.info() != Eigen::Success) throw linalg_error("Eigenvalues did not converge");
    QTRANS_AUDIT(eigh(m, es.eigenvalues(), es.eigenvectors()));
    return {es.eigenvalues().array(), Mat(es.eigenvectors().array())};
}

template <class D>
Vec eigvalsh(const Eigen::ArrayBase<D>& a) { return std::get<0>(eigh(a)); }

namespace detail {
// The four Penrose conditions for X = pinv(A), each residual over the bound a stable
// computation meets (p(n) u times the scale: ||A||^2 ||X|| for AXA = A, ||X||^2 ||A|| for
// XAX = X, ||A|| ||X|| for the symmetry of AX and XA; p(n) = 10 n): the largest ratio.
// `dropped`: the 2-norm of the singular values below the cutoff (rcond * max(s)) that were set
// aside. X is then the pseudo-inverse of A without them, and AXA - A is exactly their part.
inline double penrose_ratio(const Eigen::MatrixXd& a, const Eigen::MatrixXd& x, const char** which = nullptr,
                            double dropped = 0.0) {
    const Index n = std::max(a.rows(), a.cols());
    const double an = a.norm(), xn = x.norm();
    if (a.size() == 0 || an == 0 || xn == 0 || !a.allFinite() || !x.allFinite()) return 0.0;
    const double pn = 10.0 * double(n) * std::numeric_limits<double>::epsilon() / 2;
    const double t = double(n) * double(n) * std::numeric_limits<double>::min();
    const Eigen::MatrixXd ax = a * x, xa = x * a;
    const double r[4] = {(ax * a - a).norm() / (pn * an * an * xn + dropped * (1 + pn) + t), (xa * x - x).norm() / (pn * xn * xn * an + t),
                         (ax - ax.transpose()).norm() / (pn * an * xn + t), (xa - xa.transpose()).norm() / (pn * an * xn + t)};
    static const char* names[4] = {"pinv: A X A = A", "pinv: X A X = X", "pinv: A X symmetric", "pinv: X A symmetric"};
    int k = 0;
    for (int i = 1; i < 4; ++i)
        if (r[i] > r[k]) k = i;
    if (which) *which = names[k];
    return r[k];
}
}  // namespace detail

// np.linalg.pinv(A, rcond=1e-15). NumPy keeps the singular values s > rcond * max(s) and
// returns V diag(1/s) U^T. The rank is decided here by exactly that rule, on Eigen's SVD.
// When every singular value is kept, the pseudo-inverse is unique, and a column-pivoted
// complete orthogonal decomposition computes it more accurately on badly scaled matrices
// (entries of 1e8 next to 1e-2: the SVD formula lost 5-6 digits against a 60-digit reference
// where LAPACK's stayed near 1e-15; the QR route stays as accurate as LAPACK's). On a nearly
// rank-deficient matrix, though, the QR route can break down (the linear-algebra audit found
// A X off symmetric by 1e13 times its bound), so its result is used only when it meets all
// four Penrose conditions within a stable computation's bounds; otherwise, and for a
// rank-deficient matrix, the SVD formula, as NumPy. Singular values within rounding of the
// cutoff can be kept by one implementation and dropped by the other.
template <class D>
Mat pinv(const Eigen::ArrayBase<D>& a, double rcond = 1e-15) {
    const Eigen::MatrixXd m = a.matrix();
    if (m.size() == 0) return Mat(m.cols(), m.rows());
    if (m.hasNaN()) throw linalg_error("SVD did not converge");
#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(m);
#else
    Eigen::BDCSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
#endif
    Eigen::VectorXd s = svd.singularValues();
    const double cutoff = rcond * s.maxCoeff();
    Index rank = 0;
    for (Index i = 0; i < s.size(); ++i) rank += s(i) > cutoff;
    if (rank == s.size() && rank > 0) {
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(m);
        cod.setThreshold(0.0);  // full rank by NumPy's rule: keep every direction
        if (cod.rank() == rank) {
            const Eigen::MatrixXd x = cod.pseudoInverse();
            if (detail::penrose_ratio(m, x) <= 1.0) {
                QTRANS_AUDIT(pinv(m, x));
                return x.array();
            }
        }
    }
    double dropped = 0.0;  // what the cutoff sets aside (for the audit: AXA - A is exactly that)
    for (Index i = 0; i < s.size(); ++i) {
        if (!(s(i) > cutoff)) dropped += s(i) * s(i);
        s(i) = s(i) > cutoff ? 1.0 / s(i) : 0.0;
    }
    const Eigen::MatrixXd res = svd.matrixV() * s.asDiagonal() * svd.matrixU().transpose();
    QTRANS_AUDIT(pinv(m, res, std::sqrt(dropped)));
    (void)dropped;
    return res.array();
}

#ifdef QTRANS_LINALG_AUDIT
inline void audit::pinv(const Eigen::MatrixXd& a, const Eigen::MatrixXd& x, double dropped) {
    if (a.size() == 0 || !a.allFinite() || !x.allFinite()) return;
    const char* which = "pinv";
    const double r = detail::penrose_ratio(a, x, &which, dropped);
    record(which, r, 1.0);
}
#endif

// np.einsum for one or two operands of rank <= 2 ("ij,ij->i", "ij,jk->ik", "ii->", ...).
// Every combination of index values is visited and the products added in sequence; NumPy's
// own kernels add in a different order, so results agree to rounding, not bit for bit.
// An einsum operand: a stored Mat is used where it is; a Vec becomes a 1 x n row; any other
// expression is evaluated once.
struct EinsumArg {
    const Mat* ptr = nullptr;
    Mat own;
    EinsumArg(const Mat& m) : ptr(&m) {}  // NOLINT: implicit on purpose, for {A, B}
    EinsumArg(const Vec& v) : own(as_row(v)) {}
    template <class D>
    EinsumArg(const Eigen::ArrayBase<D>& x) : own(x) {
        if constexpr (D::ColsAtCompileTime == 1) own = as_row(x);
    }
    const Mat& get() const { return ptr ? *ptr : own; }
};

inline std::vector<std::string> einsum_split(const std::string& spec, std::string& out) {
    std::vector<std::string> ins;
    const auto arrow = spec.find("->");
    const std::string lhs = arrow == std::string::npos ? spec : spec.substr(0, arrow);
    std::string cur;
    for (char ch : lhs) {
        if (ch == ',') { ins.push_back(cur); cur.clear(); }
        else if (ch != ' ') cur += ch;
    }
    ins.push_back(cur);
    out.clear();
    if (arrow != std::string::npos) {
        for (char ch : spec.substr(arrow + 2)) if (ch != ' ') out += ch;
    } else {  // implicit mode: the labels that appear exactly once, in alphabetical order
        for (char ch = 'a'; ch <= 'z'; ++ch) {
            int count = 0;
            for (const auto& s : ins) for (char c : s) count += c == ch;
            if (count == 1) out += ch;
        }
    }
    return ins;
}

inline Mat einsum_core(const std::string& spec, std::initializer_list<EinsumArg> mats, const std::vector<int>& ranks) {
    std::string outl;
    const std::vector<std::string> ins = einsum_split(spec, outl);
    if (ins.size() != mats.size()) throw shape_error("einsum: wrong number of operands");
    std::vector<const Mat*> ops;
    for (const EinsumArg& m : mats) ops.push_back(&m.get());
    // labels in order of first appearance; each gets a slot 0..L-1 and a size
    std::string all;
    for (const auto& l : ins) for (char c : l) if (all.find(c) == std::string::npos) all += c;
    for (char c : outl) if (all.find(c) == std::string::npos) throw shape_error("einsum: output label not in inputs");
    const auto slot = [&](char c) { return static_cast<int>(all.find(c)); };
    std::vector<long long> size(all.size(), -1);
    // per operand: the slot of each of its (up to two) dimensions, -1 where it broadcasts or is absent
    struct Access { int s0 = -1, s1 = -1; };
    std::vector<Access> acc(ops.size());
    for (std::size_t q = 0; q < ops.size(); ++q) {
        const std::string& l = ins[q];
        const Mat& m = *ops[q];
        if (static_cast<int>(l.size()) != ranks[q]) throw shape_error("einsum: subscripts don't match the operand's dimensions");
        for (std::size_t d = 0; d < l.size(); ++d) {
            const long long n = ranks[q] == 1 ? static_cast<long long>(m.size()) : (d == 0 ? m.rows() : m.cols());
            long long& z = size[slot(l[d])];
            // a dimension of size 1 broadcasts against any other size, as in NumPy
            if (z >= 0 && z != n && z != 1 && n != 1)
                throw shape_error("einsum: operands could not be broadcast together");
            if (z < 0 || z == 1) z = n;
        }
    }
    for (std::size_t q = 0; q < ops.size(); ++q) {
        const std::string& l = ins[q];
        const Mat& m = *ops[q];
        const long long n0 = ranks[q] == 1 ? static_cast<long long>(m.size()) : m.rows();
        if (l.size() > 0) acc[q].s0 = n0 == 1 && size[slot(l[0])] != 1 ? -1 : slot(l[0]);
        if (l.size() > 1) acc[q].s1 = m.cols() == 1 && size[slot(l[1])] != 1 ? -1 : slot(l[1]);
    }
    const int o0 = outl.size() > 0 ? slot(outl[0]) : -1, o1 = outl.size() > 1 ? slot(outl[1]) : -1;
    // nothing broadcasts: the common patterns go to direct loops / products
    bool plain = true;
    for (std::size_t q = 0; q < ops.size(); ++q)
        plain = plain && (ins[q].size() < 1 || acc[q].s0 >= 0) && (ins[q].size() < 2 || acc[q].s1 >= 0);
    if (plain && ops.size() == 2) {
        const Mat& A = *ops[0];
        const Mat& B = *ops[1];
        const bool same = ins[0] == ins[1] && ins[0].size() == 2 && ins[0][0] != ins[0][1];
        if (same && outl == ins[0].substr(0, 1)) {  // "ij,ij->i": a dot product per row
            Mat out(A.rows(), 1);
            for (Index i = 0; i < A.rows(); ++i) {
                double acc_ = 0.0;
                for (Index j = 0; j < A.cols(); ++j) acc_ += A(i, j) * B(i, j);
                out(i, 0) = acc_;
            }
            return out;
        }
        if (ins[0].size() == 2 && ins[1].size() == 2 && ins[0][1] == ins[1][0] && ins[0][0] != ins[0][1] &&
            ins[1][0] != ins[1][1] && ins[0][0] != ins[1][1] && outl == std::string{ins[0][0], ins[1][1]})
            return (A.matrix() * B.matrix()).array();  // "ij,jk->ik"
    }
    Mat out = Mat::Zero(o0 >= 0 ? size[o0] : 1, o1 >= 0 ? size[o1] : 1);
    for (long long z : size) if (z == 0) return out;
    std::vector<long long> at(all.size(), 0);
    const std::size_t L = all.size();
    for (;;) {
        double prod = 1.0;
        for (std::size_t q = 0; q < ops.size(); ++q) {
            const Mat& m = *ops[q];
            const long long i = acc[q].s0 >= 0 ? at[acc[q].s0] : 0;
            if (ranks[q] == 0) prod *= m(0, 0);
            else if (ranks[q] == 1) prod *= m.data()[i];
            else prod *= m(i, acc[q].s1 >= 0 ? at[acc[q].s1] : 0);
        }
        out(o0 >= 0 ? at[o0] : 0, o1 >= 0 ? at[o1] : 0) += prod;
        std::size_t d = L;
        while (d > 0) {
            if (++at[d - 1] < size[d - 1]) break;
            at[d - 1] = 0;
            --d;
        }
        if (d == 0) break;
    }
    return out;
}

// The result as a number, a Vec or a Mat, by the number of output labels. A Vec operand is
// passed as a 1 x n Mat and its rank as 1.
inline double einsum0(const std::string& spec, std::initializer_list<EinsumArg> mats, const std::vector<int>& ranks) {
    return einsum_core(spec, mats, ranks)(0, 0);
}
inline Vec einsum1(const std::string& spec, std::initializer_list<EinsumArg> mats, const std::vector<int>& ranks) {
    return ravel(einsum_core(spec, mats, ranks));
}
inline Mat einsum2(const std::string& spec, std::initializer_list<EinsumArg> mats, const std::vector<int>& ranks) {
    return einsum_core(spec, mats, ranks);
}


// =====================================================================================
// int64 arrays (0.14): np.arange of ints, rng.integers, index arrays. NumPy's int64
// arithmetic wraps around at 2**64 without an error, so it is done in unsigned arithmetic
// here (where signed overflow would be undefined behaviour in C++).
// =====================================================================================
namespace ia {
inline long long wrap_add(long long a, long long b) { return static_cast<long long>(static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b)); }
inline long long wrap_sub(long long a, long long b) { return static_cast<long long>(static_cast<unsigned long long>(a) - static_cast<unsigned long long>(b)); }
inline long long wrap_mul(long long a, long long b) { return static_cast<long long>(static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b)); }
// NumPy's int floor division and modulo: Python's signs, x // 0 == x % 0 == 0 (with a warning)
inline long long np_floordiv(long long a, long long b) {
    if (b == 0) return 0;
    if (b == -1) return wrap_sub(0, a);  // (LLONG_MIN // -1 wraps to LLONG_MIN)
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
inline long long np_mod(long long a, long long b) {
    if (b == 0 || b == -1) return 0;
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
// NumPy's int ** int: int64, wrapping (a product mod 2**64 doesn't depend on the order it is
// worked out in, so squaring gives NumPy's result); a negative exponent raises, element by element
inline long long np_pow(long long a, long long b) {
    if (b < 0) throw ValueError("Integers to negative integer powers are not allowed.");
    unsigned long long base = static_cast<unsigned long long>(a), r = 1;
    for (unsigned long long e = static_cast<unsigned long long>(b); e != 0; e >>= 1) {
        if (e & 1) r *= base;
        base *= base;
    }
    return static_cast<long long>(r);
}
}  // namespace ia

inline IVec ivec(std::initializer_list<long long> values) {  // np.array([1, 2, 3])
    IVec out(static_cast<Index>(values.size()));
    Index i = 0;
    for (long long v : values) out(i++) = v;
    return out;
}

// np.arange(start, stop, step) of ints: length ceil((stop - start) / step), as NumPy works it out
inline IVec iarange(long long start, long long stop, long long step = 1) {
    if (step == 0) throw ZeroDivisionError(msg::int_truediv_zero);
    const double len = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) / static_cast<double>(step));
    const Index n = len > 0 ? static_cast<Index>(len) : 0;
    IVec out(n);
    for (Index i = 0; i < n; ++i) out(i) = ia::wrap_add(start, ia::wrap_mul(static_cast<long long>(i), step));
    return out;
}
inline IVec iarange(long long stop) { return iarange(0, stop, 1); }

inline IVec izeros(long long n) { check_dim(n); return IVec::Zero(static_cast<Index>(n)); }
inline IVec iones(long long n) { check_dim(n); return IVec::Ones(static_cast<Index>(n)); }
inline IVec ifull(long long n, long long v) { check_dim(n); return IVec::Constant(static_cast<Index>(n), v); }

// elementwise int64 arithmetic, wrapping as NumPy's does
template <class F>
IVec imap2(const IVec& a, const IVec& b, F f) {
    if (a.size() != b.size() && a.size() != 1 && b.size() != 1)
        throw shape_error("operands could not be broadcast together with shapes (" + std::to_string(a.size()) +
                          ",) (" + std::to_string(b.size()) + ",)");
    const Index n = std::max(a.size(), b.size());
    IVec out(n);
    for (Index i = 0; i < n; ++i) out(i) = f(a(a.size() == 1 ? 0 : i), b(b.size() == 1 ? 0 : i));
    return out;
}
template <class F>
IVec imap1(const IVec& a, long long k, F f, bool k_first = false) {
    IVec out(a.size());
    for (Index i = 0; i < a.size(); ++i) out(i) = k_first ? f(k, a(i)) : f(a(i), k);
    return out;
}
inline IVec iadd(const IVec& a, const IVec& b) { return imap2(a, b, ia::wrap_add); }
inline IVec isub(const IVec& a, const IVec& b) { return imap2(a, b, ia::wrap_sub); }
inline IVec imul(const IVec& a, const IVec& b) { return imap2(a, b, ia::wrap_mul); }
inline IVec ifloordiv(const IVec& a, const IVec& b) { return imap2(a, b, ia::np_floordiv); }
inline IVec imod(const IVec& a, const IVec& b) { return imap2(a, b, ia::np_mod); }
inline IVec iadd(const IVec& a, long long k) { return imap1(a, k, ia::wrap_add); }
inline IVec isub(const IVec& a, long long k) { return imap1(a, k, ia::wrap_sub); }
inline IVec isub(long long k, const IVec& a) { return imap1(a, k, ia::wrap_sub, true); }
inline IVec imul(const IVec& a, long long k) { return imap1(a, k, ia::wrap_mul); }
inline IVec ifloordiv(const IVec& a, long long k) { return imap1(a, k, ia::np_floordiv); }
inline IVec ifloordiv(long long k, const IVec& a) { return imap1(a, k, ia::np_floordiv, true); }
inline IVec imod(const IVec& a, long long k) { return imap1(a, k, ia::np_mod); }
inline IVec imod(long long k, const IVec& a) { return imap1(a, k, ia::np_mod, true); }
inline IVec ipow(const IVec& a, const IVec& b) { return imap2(a, b, ia::np_pow); }
inline IVec ipow(const IVec& a, long long k) { return imap1(a, k, ia::np_pow); }
inline IVec ipow(long long k, const IVec& a) { return imap1(a, k, ia::np_pow, true); }
inline IVec ineg(const IVec& a) { return imap1(a, 0, ia::wrap_sub, true); }
inline IVec iabs(const IVec& a) {  // (abs(LLONG_MIN) stays LLONG_MIN, as in NumPy)
    IVec out(a.size());
    for (Index i = 0; i < a.size(); ++i) out(i) = a(i) < 0 ? ia::wrap_sub(0, a(i)) : a(i);
    return out;
}

inline long long isum(const IVec& a) {
    long long s = 0;
    for (Index i = 0; i < a.size(); ++i) s = ia::wrap_add(s, a(i));
    return s;
}
inline long long iprod(const IVec& a) {
    long long s = 1;
    for (Index i = 0; i < a.size(); ++i) s = ia::wrap_mul(s, a(i));
    return s;
}
inline IVec icumsum(const IVec& a) {
    IVec out(a.size());
    long long s = 0;
    for (Index i = 0; i < a.size(); ++i) out(i) = s = ia::wrap_add(s, a(i));
    return out;
}
inline long long imax(const IVec& a) {
    if (a.size() == 0) throw ValueError("zero-size array to reduction operation maximum which has no identity");
    return a.maxCoeff();
}
inline long long imin(const IVec& a) {
    if (a.size() == 0) throw ValueError("zero-size array to reduction operation minimum which has no identity");
    return a.minCoeff();
}

// one element: idx[i] (a negative i counts from the end), with NumPy's IndexError
inline Index np_index(long long i, Index n) {
    if (i < -static_cast<long long>(n) || i >= static_cast<long long>(n))
        throw IndexError("index " + std::to_string(i) + " is out of bounds for axis 0 with size " + std::to_string(n));
    return static_cast<Index>(i < 0 ? i + static_cast<long long>(n) : i);
}

// x[idx]: the elements at those positions (NumPy's fancy indexing: a new array)
template <class A>
A take(const A& x, const IVec& idx) {
    A out(idx.size());
    for (Index k = 0; k < idx.size(); ++k) out(k) = x(np_index(idx(k), x.size()));
    return out;
}
// A[idx]: those rows of a matrix
inline Mat take_rows(const Mat& m, const IVec& idx) {
    Mat out(idx.size(), m.cols());
    for (Index k = 0; k < idx.size(); ++k) out.row(k) = m.row(np_index(idx(k), m.rows()));
    return out;
}

// x.astype(np.int64): toward zero; NaN and values out of range give what the CPU's conversion
// gives, as NumPy's own C cast does (x86: the smallest int64; ARM: saturated, NaN as 0)
inline long long to_int64(double v) {
    if (v >= -9223372036854775808.0 && v < 9223372036854775808.0) return static_cast<long long>(v);
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    if (std::isnan(v)) return 0;
    return v < 0 ? std::numeric_limits<long long>::min() : std::numeric_limits<long long>::max();
#else
    return std::numeric_limits<long long>::min();
#endif
}
template <class D>
IVec astype_int(const Eigen::ArrayBase<D>& x) {
    const auto v = x.eval();
    IVec out(v.size());
    for (Index i = 0; i < v.size(); ++i) out(i) = to_int64(v(i));
    return out;
}
inline Vec astype_float(const IVec& a) { return a.cast<double>(); }

inline long long iargmax(const IVec& a) {  // the first position of the largest
    if (a.size() == 0) throw ValueError("attempt to get argmax of an empty sequence");
    Index k = 0;
    for (Index i = 1; i < a.size(); ++i) if (a(i) > a(k)) k = i;
    return static_cast<long long>(k);
}
inline long long iargmin(const IVec& a) {
    if (a.size() == 0) throw ValueError("attempt to get argmin of an empty sequence");
    Index k = 0;
    for (Index i = 1; i < a.size(); ++i) if (a(i) < a(k)) k = i;
    return static_cast<long long>(k);
}
inline IVec isorted(IVec a) {  // np.sort of ints: any correct sort gives the same array
    std::sort(a.data(), a.data() + a.size());
    return a;
}
// two int arrays compared element by element (a length-1 side stretches, as in NumPy)
template <class F>
Mask icmp(const IVec& a, const IVec& b, F f) {
    if (a.size() != b.size() && a.size() != 1 && b.size() != 1)
        throw shape_error("operands could not be broadcast together with shapes (" + std::to_string(a.size()) +
                          ",) (" + std::to_string(b.size()) + ",)");
    const Index n = std::max(a.size(), b.size());
    Mask out(n);
    for (Index i = 0; i < n; ++i) out(i) = f(a(a.size() == 1 ? 0 : i), b(b.size() == 1 ? 0 : i));
    return out;
}

}  // namespace qt
