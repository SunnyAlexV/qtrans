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
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "qtrans_runtime.hpp"

namespace qt {
struct shape_error : std::invalid_argument {   // NumPy: ValueError (shapes), IndexError (bounds)
    using std::invalid_argument::invalid_argument;
};
struct index_error : std::out_of_range {       // NumPy: IndexError
    using std::out_of_range::out_of_range;
};
struct linalg_error : std::invalid_argument {  // numpy.linalg.LinAlgError (a ValueError)
    using std::invalid_argument::invalid_argument;
};
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
    if (step == 0.0) throw std::domain_error("division by zero");  // NumPy: ZeroDivisionError
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

// A view into a (readable and assignable), like a NumPy slice.
template <class A>
auto slice(A& a, std::optional<long long> start, std::optional<long long> stop, long long step = 1) {
    const Span s = span(a.size(), start, stop, step);
    return a(Eigen::seqN(s.first, s.size, s.step));
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
    return a.binaryExpr(b, [](double x, double y) { return std::pow(x, y); });
}

// np.maximum / np.minimum propagate NaN from either side (Eigen's max/min don't).
template <class D1, class D2>
auto maximum(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return a.binaryExpr(b, [](double x, double y) { return np_maximum(x, y); });
}
template <class D>
auto maximum(const Eigen::ArrayBase<D>& a, double b) {
    return a.unaryExpr([b](double x) { return np_maximum(x, b); });
}
template <class D>
auto maximum(double a, const Eigen::ArrayBase<D>& b) { return maximum(b, a); }
template <class D1, class D2>
auto minimum(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return a.binaryExpr(b, [](double x, double y) { return np_minimum(x, y); });
}
template <class D>
auto minimum(const Eigen::ArrayBase<D>& a, double b) {
    return a.unaryExpr([b](double x) { return np_minimum(x, b); });
}
template <class D>
auto minimum(double a, const Eigen::ArrayBase<D>& b) { return minimum(b, a); }

// np.where(cond, x, y) with arrays or scalars on either branch.
inline double at(double v, Index, Index) { return v; }
template <class D>
double at(const Eigen::ArrayBase<D>& a, Index i, Index j) { return a(i, j); }

template <class X, class C>
void check_same_shape(const X& x, const C& c) {
    if constexpr (!std::is_arithmetic_v<X>) {
        if (x.rows() != c.rows() || x.cols() != c.cols())
            throw shape_error("operands could not be broadcast together");
    }
}

template <class C, class X, class Y>
auto where(const Eigen::ArrayBase<C>& cond, const X& x, const Y& y) {
    check_same_shape(x, cond);
    check_same_shape(y, cond);
    plain_t<C> out(cond.rows(), cond.cols());
    for (Index i = 0; i < cond.rows(); ++i)
        for (Index j = 0; j < cond.cols(); ++j)
            out(i, j) = cond(i, j) ? at(x, i, j) : at(y, i, j);
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

// np.sum(a): pairwise over all elements in row-major (C) order.
template <class D>
double sum(const Eigen::ArrayBase<D>& a) {
    const plain_t<D> p = a;
    return pairwise(p.data(), p.size(), 1);
}

// np.sum(A, axis=0): NumPy adds rows one after another (column-wise sequential), except for
// a single-column matrix, which it treats as contiguous and sums pairwise.
// np.sum(A, axis=1): pairwise along each row.
template <class D>
Vec sum(const Eigen::ArrayBase<D>& a, int axis) {
    const Mat m = a;
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
    if (!(q >= 0.0 && q <= 1.0)) throw std::invalid_argument("Quantiles must be in the range [0, 1]");
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
template <class D1, class D2>
double dot(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {  // Vec @ Vec
    if (a.size() != b.size()) throw shape_error("matmul: mismatch in core dimension");
    return a.matrix().dot(b.matrix());
}

template <class D1, class D2>
Vec matvec(const Eigen::ArrayBase<D1>& m, const Eigen::ArrayBase<D2>& v) {  // Mat @ Vec
    if (m.cols() != v.size()) throw shape_error("matmul: mismatch in core dimension");
    return (m.matrix() * v.matrix()).array();
}

template <class D1, class D2>
Vec vecmat(const Eigen::ArrayBase<D1>& v, const Eigen::ArrayBase<D2>& m) {  // Vec @ Mat
    if (v.size() != m.rows()) throw shape_error("matmul: mismatch in core dimension");
    return (v.matrix().transpose() * m.matrix()).transpose().array();
}

template <class D1, class D2>
Mat matmul(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {  // Mat @ Mat
    if (a.cols() != b.rows()) throw shape_error("matmul: mismatch in core dimension");
    return (a.matrix() * b.matrix()).array();
}

template <class D1, class D2>
Mat outer(const Eigen::ArrayBase<D1>& a, const Eigen::ArrayBase<D2>& b) {
    return (a.matrix() * b.matrix().transpose()).array();
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
    if (b.rows() != a.rows()) throw shape_error("solve: right-hand side has the wrong number of rows");
    const auto lu = lu_checked(a);
    const Eigen::MatrixXd x = lu.solve(b.matrix().eval());
    return x.array();
}

template <class D>
Mat inv(const Eigen::ArrayBase<D>& a) {
    return lu_checked(a).inverse().array();
}

template <class D>
double det(const Eigen::ArrayBase<D>& a) {
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    return Eigen::MatrixXd(a.matrix()).partialPivLu().determinant();
}

// Lower-triangular L with A = L L^T, read from the lower triangle as LAPACK's potrf does.
template <class D>
Mat cholesky(const Eigen::ArrayBase<D>& a) {
    if (a.rows() != a.cols()) throw linalg_error("Last 2 dimensions of the array must be square");
    Eigen::LLT<Eigen::MatrixXd> llt{Eigen::MatrixXd(a.matrix())};
    if (llt.info() != Eigen::Success) throw linalg_error("Matrix is not positive definite");
    return Eigen::MatrixXd(llt.matrixL()).array();
}

}  // namespace qt
