// qtrans_runtime.hpp: the small set of helpers generated code relies on.
//
// Each function here exists because C++ and Python disagree on something, and the
// generated code must compute what Python computes. Everything is header-only,
// depends only on the standard library, and is written to be read.
#pragma once

#include <climits>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace qt {

inline constexpr double pi = 3.141592653589793;
inline constexpr double e = 2.718281828459045;
inline constexpr double tau = 6.283185307179586;
inline constexpr double inf = std::numeric_limits<double>::infinity();
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------
// `x: float | None` is std::optional<double>. Where Python needs the number itself
// (arithmetic, a comparison, int(x), passing it on as a float), None raises TypeError;
// qt::value checks the same way. Truthiness: None is false, otherwise the value's own.
// ---------------------------------------------------------------------------------
struct type_error : std::runtime_error {  // Python: TypeError
    using std::runtime_error::runtime_error;
};

template <class T>
const T& value(const std::optional<T>& x) {
    if (!x) throw type_error("'NoneType' used where a value is needed");
    return *x;
}

template <class T>
bool truthy(const std::optional<T>& x) {
    return x.has_value() && *x != T(0);
}

// ---------------------------------------------------------------------------------
// Integer // and %.  Python rounds the quotient toward -infinity and gives the
// remainder the sign of the divisor:  -7 // 2 == -4,  -7 % 2 == 1.
// C++ truncates toward zero:          -7 / 2  == -3,  -7 % 2 == -1.
// Division by zero raises ZeroDivisionError in Python; std::domain_error here.
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// int arithmetic that could leave 64 bits. Python ints are unbounded; a long long is not,
// and signed overflow is undefined behaviour in C++. Where the translator can't show a
// result stays small (loop indices, array sizes and small literals stay plain + - *), it
// emits these: the exact result, or OverflowError where Python would return a big int.
// ---------------------------------------------------------------------------------
struct int_overflow : std::overflow_error {  // Python: a big int; here: OverflowError
    int_overflow() : std::overflow_error("integer result beyond 64 bits: Python ints are unbounded, a C++ long long "
                                         "is not") {}
};

inline long long iadd(long long a, long long b) {
#if defined(__GNUC__) || defined(__clang__)
    long long r;
    if (__builtin_add_overflow(a, b, &r)) throw int_overflow();
    return r;
#else
    if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b)) throw int_overflow();
    return a + b;
#endif
}

inline long long isub(long long a, long long b) {
#if defined(__GNUC__) || defined(__clang__)
    long long r;
    if (__builtin_sub_overflow(a, b, &r)) throw int_overflow();
    return r;
#else
    if ((b < 0 && a > LLONG_MAX + b) || (b > 0 && a < LLONG_MIN + b)) throw int_overflow();
    return a - b;
#endif
}

inline long long imul(long long a, long long b) {
#if defined(__GNUC__) || defined(__clang__)
    long long r;
    if (__builtin_mul_overflow(a, b, &r)) throw int_overflow();
    return r;
#else
    if (a == 0 || b == 0) return 0;
    if (a > 0 ? (b > 0 ? a > LLONG_MAX / b : b < LLONG_MIN / a)
              : (b > 0 ? a < LLONG_MIN / b : a < LLONG_MAX / b))
        throw int_overflow();
    return a * b;
#endif
}

inline long long ineg(long long a) {
    if (a == LLONG_MIN) throw int_overflow();
    return -a;
}

inline long long iabs(long long a) {
    if (a == LLONG_MIN) throw int_overflow();
    return a < 0 ? -a : a;
}

inline long long floordiv(long long a, long long b) {
    if (b == 0) throw std::domain_error("integer division or modulo by zero");
    if (a == LLONG_MIN && b == -1) throw int_overflow();  // -(-2**63) is 2**63
    long long q = a / b;
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) --q;
    return q;
}

inline long long mod(long long a, long long b) {
    if (b == 0) throw std::domain_error("integer division or modulo by zero");
    if (b == -1) return 0;  // LLONG_MIN % -1 is undefined in C++; Python gives 0
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

// ---------------------------------------------------------------------------------
// Float // and %: a line-for-line port of CPython's float_rem and _float_div_mod
// (Objects/floatobject.c). The naive std::floor(a / b) is wrong in edge cases:
// 1.0 // 0.1 is 9.0 in Python (0.1 is slightly above 1/10), but floor(1.0 / 0.1)
// is 10.0 because the division rounds up to exactly 10.
// ---------------------------------------------------------------------------------
inline double mod(double vx, double wx) {
    if (wx == 0.0) throw std::domain_error("float modulo");
    double m = std::fmod(vx, wx);
    if (m != 0.0) {
        if ((wx < 0) != (m < 0)) m += wx;
    } else {
        m = std::copysign(0.0, wx);
    }
    return m;
}

inline double floordiv(double vx, double wx) {
    if (wx == 0.0) throw std::domain_error("float floor division by zero");
    double m = std::fmod(vx, wx);
    double div = (vx - m) / wx;
    if (m != 0.0) {
        if ((wx < 0) != (m < 0)) div -= 1.0;
    }
    double result;
    if (div != 0.0) {
        result = std::floor(div);
        if (div - result > 0.5) result += 1.0;
    } else {
        result = std::copysign(0.0, vx / wx);
    }
    return result;
}

// ---------------------------------------------------------------------------------
// float -> int (int(), round(), math.floor/ceil/trunc).  A bare static_cast is
// undefined behaviour for NaN, inf or anything beyond 64 bits. Python raises
// ValueError for NaN and OverflowError for inf; for huge finite values Python returns
// a big integer, which a long long can't hold, so this raises OverflowError instead of
// returning garbage. The argument is already integral (floored, rounded or truncated).
// ---------------------------------------------------------------------------------
inline long long to_int(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");  // as Python
    if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0))
        throw int_overflow();  // Python returns a big int
    return static_cast<long long>(x);
}

// ---------------------------------------------------------------------------------
// int ** int for a non-negative constant exponent, by repeated squaring.
// ---------------------------------------------------------------------------------
inline long long ipow(long long base, long long exp) {
    // checked: base is squared only while higher exponent bits remain, so an overflowing square
    // means the result itself overflows (for |base| >= 2)
    long long result = 1;
    while (exp > 0) {
        if (exp & 1) result = imul(result, base);
        exp >>= 1;
        if (exp) base = imul(base, base);
    }
    return result;
}

// ---------------------------------------------------------------------------------
// numpy.maximum / numpy.minimum propagate NaN from either argument; std::max does not.
// On a tie they return the SECOND argument, which shows with signed zeros:
// np.maximum(-0.0, 0.0) is 0.0 and np.maximum(0.0, -0.0) is -0.0 (and 1/x tells them apart).
// np.clip breaks ties the other way, keeping x: np.clip(-0.0, 0.0, 1.0) is -0.0.
// ---------------------------------------------------------------------------------
inline double np_maximum(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return nan;
    return a > b ? a : b;
}

inline double np_minimum(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return nan;
    return a < b ? a : b;
}

// np.clip(x, lo, hi) = min(max(x, lo), hi), with NaN from any argument giving NaN and ties
// keeping x. When lo > hi the result is hi, as in NumPy.
inline double np_clip(double x, double lo, double hi) {
    if (std::isnan(x) || std::isnan(lo) || std::isnan(hi)) return nan;
    const double m = x >= lo ? x : lo;
    return m <= hi ? m : hi;
}

// np.where(c, x, y) with a scalar condition: both branches are evaluated first, as NumPy
// evaluates its arguments, and then one is chosen.
inline double where(bool c, double x, double y) { return c ? x : y; }
inline long long where(bool c, long long x, long long y) { return c ? x : y; }

template <class T>
inline T np_sign(T x) {
    if (x > 0) return T(1);
    if (x < 0) return T(-1);
    return x;  // 0 stays 0 (with its sign); NaN stays NaN, as in numpy
}

template <class T>
inline T square(T x) { return x * x; }

// ---------------------------------------------------------------------------------
// The standard normal distribution, computed the way SciPy computes it, so results
// agree to the last bit or two.
//
// norm_cdf: SciPy's ndtr. Near zero it uses erf; in the tails it uses erfc of |x|,
//   which keeps full relative accuracy for tiny probabilities (1 - 0.5 * erfc(...)
//   is only taken on the side where the result is close to 1).
// norm_pdf: exp(-x^2 / 2) / sqrt(2 pi), as scipy.stats.norm._pdf.
// norm_ppf: inverse CDF. Acklam's rational approximation (relative error < 1.2e-9),
//   then one Halley step on norm_cdf(x) - p, which takes it to double precision.
// ---------------------------------------------------------------------------------
inline double norm_cdf(double a) {
    if (std::isnan(a)) return nan;
    const double sqrt1_2 = 0.7071067811865476;
    const double x = a * sqrt1_2;
    const double z = std::fabs(x);
    if (z < sqrt1_2) return 0.5 + 0.5 * std::erf(x);
    const double y = 0.5 * std::erfc(z);
    return x > 0 ? 1.0 - y : y;
}

inline double norm_sf(double x) { return norm_cdf(-x); }

inline double norm_pdf(double x) {
    return std::exp(-x * x / 2.0) / 2.5066282746310002;  // sqrt(2 pi)
}

inline double norm_ppf(double p) {
    if (std::isnan(p) || p < 0.0 || p > 1.0) return nan;
    if (p == 0.0) return -inf;
    if (p == 1.0) return inf;

    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    double x;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log1p(-p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    // One Halley step. In the upper tail, work with the survival function so the
    // residual isn't lost to cancellation against 1.
    double err;
    if (p > 0.5) {
        err = -(norm_sf(x) - (1.0 - p));
    } else {
        err = norm_cdf(x) - p;
    }
    const double u = err * 2.5066282746310002 * std::exp(x * x / 2.0);
    x = x - u / (1.0 + x * u / 2.0);
    return x;
}

// With loc and scale, exactly as scipy.stats.norm applies them, including scipy's rule
// that a non-positive (or NaN) scale gives NaN rather than a number.
inline double norm_cdf(double x, double loc, double scale) {
    if (!(scale > 0.0)) return nan;
    return norm_cdf((x - loc) / scale);
}

inline double norm_sf(double x, double loc, double scale) {
    if (!(scale > 0.0)) return nan;
    return norm_sf((x - loc) / scale);
}

inline double norm_pdf(double x, double loc, double scale) {
    if (!(scale > 0.0)) return nan;
    return norm_pdf((x - loc) / scale) / scale;
}

inline double norm_ppf(double q, double loc, double scale) {
    if (!(scale > 0.0)) return nan;
    return norm_ppf(q) * scale + loc;
}

}  // namespace qt
