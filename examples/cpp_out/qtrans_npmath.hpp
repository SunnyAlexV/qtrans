// qtrans_npmath.hpp: NumPy's float64 maths (np.exp, np.log, np.power, ...) computed as NumPy computes
// them on this machine, for the NumPy release named by QTRANS_NUMPY_VERSION (qtrans_numpy.hpp).
//
// NumPy does not call the C library for these everywhere. Which code it runs depends on the
// platform, the CPU and its version (measured on NumPy 1.21 to 2.5; docs/SPEC.md has the table):
//
//   - Linux x86-64 with AVX-512 (x86-64-v4): Intel's SVML routines for 19 functions, the
//     high-accuracy ones from NumPy 2.0, the low-accuracy ones in 1.22 to 1.26
//     (qtrans_svml_ha.hpp / qtrans_svml_la.hpp: NumPy's own assembly);
//   - AVX512F without SVML (macOS and Linux x86-64 for NumPy before 1.23): NumPy's exp and log by
//     Tang's method (qtrans_npy_own.hpp: NumPy's own C);
//   - tanh, from NumPy 1.23, on any CPU with fused multiply-add SIMD (AVX2+FMA, AVX-512, ARM64):
//     NumPy's own kernel, ported below;
//   - otherwise the C library, as NumPy does (Windows, CPUs without AVX-512, other functions).
//
// The choice is made once, at the first call, from the CPU (and NumPy's own switches
// NPY_DISABLE_CPU_FEATURES / NPY_ENABLE_CPU_FEATURES, which a C++ program honours as NumPy does).
// Define QTRANS_NO_SVML to always use the C library where NumPy would call SVML.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "qtrans_numpy.hpp"
#include "qtrans_npy_own.hpp"

#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && !defined(QTRANS_NO_SVML)
#define QT_NPM_SVML 1
#if QTRANS_NUMPY_VERSION >= 20000
#include "qtrans_svml_ha.hpp"
#define QT_SVML_NS ::qt::svml_ha
#else
#include "qtrans_svml_la.hpp"
#define QT_SVML_NS ::qt::svml_la
#endif
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#define QT_NPM_X86 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif
#endif

namespace qt {
namespace npm {

// the functions NumPy may compute with its own code
enum class Fn : unsigned char {
    exp, log, exp2, log2, log10, expm1, log1p, cbrt, tan, arcsin, arccos, arctan, sinh, cosh, tanh,
    arcsinh, arccosh, arctanh, sin, cos, power, arctan2, count_
};
enum class Path : unsigned char { libm, svml, tang, own };

constexpr long numpy_version = QTRANS_NUMPY_VERSION;  // e.g. 20504 for NumPy 2.5.4

// --- the CPU, as NumPy's dispatcher sees it ------------------------------------------------------
struct Cpu {
    bool avx512f = false;  // AVX512F
    bool v4 = false;       // x86-64-v4 (AVX512 F, CD, BW, DQ, VL): NumPy's X86_V4 / AVX512_SKX
    bool fma_simd = false; // a SIMD target with native FMA: x86-64 AVX2 + FMA3, or ARM64
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, at the first call
#endif
inline const char* env(const char* var) { return std::getenv(var); }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

inline bool env_names(const char* var, const char* const* names) {
    const char* s = env(var);
    if (s == nullptr) return false;
    for (const char* const* n = names; *n; ++n) {
        const std::size_t len = std::strlen(*n);
        for (const char* p = s; (p = std::strstr(p, *n)) != nullptr; p += len) {
            const bool start = p == s || p[-1] == ' ' || p[-1] == ',' || p[-1] == '\t';
            const char e = p[len];
            if (start && (e == '\0' || e == ' ' || e == ',' || e == '\t')) return true;
        }
    }
    return false;
}

inline Cpu detect_cpu() {
    Cpu c;
#if defined(QT_NPM_X86) && !defined(_MSC_VER)
    __builtin_cpu_init();
    c.avx512f = __builtin_cpu_supports("avx512f");
    c.v4 = c.avx512f && __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
    c.fma_simd = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
                 __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2");
#elif defined(QT_NPM_X86) && defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    const int top = r[0];
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, f16c = (r[2] >> 29) & 1;
    unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    bool avx2 = false, bmi = false, bmi2 = false;
    if (top >= 7) {
        __cpuidex(r, 7, 0);
        avx2 = (r[1] >> 5) & 1; bmi = (r[1] >> 3) & 1; bmi2 = (r[1] >> 8) & 1;
        const bool os512 = (xcr0 & 0xe6) == 0xe6;
        c.avx512f = os512 && ((r[1] >> 16) & 1);
        c.v4 = c.avx512f && ((r[1] >> 28) & 1) && ((r[1] >> 30) & 1) && ((r[1] >> 17) & 1) && ((r[1] >> 31) & 1);
    }
    c.fma_simd = avx && ((xcr0 & 6) == 6) && avx2 && fma && f16c && bmi && bmi2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    c.fma_simd = true;  // NEON with FMA is the ARM64 baseline
#endif
    // NumPy's switches, read as that release reads them (checked against numpy.lib.introspect):
    // from 2.4 its x86 features are the levels X86_V3 and X86_V4; before, the single features
    // (AVX512F, AVX2, ...), where leaving out AVX512_SKX alone changes nothing. An enable list
    // turns a level on only with everything below it listed too.
    const bool levels = QTRANS_NUMPY_VERSION >= 20400;
#if !defined(__aarch64__) && !defined(_M_ARM64)
    static const char* const v4_off_new[] = {"X86_V4", nullptr};
    static const char* const v3_off_new[] = {"X86_V3", nullptr};
    static const char* const f_off_old[] = {"AVX512F", nullptr};
    static const char* const v3_off_old[] = {"AVX2", "FMA3", "F16C", "AVX", nullptr};
    if (env_names("NPY_DISABLE_CPU_FEATURES", levels ? v3_off_new : v3_off_old)) c.avx512f = c.v4 = c.fma_simd = false;
    if (env_names("NPY_DISABLE_CPU_FEATURES", levels ? v4_off_new : f_off_old)) c.avx512f = c.v4 = false;
    const char* on = env("NPY_ENABLE_CPU_FEATURES");
    if (on != nullptr && *on != '\0') {
        static const char* const sse[] = {"SSSE3", "SSE41", "POPCNT", "SSE42", nullptr};
        auto all = [](std::initializer_list<const char*> names) {
            for (const char* n : names) {
                const char* one[] = {n, nullptr};
                if (!env_names("NPY_ENABLE_CPU_FEATURES", one)) return false;
            }
            return true;
        };
        bool v3, f, v4;
        if (levels) {
            v3 = all({"X86_V3"});
            f = v4 = v3 && all({"X86_V4"});
        } else {
            const bool base = all({sse[0], sse[1], sse[2], sse[3]});
            v3 = base && all({"AVX", "F16C", "FMA3", "AVX2"});
            f = v3 && all({"AVX512F"});
            v4 = f && all({"AVX512CD", "AVX512_SKX"});
        }
        c.fma_simd = c.fma_simd && v3;
        c.avx512f = c.avx512f && f;
        c.v4 = c.v4 && v4;
    }
#endif
    return c;
}

inline const Cpu& cpu() {
    static const Cpu c = detect_cpu();
    return c;
}

// --- which code NumPy runs, by function and version --------------------------------------------
// The float64 functions NumPy (2.0 and later) sends to SVML on Linux x86-64 with AVX-512
inline bool svml_ha_function(Fn f) { return f != Fn::tanh && f != Fn::sin && f != Fn::cos; }

// ... and NumPy 1.22 to 1.26, which used the low-accuracy routines and changed its mind on a few
inline bool svml_la_function(Fn f, long v) {
    if (v < 12200) return false;
    switch (f) {
        case Fn::exp: case Fn::log: return v >= 12300;          // 1.22 used its own exp / log
        case Fn::sin: case Fn::cos: return v < 12500;           // 1.22 to 1.24 only
        case Fn::power: case Fn::arctan2: return v >= 12400;    // from 1.24
        case Fn::tanh: return v < 12300;                        // 1.22; then NumPy's own kernel
        default: return true;
    }
}

inline Path choose(Fn f, const Cpu& c, long v) {
#if defined(QT_NPM_SVML)
    if (c.v4 && (v >= 20000 ? svml_ha_function(f) : svml_la_function(f, v))) return Path::svml;
#endif
#if defined(QT_NPY_OWN_TANG)
    if ((f == Fn::exp || f == Fn::log) && c.avx512f) {
#if defined(__linux__)
        // NumPy built with SVML runs Tang's kernels where it does not call SVML for exp / log; from
        // 2.4 it builds its x86 kernels for x86-64-v4 and v3 only, so an AVX512F-only CPU has none
        if ((v < 12300 || !c.v4) && v < 20400) return Path::tang;
#else
        if (v < 20400 || c.v4) return Path::tang;
#endif
    }
#endif
    if (f == Fn::tanh && v >= 12300 && (c.fma_simd || c.v4)) return Path::own;
    return Path::libm;
}

inline Path path(Fn f) {
    static const auto table = [] {
        struct T { Path p[static_cast<int>(Fn::count_)]; } t{};
        for (int i = 0; i < static_cast<int>(Fn::count_); ++i)
            t.p[i] = choose(static_cast<Fn>(i), cpu(), numpy_version);
        return t;
    }();
    return table.p[static_cast<int>(f)];
}

// --- the C library: NumPy's npy_exp, npy_log, ... -----------------------------------------------
// NumPy's Linux wheels are built against an old glibc, so they call exp, exp2, log, log2 and pow at
// their old symbol versions (exp@GLIBC_2.2.5, ...), whose error handling gives +NaN where today's
// versions give -NaN (np.log(-1.0)). A program built now would get the new ones; these bind the
// old ones, as NumPy does. Define QTRANS_NO_GLIBC_COMPAT to use the default ones (a static build).
}  // namespace npm
}  // namespace qt
#if defined(__linux__) && defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(QTRANS_NO_GLIBC_COMPAT)
#define QT_NPM_GLIBC_COMPAT 1
#if defined(__x86_64__)
#define QT_NPM_GLIBC_OLD "GLIBC_2.2.5"
#else
#define QT_NPM_GLIBC_OLD "GLIBC_2.17"
#endif
extern "C" {
double qt_npy_glibc_exp(double);
double qt_npy_glibc_exp2(double);
double qt_npy_glibc_log(double);
double qt_npy_glibc_log2(double);
double qt_npy_glibc_pow(double, double);
}
__asm__(".symver qt_npy_glibc_exp, exp@" QT_NPM_GLIBC_OLD "\n\t"
        ".symver qt_npy_glibc_exp2, exp2@" QT_NPM_GLIBC_OLD "\n\t"
        ".symver qt_npy_glibc_log, log@" QT_NPM_GLIBC_OLD "\n\t"
        ".symver qt_npy_glibc_log2, log2@" QT_NPM_GLIBC_OLD "\n\t"
        ".symver qt_npy_glibc_pow, pow@" QT_NPM_GLIBC_OLD);
#endif
namespace qt {
namespace npm {

inline double npy_exp(double x) {
#if defined(QT_NPM_GLIBC_COMPAT)
    return qt_npy_glibc_exp(x);
#else
    return std::exp(x);
#endif
}
inline double npy_exp2(double x) {
#if defined(QT_NPM_GLIBC_COMPAT)
    return qt_npy_glibc_exp2(x);
#else
    return std::exp2(x);
#endif
}
inline double npy_log(double x) {
#if defined(QT_NPM_GLIBC_COMPAT)
    return qt_npy_glibc_log(x);
#else
    return std::log(x);
#endif
}
inline double npy_log2(double x) {
#if defined(QT_NPM_GLIBC_COMPAT)
    return qt_npy_glibc_log2(x);
#else
    return std::log2(x);
#endif
}
inline double npy_pow(double x, double y) {
#if defined(QT_NPM_GLIBC_COMPAT)
    return qt_npy_glibc_pow(x, y);
#else
    return std::pow(x, y);
#endif
}

inline double libm1(Fn f, double x) {
    switch (f) {
        case Fn::exp: return npy_exp(x);
        case Fn::log: return npy_log(x);
        case Fn::exp2: return npy_exp2(x);
        case Fn::log2: return npy_log2(x);
        case Fn::log10: return std::log10(x);
        case Fn::expm1: return std::expm1(x);
        case Fn::log1p: return std::log1p(x);
        case Fn::cbrt: return std::cbrt(x);
        case Fn::tan: return std::tan(x);
        case Fn::arcsin: return std::asin(x);
        case Fn::arccos: return std::acos(x);
        case Fn::arctan: return std::atan(x);
        case Fn::sinh: return std::sinh(x);
        case Fn::cosh: return std::cosh(x);
        case Fn::tanh: return std::tanh(x);
        case Fn::arcsinh: return std::asinh(x);
        case Fn::arccosh: return std::acosh(x);
        case Fn::arctanh: return std::atanh(x);
        case Fn::sin: return std::sin(x);
        case Fn::cos: return std::cos(x);
        default: return std::nan("");
    }
}

// --- NumPy's tanh (numpy/_core/src/umath/loops_hyperbolic.dispatch.cpp.src), one lane at a time --
// tanh(|x|) on 16 intervals, a degree-16 polynomial in |x| - b on each, every step a fused
// multiply-add, as NumPy's SIMD code does it in each lane.
#if defined(QT_NPM_X86) && (defined(__GNUC__) || defined(__clang__))
#define QT_NPM_FMA_TARGET __attribute__((target("fma")))
QT_NPM_FMA_TARGET inline double fma_hw(double a, double b, double c) { return __builtin_fma(a, b, c); }
#elif defined(QT_NPM_X86) && defined(_MSC_VER)
#define QT_NPM_FMA_TARGET
inline double fma_hw(double a, double b, double c) {
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
}
#else
#define QT_NPM_FMA_TARGET
inline double fma_hw(double a, double b, double c) { return std::fma(a, b, c); }  // ARM64: one instruction
#endif

QT_NPM_FMA_TARGET inline double numpy_tanh(double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, 8);
    const std::uint64_t ndnan = bits & 0x7ff8000000000000ull;
    const bool in_range = ndnan <= 0x7fe0000000000000ull;  // |x| below the huge threshold, not inf / NaN
    // the interval: the top 32 bits of (ndnan - 0x3fc0000000000000), clamped to [0, 0x780000], >> 19
    std::int32_t hi = static_cast<std::int32_t>(static_cast<std::uint32_t>(ndnan >> 32) - 0x3fc00000u);
    hi = hi < 0 ? 0 : (hi > 0x780000 ? 0x780000 : hi);
    const unsigned idx = static_cast<unsigned>(hi) >> 19;
    auto coef = [idx](int row) {
        double d;
        std::memcpy(&d, &npy_own::tanh_lut16x18[row * 16 + static_cast<int>(idx)], 8);
        return d;
    };
    double ax;
    const std::uint64_t abits = bits & 0x7fffffffffffffffull;
    std::memcpy(&ax, &abits, 8);
    const double y = ax - coef(0);
    double r = fma_hw(coef(17), y, coef(16));
    for (int k = 15; k >= 1; --k) r = fma_hw(r, y, coef(k));
    if (!in_range) r = 1.0;
    std::uint64_t rb;
    std::memcpy(&rb, &r, 8);
    rb |= bits & 0x8000000000000000ull;
    if (x != x) rb = 0x7ff8000000000000ull;  // NumPy's NaN
    std::memcpy(&r, &rb, 8);
    return r;
}

// --- SVML: eight lanes at a time, the tail masked, as NumPy's loops do -------------------------
#if defined(QT_NPM_SVML)
// what NumPy loads into the lanes past the end (they are computed and thrown away)
inline double svml_fill(Fn f) {
    switch (f) {
        case Fn::log: case Fn::log2: case Fn::log10: case Fn::expm1: case Fn::arccosh: case Fn::power:
        case Fn::arctan2: return 1.0;
        default: return 0.0;
    }
}

template <__m512d (*K)(__m512d)>
QT_SVML_TARGET void svml_unary(double* y, const double* x, std::ptrdiff_t n, double fill, bool keep_nan) {
    const __m512d vfill = _mm512_set1_pd(fill);
    for (std::ptrdiff_t i = 0; i < n; i += 8) {
        const std::ptrdiff_t k = n - i < 8 ? n - i : 8;
        const __mmask8 m = static_cast<__mmask8>((1u << k) - 1u);
        const __m512d v = _mm512_mask_loadu_pd(vfill, m, x + i);
        __m512d r;
        if (keep_nan) {  // NumPy >= 1.24, expm1: NaN lanes are given 1.0, and keep their NaN
            const __mmask8 nan = _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q);
            r = _mm512_mask_blend_pd(nan, K(_mm512_mask_blend_pd(nan, v, _mm512_set1_pd(1.0))), v);
        } else {
            r = K(v);
        }
        _mm512_mask_storeu_pd(y + i, m, r);
    }
}

template <__m512d (*K)(__m512d, __m512d)>
QT_SVML_TARGET void svml_binary(double* y, const double* a, const double* b, std::ptrdiff_t n) {
    const __m512d one = _mm512_set1_pd(1.0);
    for (std::ptrdiff_t i = 0; i < n; i += 8) {
        const std::ptrdiff_t k = n - i < 8 ? n - i : 8;
        const __mmask8 m = static_cast<__mmask8>((1u << k) - 1u);
        _mm512_mask_storeu_pd(y + i, m, K(_mm512_mask_loadu_pd(one, m, a + i), _mm512_mask_loadu_pd(one, m, b + i)));
    }
}

inline void svml1(Fn f, double* y, const double* x, std::ptrdiff_t n) {
    const double fill = svml_fill(f);
    switch (f) {
        case Fn::exp: return svml_unary<QT_SVML_NS::exp>(y, x, n, fill, false);
        case Fn::log: return svml_unary<QT_SVML_NS::log>(y, x, n, fill, false);
        case Fn::exp2: return svml_unary<QT_SVML_NS::exp2>(y, x, n, fill, false);
        case Fn::log2: return svml_unary<QT_SVML_NS::log2>(y, x, n, fill, false);
        case Fn::log10: return svml_unary<QT_SVML_NS::log10>(y, x, n, fill, false);
        case Fn::expm1: return svml_unary<QT_SVML_NS::expm1>(y, x, n, fill, numpy_version >= 12400);
        case Fn::log1p: return svml_unary<QT_SVML_NS::log1p>(y, x, n, fill, false);
        case Fn::cbrt: return svml_unary<QT_SVML_NS::cbrt>(y, x, n, fill, false);
        case Fn::tan: return svml_unary<QT_SVML_NS::tan>(y, x, n, fill, false);
        case Fn::arcsin: return svml_unary<QT_SVML_NS::asin>(y, x, n, fill, false);
        case Fn::arccos: return svml_unary<QT_SVML_NS::acos>(y, x, n, fill, false);
        case Fn::arctan: return svml_unary<QT_SVML_NS::atan>(y, x, n, fill, false);
        case Fn::sinh: return svml_unary<QT_SVML_NS::sinh>(y, x, n, fill, false);
        case Fn::cosh: return svml_unary<QT_SVML_NS::cosh>(y, x, n, fill, false);
        case Fn::arcsinh: return svml_unary<QT_SVML_NS::asinh>(y, x, n, fill, false);
        case Fn::arccosh: return svml_unary<QT_SVML_NS::acosh>(y, x, n, fill, false);
        case Fn::arctanh: return svml_unary<QT_SVML_NS::atanh>(y, x, n, fill, false);
#if QTRANS_NUMPY_VERSION < 20000
        case Fn::tanh: return svml_unary<QT_SVML_NS::tanh>(y, x, n, fill, false);
        case Fn::sin: return svml_unary<QT_SVML_NS::sin>(y, x, n, fill, false);
        case Fn::cos: return svml_unary<QT_SVML_NS::cos>(y, x, n, fill, false);
#endif
        default: for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = libm1(f, x[i]);
    }
}
#endif

// --- one function over n values: y[i] = f(x[i]) (y may be x) ------------------------------------
inline void apply(Fn f, double* y, const double* x, std::ptrdiff_t n) {
    switch (path(f)) {
#if defined(QT_NPM_SVML)
        case Path::svml: return svml1(f, y, x, n);
#endif
#if defined(QT_NPY_OWN_TANG)
        case Path::tang:
            if (f == Fn::exp) npy_own::AVX512F_exp_DOUBLE(y, const_cast<double*>(x), n, static_cast<std::ptrdiff_t>(sizeof(double)));
            else npy_own::AVX512F_log_DOUBLE(y, const_cast<double*>(x), n, static_cast<std::ptrdiff_t>(sizeof(double)));
            return;
#endif
        case Path::own:
            for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = numpy_tanh(x[i]);
            return;
        default:
            for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = libm1(f, x[i]);
    }
}

// --- np.power and np.arctan2 over pairs --------------------------------------------------------
inline void apply2(Fn f, double* y, const double* a, const double* b, std::ptrdiff_t n) {
#if defined(QT_NPM_SVML)
    if (path(f) == Path::svml) {
        if (f == Fn::power) return svml_binary<QT_SVML_NS::pow>(y, a, b, n);
        return svml_binary<QT_SVML_NS::atan2>(y, a, b, n);
    }
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = f == Fn::power ? npy_pow(a[i], b[i]) : std::atan2(a[i], b[i]);
}

// The shortcuts NumPy's power loop takes for a scalar exponent (np.power(a, s); none before 2.1,
// x*x from 2.1, and 1/x, 1, sqrt(x), x from 2.3): true, with the result in r, when one applies.
inline bool power_loop_shortcut(double x, double s, double& r) {
    if (numpy_version >= 20300) {
        if (s == -1.0) { r = 1.0 / x; return true; }
        if (s == 0.0) { r = 1.0; return true; }
        if (s == 0.5) { r = std::sqrt(x); return true; }
        if (s == 1.0) { r = x; return true; }
    }
    if (numpy_version >= 20100 && s == 2.0) { r = x * x; return true; }
    return false;
}

// The shortcuts of the ** operator on an array (NumPy's fast_scalar_power, and from 2.3 the
// loop's): the same results in every NumPy release
inline bool power_operator_shortcut(double x, double s, double& r) {
    if (s == -1.0) { r = 1.0 / x; return true; }
    if (s == 0.0) { r = 1.0; return true; }
    if (s == 0.5) { r = std::sqrt(x); return true; }
    if (s == 1.0) { r = x; return true; }
    if (s == 2.0) { r = x * x; return true; }
    return false;
}

// y[i] = a[i] ** s, NumPy's way: `op` for the ** operator, else np.power
inline void power_scalar(double* y, const double* a, double s, std::ptrdiff_t n, bool op) {
    double r;
    if (n > 0 && (op ? power_operator_shortcut(a[0], s, r) : power_loop_shortcut(a[0], s, r))) {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            if (op) power_operator_shortcut(a[i], s, y[i]);
            else power_loop_shortcut(a[i], s, y[i]);
        }
        return;
    }
#if defined(QT_NPM_SVML)
    if (path(Fn::power) == Path::svml) {
        const double block[8] = {s, s, s, s, s, s, s, s};
        for (std::ptrdiff_t i = 0; i < n; i += 8)
            svml_binary<QT_SVML_NS::pow>(y + i, a + i, block, n - i < 8 ? n - i : 8);
        return;
    }
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = npy_pow(a[i], s);
}

// y[i] = b ** a[i]: a scalar base takes no shortcut
inline void power_base(double* y, double b, const double* a, std::ptrdiff_t n) {
#if defined(QT_NPM_SVML)
    if (path(Fn::power) == Path::svml) {
        const double block[8] = {b, b, b, b, b, b, b, b};
        for (std::ptrdiff_t i = 0; i < n; i += 8)
            svml_binary<QT_SVML_NS::pow>(y + i, block, a + i, n - i < 8 ? n - i : 8);
        return;
    }
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = npy_pow(b, a[i]);
}

// --- the scalar forms: np.exp(x) on a number is the same loop, run on one value -----------------
#define QT_NPM_SCALAR(NAME)                                   \
    inline double NAME(double x) {                            \
        double y;                                             \
        apply(Fn::NAME, &y, &x, 1);                           \
        return y;                                             \
    }
QT_NPM_SCALAR(exp)
QT_NPM_SCALAR(log)
QT_NPM_SCALAR(exp2)
QT_NPM_SCALAR(log2)
QT_NPM_SCALAR(log10)
QT_NPM_SCALAR(expm1)
QT_NPM_SCALAR(log1p)
QT_NPM_SCALAR(cbrt)
QT_NPM_SCALAR(tan)
QT_NPM_SCALAR(arcsin)
QT_NPM_SCALAR(arccos)
QT_NPM_SCALAR(arctan)
QT_NPM_SCALAR(sinh)
QT_NPM_SCALAR(cosh)
QT_NPM_SCALAR(tanh)
QT_NPM_SCALAR(arcsinh)
QT_NPM_SCALAR(arccosh)
QT_NPM_SCALAR(arctanh)
QT_NPM_SCALAR(sin)
QT_NPM_SCALAR(cos)
#undef QT_NPM_SCALAR

// np.power(x, s) on numbers (the loop's shortcuts apply: both arguments are scalars)
inline double power(double x, double s) {
    double y;
    power_scalar(&y, &x, s, 1, false);
    return y;
}

// one pair of a broadcast np.power whose exponent varies along the loop: the kernel, no shortcut
inline double power_pair(double x, double y) {
    double r;
    apply2(Fn::power, &r, &x, &y, 1);
    return r;
}

inline double arctan2(double a, double b) {
    double y;
    apply2(Fn::arctan2, &y, &a, &b, 1);
    return y;
}

}  // namespace npm

// SciPy's norm.pdf (scipy.stats.norm._pdf): np.exp(-x**2/2.0) / sqrt(2 pi), NumPy's exp included
inline double norm_pdf(double x) {
    return npm::exp(-x * x / 2.0) / 2.5066282746310002;  // sqrt(2 pi)
}
inline double norm_pdf(double x, double loc, double scale) {
    if (!(scale > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return norm_pdf((x - loc) / scale) / scale;
}

namespace npm {

// which code each function runs here, for `qtrans doctor` and the tests
inline const char* path_name(Fn f) {
    switch (path(f)) {
        case Path::svml: return numpy_version >= 20000 ? "svml_ha" : "svml_la";
        case Path::tang: return "numpy_tang";
        case Path::own: return "numpy_own";
        default: return "libm";
    }
}

}  // namespace npm
}  // namespace qt
