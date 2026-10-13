"""Regenerate qtrans/runtime/qtrans_random.hpp from NumPy's source distribution.

    python tools/gen_random_header.py path/to/numpy-X.Y.Z   (an unpacked sdist)

Only the ziggurat tables are copied (verbatim, as decimal and hex literals); the algorithms
around them are ported by hand in the template below and documented there.
"""
import re
import sys
from pathlib import Path

TEMPLATE = r'''// qtrans_random.hpp: NumPy's random number generator, reproduced exactly.
//
//   Python:  rng = np.random.default_rng(seed)        C++:  qt::Rng rng(seed);
//            rng.standard_normal(n)                          rng.standard_normal(n)
//
// Same seed, same draws, bit for bit: SeedSequence seeding, the PCG64 generator, its
// 53-bit doubles, and the ziggurat method behind standard_normal / normal are ported from
// NumPy @@VERSION@@ (numpy/random: bit_generator.pyx, src/pcg64/pcg64.h,
// src/distributions/distributions.c, _generator.pyx). The ziggurat tables are NumPy's own,
// copied verbatim from src/distributions/ziggurat_constants.h under NumPy's licence:
//
@@LICENSE@@
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qtrans_arrays.hpp"

namespace qt {
namespace detail {

// ---- ziggurat tables (NumPy src/distributions/ziggurat_constants.h) ----------------
@@TABLES@@
inline constexpr double ziggurat_nor_r = 3.6541528853610087963519472518;
inline constexpr double ziggurat_nor_inv_r = 0.27366123732975827203338247596;

// ---- 128-bit unsigned arithmetic, portable (no __int128 needed, so MSVC works) ----
struct U128 {
    std::uint64_t hi, lo;
};

inline U128 mul64(std::uint64_t a, std::uint64_t b) {  // full 64 x 64 -> 128 product
    const std::uint64_t a_lo = a & 0xFFFFFFFFu, a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xFFFFFFFFu, b_hi = b >> 32;
    const std::uint64_t p0 = a_lo * b_lo, p1 = a_lo * b_hi, p2 = a_hi * b_lo, p3 = a_hi * b_hi;
    const std::uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFu) + (p2 & 0xFFFFFFFFu);
    return {p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32), (mid << 32) | (p0 & 0xFFFFFFFFu)};
}

inline U128 mul128(U128 a, U128 b) {  // mod 2^128
#if defined(__SIZEOF_INT128__)
    // native 128-bit type (GCC, Clang): same result, fewer instructions. __extension__ marks
    // the non-standard type as intended, so -Wpedantic stays quiet.
    __extension__ typedef unsigned __int128 native128;
    const native128 x = (static_cast<native128>(a.hi) << 64) | a.lo;
    const native128 y = (static_cast<native128>(b.hi) << 64) | b.lo;
    const native128 z = x * y;
    return {static_cast<std::uint64_t>(z >> 64), static_cast<std::uint64_t>(z)};
#else
    U128 r = mul64(a.lo, b.lo);
    r.hi += a.lo * b.hi + a.hi * b.lo;
    return r;
#endif
}

inline U128 add128(U128 a, U128 b) {
    U128 r{a.hi + b.hi, a.lo + b.lo};
    if (r.lo < a.lo) ++r.hi;
    return r;
}

// ---- SeedSequence (NumPy bit_generator.pyx) -------------------------------------------
inline std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= 0x931e8875u;  // MULT_A
    value *= hash_const;
    value ^= value >> 16;
    return value;
}

inline std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = 0xca01f9ddu * x - 0x4973f715u * y;  // MIX_MULT_L, MIX_MULT_R
    result ^= result >> 16;
    return result;
}

// SeedSequence(seed).generate_state(4, uint64) for a non-negative integer seed.
inline void seed_sequence_state(std::uint64_t seed, std::uint64_t out[4]) {
    std::vector<std::uint32_t> entropy;  // the seed as 32-bit words, least significant first
    if (seed == 0) entropy.push_back(0);
    for (std::uint64_t n = seed; n > 0; n >>= 32) entropy.push_back(static_cast<std::uint32_t>(n & 0xFFFFFFFFu));
    std::uint32_t pool[4];
    std::uint32_t hash_const = 0x43b0d7e5u;  // INIT_A
    for (std::size_t i = 0; i < 4; ++i)
        pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0u, hash_const);
    for (std::size_t src = 0; src < 4; ++src)
        for (std::size_t dst = 0; dst < 4; ++dst)
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hash_const));
    for (std::size_t src = 4; src < entropy.size(); ++src)
        for (std::size_t dst = 0; dst < 4; ++dst) pool[dst] = mix(pool[dst], hashmix(entropy[src], hash_const));
    std::uint32_t words[8];
    std::uint32_t h = 0x8b51f9ddu;  // INIT_B
    for (std::size_t i = 0; i < 8; ++i) {
        std::uint32_t v = pool[i % 4];
        v ^= h;
        h *= 0x58f38dedu;  // MULT_B
        v *= h;
        v ^= v >> 16;
        words[i] = v;
    }
    for (std::size_t k = 0; k < 4; ++k)  // pairs of 32-bit words, little-endian
        out[k] = static_cast<std::uint64_t>(words[2 * k]) | (static_cast<std::uint64_t>(words[2 * k + 1]) << 32);
}

}  // namespace detail

// np.random.Generator(PCG64(seed)), i.e. np.random.default_rng(seed).
class Rng {
public:
    // A placeholder with an all-zero state, only so an object holding a generator can be
    // declared before it is assigned; Python has no unseeded generators, so translated code
    // never draws from one.
    Rng() = default;

    explicit Rng(long long seed) {
        if (seed < 0) throw ValueError("expected non-negative integer");
        std::uint64_t v[4];
        detail::seed_sequence_state(static_cast<std::uint64_t>(seed), v);
        const detail::U128 initstate{v[0], v[1]}, initseq{v[2], v[3]};
        // pcg_setseq_128_srandom_r
        state_ = {0, 0};
        inc_ = {(initseq.hi << 1) | (initseq.lo >> 63), (initseq.lo << 1) | 1u};
        step();
        state_ = detail::add128(state_, initstate);
        step();
    }

    // The generator in exactly this PCG64 state: Generator.bit_generator.state["state"] and
    // ["inc"], split into high and low 64-bit halves. (How an object holding a generator that
    // has already drawn is handed to C++ mid-stream.)
    static Rng from_state(std::uint64_t state_hi, std::uint64_t state_lo, std::uint64_t inc_hi, std::uint64_t inc_lo) {
        Rng g(0);
        g.state_ = {state_hi, state_lo};
        g.inc_ = {inc_hi, inc_lo};
        return g;
    }

    // The current state, as from_state takes it: {state_hi, state_lo, inc_hi, inc_lo}.
    void state_words(std::uint64_t out[4]) const {
        out[0] = state_.hi;
        out[1] = state_.lo;
        out[2] = inc_.hi;
        out[3] = inc_.lo;
    }

    // Generator.bit_generator.random_raw(): the next raw 64-bit output.
    std::uint64_t next_uint64() {
        step();
        const std::uint64_t x = state_.hi ^ state_.lo;  // XSL-RR output function
        const unsigned rot = static_cast<unsigned>(state_.hi >> 58);
        return (x >> rot) | (x << ((64u - rot) & 63u));
    }

    // Generator.random(): uniform on [0, 1) with 53 random bits.
    double random() { return static_cast<double>(next_uint64() >> 11) * (1.0 / 9007199254740992.0); }

    // Generator.standard_normal(): NumPy's ziggurat, line for line.
    double standard_normal() {
        using namespace detail;
        for (;;) {
            std::uint64_t r = next_uint64();
            const int idx = static_cast<int>(r & 0xff);
            r >>= 8;
            const int sign = static_cast<int>(r & 0x1);
            const std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
            double x = static_cast<double>(rabs) * wi_double[idx];
            if (sign & 0x1) x = -x;
            if (rabs < ki_double[idx]) return x;  // 99.3% of draws stop here
            if (idx == 0) {
                for (;;) {
                    const double xx = -ziggurat_nor_inv_r * std::log1p(-random());
                    const double yy = -std::log1p(-random());
                    if (yy + yy > xx * xx)
                        return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
                }
            } else if ((fi_double[idx - 1] - fi_double[idx]) * random() + fi_double[idx] < std::exp(-0.5 * x * x)) {
                return x;
            }
        }
    }

    // Generator.normal(loc, scale): loc + scale * standard_normal().
    double normal(double loc, double scale) {
        check_scale(scale);
        return loc + scale * standard_normal();
    }

    // Generator.uniform(low, high): low + (high - low) * random().
    double uniform(double low, double high) {
        const double range = check_range(low, high);
        return low + range * random();
    }

    // The same, filling arrays in row-major order, exactly as NumPy fills them.
    Vec standard_normal(long long n) { return fill(n, [this] { return standard_normal(); }); }
    Mat standard_normal(long long r, long long c) { return fill(r, c, [this] { return standard_normal(); }); }
    Vec random(long long n) { return fill(n, [this] { return random(); }); }
    Mat random(long long r, long long c) { return fill(r, c, [this] { return random(); }); }
    Vec normal(double loc, double scale, long long n) {
        check_scale(scale);
        return fill(n, [&] { return loc + scale * standard_normal(); });
    }
    Mat normal(double loc, double scale, long long r, long long c) {
        check_scale(scale);
        return fill(r, c, [&] { return loc + scale * standard_normal(); });
    }
    Vec uniform(double low, double high, long long n) {
        const double range = check_range(low, high);
        return fill(n, [&] { return low + range * random(); });
    }
    Mat uniform(double low, double high, long long r, long long c) {
        const double range = check_range(low, high);
        return fill(r, c, [&] { return low + range * random(); });
    }

private:
    detail::U128 state_{0, 0}, inc_{0, 0};

    void step() {  // state = state * multiplier + inc  (mod 2^128)
        static constexpr detail::U128 multiplier{2549297995355413924ULL, 4865540595714422341ULL};
        state_ = detail::add128(detail::mul128(state_, multiplier), inc_);
    }

    static void check_scale(double scale) {
        if (!std::isnan(scale) && std::signbit(scale)) throw ValueError("scale < 0");
    }

    static double check_range(double low, double high) {
        const double range = high - low;
        if (!std::isfinite(range)) throw OverflowError("high - low range exceeds valid bounds");
        if (std::signbit(range)) throw ValueError("high - low < 0");
        return range;
    }

    template <class F>
    static Vec fill(long long n, F draw) {
        check_dim(n);
        Vec out(n);
        for (Index i = 0; i < n; ++i) out(i) = draw();
        return out;
    }

    template <class F>
    static Mat fill(long long r, long long c, F draw) {
        check_dim(r);
        check_dim(c);
        Mat out(r, c);
        for (Index i = 0; i < r; ++i)
            for (Index j = 0; j < c; ++j) out(i, j) = draw();
        return out;
    }
};

}  // namespace qt
'''


def main(sdist):
    root = Path(sdist)
    src = (root / "numpy/random/src/distributions/ziggurat_constants.h").read_text()
    version = root.name.split("-")[-1]
    tables = []
    for name in ("ki_double", "wi_double", "fi_double"):
        m = re.search(r"static const (\w+) " + name + r"\[\] = \{(.*?)\};", src, re.S)
        ctype = "std::uint64_t" if m.group(1) == "uint64_t" else "double"
        vals = [v.strip() for v in m.group(2).replace("\n", " ").split(",") if v.strip()]
        assert len(vals) == 256, name
        rows = [", ".join(vals[i:i + 4]) for i in range(0, 256, 4)]
        tables.append(f"inline constexpr {ctype} {name}[256] = {{\n    " + ",\n    ".join(rows) + "};")
    licence = "\n".join(("// " + line).rstrip() for line in (root / "LICENSE.txt").read_text().splitlines()[:30])
    out = TEMPLATE.replace("@@TABLES@@", "\n".join(tables)).replace("@@VERSION@@", version) \
                  .replace("@@LICENSE@@", licence)
    dest = Path(__file__).resolve().parent.parent / "qtrans" / "runtime" / "qtrans_random.hpp"
    dest.write_text(out)
    print(f"wrote {dest} from NumPy {version}")


if __name__ == "__main__":
    main(sys.argv[1])
