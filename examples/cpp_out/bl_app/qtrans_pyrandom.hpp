// qtrans_pyrandom.hpp: Python's random module, reproduced exactly (0.14).
//
// CPython's generator is the Mersenne Twister MT19937, seeded from an int by init_by_array with
// the int's 32-bit words; random() joins 27 and 26 bits of two outputs into a double; randint,
// randrange, choice, shuffle and sample draw an int below n by asking for n.bit_length() bits
// until one is below n; gauss, normalvariate, expovariate, uniform and triangular are the
// formulas of Lib/random.py, operation for operation, with the C library's cos, sin, log, sqrt
// and exp as Python calls them. So a program that seeds the generator gets Python's numbers,
// in Python's order. The error messages are the running Python's (qtrans_python.hpp).
//
//     random.seed(42)            ->  qt::py::random_module().seed(42)
//     random.gauss(0.0, 1.0)     ->  qt::py::random_module().gauss(0.0, 1.0)
//     rng = random.Random(7)     ->  qt::py::Random rng(7);
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "qtrans_runtime.hpp"
#include "qtrans_python.hpp"

namespace qt {
namespace py {

class Random {
public:
    Random() { seed(0); }
    explicit Random(long long a) { seed(a); }

    // random.seed(a) for an int a: init_by_array with the 32-bit words of abs(a), lowest first
    void seed(long long a) {
        unsigned long long n = a < 0 ? 0ULL - static_cast<unsigned long long>(a) : static_cast<unsigned long long>(a);
        std::uint32_t key[2] = {static_cast<std::uint32_t>(n & 0xffffffffULL), static_cast<std::uint32_t>(n >> 32)};
        init_by_array(key, (n >> 32) ? 2 : 1);
        gauss_next_.reset();
    }

    // random(): [0, 1) with 53 random bits
    double random() {
        const std::uint32_t a = genrand() >> 5, b = genrand() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // getrandbits(k) for 0 <= k <= 64: the words come lowest first, the last one cut to size
    unsigned long long getrandbits(int k) {
        if (k < 0) throw ValueError("number of bits must be non-negative");
        if (k == 0) return 0;
        if (k <= 32) return genrand() >> (32 - k);
        unsigned long long lo = genrand();
        unsigned long long hi = genrand() >> (64 - k);
        return lo | (hi << 32);
    }

    // _randbelow(n): k = n.bit_length() bits, drawn until below n (n > 0)
    long long randbelow(long long n) {
        if (n <= 0) return 0;  // (3.10's own rule; later versions never ask for n == 0)
        int k = 0;
        for (unsigned long long v = static_cast<unsigned long long>(n); v; v >>= 1) ++k;
        unsigned long long r = getrandbits(k);
        while (r >= static_cast<unsigned long long>(n)) r = getrandbits(k);
        return static_cast<long long>(r);
    }

    long long randrange(long long stop) {
        if (stop > 0) return randbelow(stop);
        throw ValueError(msg::random_randrange_empty);
    }

    long long randrange(long long start, long long stop) {
        const long long width = stop - start;
        if (width > 0) return start + randbelow(width);
        throw ValueError(range_text(msg::random_randrange_two, msg::random_randrange_two_form == 0
                                       ? std::vector<long long>{start, stop, width} : std::vector<long long>{start, stop}));
    }

    long long randrange(long long start, long long stop, long long step) {
        const long long width = stop - start;
        if (step == 1) return randrange(start, stop);
        long long n;
        if (step > 0) n = qt::floordiv(width + step - 1, step);
        else if (step < 0) n = qt::floordiv(width + step + 1, step);
        else throw ValueError(msg::random_zero_step);
        if (n <= 0)
            throw ValueError(msg::random_randrange_three_form == 0 ? std::string(msg::random_randrange_three)
                                                                   : range_text(msg::random_randrange_three, {start, stop, step}));
        return start + step * randbelow(n);
    }

    long long randint(long long a, long long b) {
        if (msg::random_randint_form == 2) {  // 3.14: its own check and words
            if (b < a) throw ValueError(range_text(msg::random_randint, {a, b}));
            return a + randbelow(b - a + 1);
        }
        return randrange(a, b + 1);
    }

    double uniform(double a, double b) { return a + (b - a) * random(); }

    double triangular(double low = 0.0, double high = 1.0) { return triangular_(low, high, 0.5); }
    double triangular(double low, double high, double mode) {
        const double u0 = random();
        if (high - low == 0.0) return low;  // (mode - low) / (high - low): ZeroDivisionError, caught
        return finish_triangular(u0, low, high, (mode - low) / (high - low));
    }

    double gauss(double mu = 0.0, double sigma = 1.0) {
        double z;
        if (gauss_next_) {
            z = *gauss_next_;
            gauss_next_.reset();
        } else {
            const double x2pi = random() * twopi();
            const double g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
            z = std::cos(x2pi) * g2rad;
            gauss_next_ = std::sin(x2pi) * g2rad;
        }
        return mu + z * sigma;
    }

    double normalvariate(double mu = 0.0, double sigma = 1.0) {
        const double magic = nv_magicconst();
        double z;
        while (true) {
            const double u1 = random();
            const double u2 = 1.0 - random();
            z = magic * (u1 - 0.5) / u2;
            const double zz = z * z / 4.0;
            if (zz <= -std::log(u2)) break;
        }
        return mu + z * sigma;
    }

    double lognormvariate(double mu, double sigma) { return std::exp(normalvariate(mu, sigma)); }

    double expovariate(double lambd = 1.0) {
        const double x = -std::log(1.0 - random());
        if (lambd == 0.0) throw ZeroDivisionError(msg::float_truediv_zero);
        return x / lambd;
    }

    template <class T>
    T choice(const std::vector<T>& seq) {
        if (seq.empty()) throw IndexError(msg::random_choice_empty);
        return seq[static_cast<std::size_t>(randbelow(static_cast<long long>(seq.size())))];
    }

    template <class T>
    void shuffle(std::vector<T>& x) {
        for (std::size_t i = x.size(); i-- > 1;) {
            const std::size_t j = static_cast<std::size_t>(randbelow(static_cast<long long>(i) + 1));
            std::swap(x[i], x[j]);
        }
    }

    template <class T>
    std::vector<T> sample(const std::vector<T>& population, long long k) {
        const long long n = static_cast<long long>(population.size());
        if (!(0 <= k && k <= n)) throw ValueError(msg::random_sample_size);
        std::vector<T> result;
        result.reserve(static_cast<std::size_t>(k));
        long long setsize = 21;
        if (k > 5) {  // 4 ** ceil(log(k * 3, 4)): math.log with a base, as Python works it out
            const double e = std::ceil(std::log(static_cast<double>(k * 3)) / log4());
            setsize += static_cast<long long>(std::pow(4.0, e));
        }
        if (n <= setsize) {
            std::vector<T> pool(population);
            for (long long i = 0; i < k; ++i) {
                const long long j = randbelow(n - i);
                result.push_back(pool[static_cast<std::size_t>(j)]);
                pool[static_cast<std::size_t>(j)] = pool[static_cast<std::size_t>(n - i - 1)];
            }
        } else {
            std::unordered_set<long long> selected;
            for (long long i = 0; i < k; ++i) {
                long long j = randbelow(n);
                while (selected.count(j)) j = randbelow(n);
                selected.insert(j);
                result.push_back(population[static_cast<std::size_t>(j)]);
            }
        }
        return result;
    }

    // choices(population, k=k): floor(random() * n)
    template <class T>
    std::vector<T> choices(const std::vector<T>& population, long long k = 1) {
        const double n = static_cast<double>(population.size());
        std::vector<T> out;
        for (long long i = 0; i < k; ++i) {
            const double x = std::floor(random() * n);
            if (population.empty()) throw IndexError(msg::list_index);
            out.push_back(population[static_cast<std::size_t>(x)]);
        }
        return out;
    }

    // choices(population, weights, k=k): the running totals of the weights, then bisect
    template <class T, class W>
    std::vector<T> choices(const std::vector<T>& population, const std::vector<W>& weights, long long k = 1) {
        std::vector<W> cum;
        W acc{};
        for (std::size_t i = 0; i < weights.size(); ++i) {
            acc = i == 0 ? weights[0] : acc + weights[i];
            cum.push_back(acc);
        }
        if (cum.size() != population.size()) throw ValueError(msg::random_weights_count);
        if (cum.empty()) throw IndexError(msg::list_index);
        const double total = static_cast<double>(cum.back()) + 0.0;
        if (total <= 0.0) throw ValueError(msg::random_weights_total);
        if (!std::isfinite(total)) throw ValueError(msg::random_weights_finite);
        const std::size_t hi = population.size() - 1;
        std::vector<T> out;
        for (long long i = 0; i < k; ++i) {
            const double x = random() * total;
            std::size_t lo = 0, h = hi;  // bisect_right(cum, x, 0, hi)
            while (lo < h) {
                const std::size_t mid = (lo + h) / 2;
                if (x < static_cast<double>(cum[mid])) h = mid; else lo = mid + 1;
            }
            out.push_back(population[lo]);
        }
        return out;
    }

    // what verify compares after a call: the next 32 bits, and gauss()'s saved second value
    std::uint32_t peek32() const {
        Random copy = *this;
        return copy.genrand();
    }
    const std::optional<double>& gauss_next() const { return gauss_next_; }

private:
    std::uint32_t mt_[624];
    int index_ = 625;
    std::optional<double> gauss_next_;

    static double runtime(double x) {  // a value worked out when the program runs, as Python does
        volatile double v = x;
        return v;
    }
    static double twopi() { return 2.0 * 3.141592653589793238462643383279502884; }  // TWOPI = 2.0 * math.pi
    static double nv_magicconst() {  // 4 * exp(-0.5) / sqrt(2.0), with the C library's exp
        static const double v = 4 * std::exp(runtime(-0.5)) / std::sqrt(2.0);
        return v;
    }
    static double log4() {
        static const double v = std::log(runtime(4.0));
        return v;
    }
    double triangular_(double low, double high, double c) { return finish_triangular(random(), low, high, c); }
    static double finish_triangular(double u, double low, double high, double c) {
        if (u > c) {
            u = 1.0 - u;
            c = 1.0 - c;
            std::swap(low, high);
        }
        return low + (high - low) * std::sqrt(u * c);
    }

    static std::string range_text(const char* prefix, const std::vector<long long>& v) {
        std::string s = prefix;
        s += "(";
        for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
        return s + ")";
    }

    void init_genrand(std::uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < 624; i++) mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<std::uint32_t>(i);
        index_ = 624;
    }
    void init_by_array(const std::uint32_t* key, int key_length) {
        init_genrand(19650218U);
        int i = 1, j = 0;
        for (int k = 624 > key_length ? 624 : key_length; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[j] + static_cast<std::uint32_t>(j);
            i++;
            j++;
            if (i >= 624) {
                mt_[0] = mt_[623];
                i = 1;
            }
            if (j >= key_length) j = 0;
        }
        for (int k = 623; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - static_cast<std::uint32_t>(i);
            i++;
            if (i >= 624) {
                mt_[0] = mt_[623];
                i = 1;
            }
        }
        mt_[0] = 0x80000000U;
        index_ = 624;
    }
    std::uint32_t genrand() {
        static const std::uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        std::uint32_t y;
        if (index_ >= 624) {
            int kk;
            for (kk = 0; kk < 624 - 397; kk++) {
                y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < 623; kk++) {
                y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (mt_[623] & 0x80000000U) | (mt_[0] & 0x7fffffffU);
            mt_[623] = mt_[396] ^ (y >> 1) ^ mag01[y & 0x1U];
            index_ = 0;
        }
        y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
};

// the generator behind random.random(), random.seed(), ... (Python seeds it from the operating
// system when the module is imported; qtrans refuses a program that draws before random.seed)
inline Random& random_module() {
    static Random r;
    return r;
}

}  // namespace py
}  // namespace qt
