// qtrans_list.hpp: Python lists of numbers or strings as std::vector, with Python's rules where
// a plain std::vector differs: negative indexes and IndexError, slices, sum() as CPython adds
// floats, min() / max() / sorted() as Python picks, and printing as Python prints a list.
//
// A list is a std::vector<double>, std::vector<long long> or std::vector<std::string>; the
// translator only uses these helpers where Python's behaviour needs them (xs.size() and
// xs.push_back(x) are written as they are).
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "qtrans_print.hpp"
#include "qtrans_runtime.hpp"

namespace qt {
namespace py {

inline constexpr std::nullopt_t none = std::nullopt;  // a slice bound left out: xs[1:]

// xs[i]: i < 0 counts from the end; out of range is IndexError, as in Python
template <class T>
T& item(std::vector<T>& xs, long long i) {
    const long long n = static_cast<long long>(xs.size());
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw std::out_of_range("list index out of range");
    return xs[static_cast<size_t>(i)];
}
template <class T>
const T& item(const std::vector<T>& xs, long long i) {
    const long long n = static_cast<long long>(xs.size());
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw std::out_of_range("list index out of range");
    return xs[static_cast<size_t>(i)];
}

// xs[start:stop:step], Python's rules (PySlice_AdjustIndices): bounds clip, negatives count from
// the end, any step but 0
template <class T>
std::vector<T> slice(const std::vector<T>& xs, std::optional<long long> start, std::optional<long long> stop,
                     std::optional<long long> step = std::nullopt) {
    const long long n = static_cast<long long>(xs.size());
    const long long s = step.value_or(1);
    if (s == 0) throw std::invalid_argument("slice step cannot be zero");
    auto adjust = [&](std::optional<long long> v, long long dflt) {
        if (!v) return dflt;
        long long x = *v;
        if (x < 0) {
            x += n;
            if (x < 0) x = s < 0 ? -1 : 0;
        } else if (x >= n) {
            x = s < 0 ? n - 1 : n;
        }
        return x;
    };
    const long long a = adjust(start, s < 0 ? n - 1 : 0), b = adjust(stop, s < 0 ? -1 : n);
    std::vector<T> out;
    if (s > 0) {
        for (long long i = a; i < b; i += s) out.push_back(xs[static_cast<size_t>(i)]);
    } else {
        for (long long i = a; i > b; i += s) out.push_back(xs[static_cast<size_t>(i)]);
    }
    return out;
}

// sum(xs) as CPython 3.12+ adds floats: from int 0, then Neumaier's compensated sum
// (Python/bltinmodule.c, builtin_sum_impl), the correction added at the end if finite
inline double sum(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;  // Python: the int 0 (the translator prints it as 0)
    double f = 0.0 + xs[0];      // 0 + x: the int start turns into the first float (-0.0 -> 0.0)
    double c = 0.0;
    for (size_t k = 1; k < xs.size(); ++k) {
        const double x = xs[k];
        const double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x;
        else c += (x - t) + f;
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}
// sum(xs) as Python 3.10 and 3.11 add floats: left to right, from int 0
inline double sum_left(const std::vector<double>& xs) {
    double f = 0.0;
    for (size_t k = 0; k < xs.size(); ++k) f = k ? f + xs[k] : 0.0 + xs[k];
    return f;
}
inline long long sum(const std::vector<long long>& xs) {
    long long s = 0;
    for (long long x : xs) s = qt::iadd(s, x);  // beyond 64 bits: OverflowError
    return s;
}

// min(xs) / max(xs): the first element no later one beats (Python keeps the current one unless a
// later one is strictly smaller / greater, so NaN behaves as it does in Python)
template <class T>
T min(const std::vector<T>& xs) {
    if (xs.empty()) throw std::invalid_argument("min() iterable argument is empty");
    T m = xs[0];
    for (size_t k = 1; k < xs.size(); ++k)
        if (xs[k] < m) m = xs[k];
    return m;
}
template <class T>
T max(const std::vector<T>& xs) {
    if (xs.empty()) throw std::invalid_argument("max() iterable argument is empty");
    T m = xs[0];
    for (size_t k = 1; k < xs.size(); ++k)
        if (xs[k] > m) m = xs[k];
    return m;
}

// sorted(xs) / xs.sort(): stable, as Python's sort. With a NaN in the list Python's order depends
// on its own algorithm (timsort), which isn't reproduced: that is refused at run time.
template <class T>
void check_sortable(const std::vector<T>& xs) {
    if constexpr (std::is_floating_point_v<T>) {
        for (const T& x : xs)
            if (std::isnan(x)) throw std::runtime_error("qtrans: sorting a list with NaN in it is not reproduced");
    }
}
template <class T>
std::vector<T> sorted(std::vector<T> xs, bool reverse = false) {
    check_sortable(xs);
    if (reverse) {
        // Python's reverse=True keeps equal elements in their original order
        std::stable_sort(xs.begin(), xs.end(), [](const T& a, const T& b) { return b < a; });
    } else {
        std::stable_sort(xs.begin(), xs.end());
    }
    return xs;
}
template <class T>
void sort(std::vector<T>& xs, bool reverse = false) { xs = sorted(std::move(xs), reverse); }

// xs.pop() / xs.pop(i)
template <class T>
T pop(std::vector<T>& xs, long long i = -1) {
    if (xs.empty()) throw std::out_of_range("pop from empty list");
    const long long n = static_cast<long long>(xs.size());
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw std::out_of_range("pop index out of range");
    T v = xs[static_cast<size_t>(i)];
    xs.erase(xs.begin() + i);
    return v;
}

template <class T>
T pop(std::vector<T>&& xs, long long i = -1) { return pop(xs, i); }  // [5, 6, 7].pop(1)

// xs.insert(i, x): i clipped to the list, negatives from the end, as Python
template <class T>
void insert(std::vector<T>& xs, long long i, T x) {
    const long long n = static_cast<long long>(xs.size());
    if (i < 0) i = std::max(0LL, i + n);
    if (i > n) i = n;
    xs.insert(xs.begin() + i, std::move(x));
}

// xs.index(x) (ValueError if absent), xs.count(x), x in xs
template <class T>
long long index(const std::vector<T>& xs, const T& x) {
    for (size_t k = 0; k < xs.size(); ++k)
        if (xs[k] == x) return static_cast<long long>(k);
    throw std::invalid_argument("list.index(x): x not in list");
}
template <class T>
long long count(const std::vector<T>& xs, const T& x) {
    return static_cast<long long>(std::count(xs.begin(), xs.end(), x));
}
template <class T>
bool contains(const std::vector<T>& xs, const T& x) {
    return std::find(xs.begin(), xs.end(), x) != xs.end();
}

// a + b
template <class T>
std::vector<T> concat(std::vector<T> a, const std::vector<T>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// xs * n (and n * xs): n copies one after the other; none for n <= 0
template <class T>
std::vector<T> repeat(const std::vector<T>& xs, long long n) {
    std::vector<T> out;
    for (long long k = 0; k < n; ++k) out.insert(out.end(), xs.begin(), xs.end());
    return out;
}

// len(s) of a str: its characters (UTF-8 code points), not its bytes
inline long long len(const std::string& s) {
    long long n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

// list(range(...)): the ints of a range
inline std::vector<long long> range_list(long long start, long long stop, long long step = 1) {
    if (step == 0) throw std::invalid_argument("range() arg 3 must not be zero");
    std::vector<long long> out;
    if (step > 0) for (long long i = start; i < stop; i += step) out.push_back(i);
    else for (long long i = start; i > stop; i += step) out.push_back(i);
    return out;
}

// printing: [1.0, 2.5], ['a', 'b'], [] (each element's repr); elements NumPy made show as NumPy
// scalars ([np.float64(1.0)])
template <class T>
std::string repr(const std::vector<T>& xs, bool numpy = false) {
    std::string out = "[";
    for (size_t k = 0; k < xs.size(); ++k) {
        if (k) out += ", ";
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long long>) {
            out += numpy ? repr(np(xs[k])) : repr(xs[k]);
        } else {
            out += repr(xs[k]);
        }
    }
    return out + "]";
}
template <class T>
std::string str(const std::vector<T>& xs) { return repr(xs); }

}  // namespace py
}  // namespace qt
