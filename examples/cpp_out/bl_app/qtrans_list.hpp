// qtrans_list.hpp: Python lists of numbers or strings as std::vector, with Python's rules where
// a plain std::vector differs: negative indexes and IndexError, slices, sum() as CPython adds
// floats, min() / max() / sorted() as Python picks, and printing as Python prints a list.
// Also Python's dict, as qt::py::dict: keys kept in the order they were first added.
//
// A list is a std::vector<double>, std::vector<long long> or std::vector<std::string>; the
// translator only uses these helpers where Python's behaviour needs them (xs.size() and
// xs.push_back(x) are written as they are).
#pragma once
#define QTRANS_LIST_HPP 1  // lets other code see that qt::py::dict is here

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <tuple>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qtrans_print.hpp"
#include "qtrans_runtime.hpp"

namespace qt {
namespace py {

inline constexpr std::nullopt_t none = std::nullopt;

template <class T>
struct is_tuple : std::false_type {};
template <class... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};
template <class T>
void check_sortable(const std::vector<T>& xs);  // below: NaN can't be ordered as Python orders it

// list(zip(xs, ys)): pairs up to the shorter one
template <class... T>
std::vector<std::tuple<T...>> zip(const std::vector<T>&... ls) {
    std::size_t n = std::min({ls.size()...});
    std::vector<std::tuple<T...>> out;
    out.reserve(n);
    for (std::size_t k = 0; k < n; ++k) out.emplace_back(ls[k]...);
    return out;
}
// list(enumerate(xs, start))
template <class T>
std::vector<std::tuple<long long, T>> enumerate(const std::vector<T>& xs, long long start = 0) {
    std::vector<std::tuple<long long, T>> out;
    out.reserve(xs.size());
    for (std::size_t k = 0; k < xs.size(); ++k) out.emplace_back(start + static_cast<long long>(k), xs[k]);
    return out;
}  // a slice bound left out: xs[1:]

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
// sum(xs, start) with a float start, as CPython 3.12+ adds floats: from start, compensated,
// the correction added at the end if finite (an empty list gives start itself)
inline double sum(const std::vector<double>& xs, double start) {
    double f = start, c = 0.0;
    for (double x : xs) {
        const double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x;
        else c += (x - t) + f;
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}
// ... and as Python 3.10 / 3.11 add them: left to right from start
inline double sum_left(const std::vector<double>& xs, double start) {
    double f = start;
    for (double x : xs) f += x;
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
    if constexpr (is_tuple<T>::value) check_sortable(xs);
    T m = xs[0];
    for (size_t k = 1; k < xs.size(); ++k)
        if (xs[k] < m) m = xs[k];
    return m;
}
template <class T>
T max(const std::vector<T>& xs) {
    if (xs.empty()) throw std::invalid_argument("max() iterable argument is empty");
    if constexpr (is_tuple<T>::value) check_sortable(xs);
    T m = xs[0];
    for (size_t k = 1; k < xs.size(); ++k)
        if (xs[k] > m) m = xs[k];
    return m;
}

// sorted(xs) / xs.sort(): stable, as Python's sort. With a NaN in the list Python's order depends
// on its own algorithm (timsort), which isn't reproduced: that is refused at run time.
template <class T>
bool has_nan(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(x);
    } else if constexpr (is_tuple<T>::value) {
        return std::apply([](const auto&... e) { return (has_nan(e) || ...); }, x);
    } else {
        return false;
    }
}
// With a NaN in it, Python's order depends on its own algorithm (timsort), and Python compares
// tuples element by element with == first, which a NaN never passes, where C++'s tuple < moves
// on to the next element: not reproduced, so refused at run time (min / max too).
template <class T>
void check_sortable(const std::vector<T>& xs) {
    for (const T& x : xs)
        if (has_nan(x)) throw std::runtime_error("qtrans: ordering a list with NaN in it is not reproduced");
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

// sorted(xs, key=...): each element's key computed once, in order (as Python does), then a stable
// sort by the keys; reverse=True keeps equal keys in their original order, as Python's does
template <class T, class K>
std::vector<T> sorted_by(const std::vector<T>& xs, const std::vector<K>& keys, bool reverse = false) {
    check_sortable(keys);
    std::vector<std::size_t> idx(xs.size());
    for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
    if (reverse) {
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return keys[b] < keys[a]; });
    } else {
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
    }
    std::vector<T> out;
    out.reserve(xs.size());
    for (std::size_t k : idx) out.push_back(xs[k]);
    return out;
}
// min(xs, key=...) / max(...): the first element whose key no later one beats (Python's own
// comparisons, so NaN keys behave as in Python; tuple keys holding NaN are refused, see above)
template <class T, class K>
T min_by(const std::vector<T>& xs, const std::vector<K>& keys) {
    if (xs.empty()) throw std::invalid_argument("min() iterable argument is empty");
    if constexpr (is_tuple<K>::value) check_sortable(keys);
    std::size_t best = 0;
    for (std::size_t k = 1; k < xs.size(); ++k)
        if (keys[k] < keys[best]) best = k;
    return xs[best];
}
template <class T, class K>
T max_by(const std::vector<T>& xs, const std::vector<K>& keys) {
    if (xs.empty()) throw std::invalid_argument("max() iterable argument is empty");
    if constexpr (is_tuple<K>::value) check_sortable(keys);
    std::size_t best = 0;
    for (std::size_t k = 1; k < xs.size(); ++k)
        if (keys[best] < keys[k]) best = k;
    return xs[best];
}

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
    for (char ch : s) n += (static_cast<unsigned char>(ch) & 0xC0) != 0x80;
    return n;
}

// sep.join(xs) for a list of str
inline std::string join(const std::string& sep, const std::vector<std::string>& xs) {
    std::string out;
    for (size_t k = 0; k < xs.size(); ++k) {
        if (k) out += sep;
        out += xs[k];
    }
    return out;
}

// --- strings -----------------------------------------------------------------------------------
// Python's str methods on UTF-8 text. Indexes count characters (code points), as Python's do.
// Case, character classes and digits come from qtrans_unicode.hpp, the tables of the Python
// the translation was made with (written from that Python by asking it, character by character).
namespace strdetail {
inline std::vector<std::size_t> offsets(const std::string& s) {  // where each character starts, and the end
    std::vector<std::size_t> out;
    for (std::size_t k = 0; k < s.size(); ++k)
        if ((static_cast<unsigned char>(s[k]) & 0xC0) != 0x80) out.push_back(k);
    out.push_back(s.size());
    return out;
}
using uni::decode;
using uni::encode;
using uni::find_range;
using uni::in;
inline std::vector<char32_t> code_points(const std::string& s) {
    std::vector<char32_t> out;
    const auto off = offsets(s);
    for (std::size_t i = 0; i + 1 < off.size(); ++i) out.push_back(decode(s, off[i]));
    return out;
}
// c's mapping (to upper / lower / title case), appended to out
template <std::size_t N, std::size_t M>
void map_to(std::string& out, const uni::Delta (&runs)[N], const uni::Map (&full)[M], char32_t c) {
    const uni::Map* m = std::lower_bound(full, full + M, c, [](const uni::Map& q, char32_t x) { return q.cp < x; });
    if (m != full + M && m->cp == c) {
        for (char32_t t : m->to)
            if (t) encode(out, t);
        return;
    }
    const uni::Delta* r = std::upper_bound(runs, runs + N, c, [](char32_t x, const uni::Delta& q) { return x < q.lo; });
    if (r != runs && c <= (r - 1)->hi && (c - (r - 1)->lo) % (r - 1)->step == 0) {
        encode(out, static_cast<char32_t>(static_cast<long long>(c) + (r - 1)->delta));
        return;
    }
    encode(out, c);
}
inline bool space(char32_t c) { return in(uni::space, c); }
// lower() of a capital sigma: the final form at the end of a word (CPython's handle_capital_sigma)
inline bool final_sigma(const std::vector<char32_t>& cs, std::size_t i) {
    std::size_t j = i;
    while (j > 0 && in(uni::case_ignorable, cs[j - 1])) --j;
    if (j == 0 || !in(uni::cased, cs[j - 1])) return false;
    j = i + 1;
    while (j < cs.size() && in(uni::case_ignorable, cs[j])) ++j;
    return j == cs.size() || !in(uni::cased, cs[j]);
}
inline void lower_at(std::string& out, const std::vector<char32_t>& cs, std::size_t i) {
    if (cs[i] == 0x3A3) {
        encode(out, final_sigma(cs, i) ? 0x3C2 : 0x3C3);
        return;
    }
    map_to(out, uni::to_lower, uni::to_lower_full, cs[i]);
}
template <class Pred>
bool all_chars(const std::string& s, Pred p) {  // Python's isdigit() & co: not empty, every character
    const auto cs = code_points(s);
    return !cs.empty() && std::all_of(cs.begin(), cs.end(), p);
}
// Python's slice bounds (start, end) of a string of n characters, as find() / count() take them
inline void bounds(long long n, std::optional<long long> a, std::optional<long long> b, long long& lo, long long& hi) {
    lo = a.value_or(0);
    hi = b.value_or(n);
    if (lo < 0) lo = std::max(0LL, lo + n);
    if (hi < 0) hi = std::max(0LL, hi + n);
    hi = std::min(hi, n);  // a start past the end stays there: "abc".find("", 5) is -1
}
}  // namespace strdetail

inline std::string upper(const std::string& s) {
    std::string out;
    for (char32_t c : strdetail::code_points(s)) strdetail::map_to(out, uni::to_upper, uni::to_upper_full, c);
    return out;
}
inline std::string lower(const std::string& s) {
    std::string out;
    const auto cs = strdetail::code_points(s);
    for (std::size_t i = 0; i < cs.size(); ++i) strdetail::lower_at(out, cs, i);
    return out;
}
inline std::string capitalize(const std::string& s) {  // the first character in title case, the rest lower
    std::string out;
    const auto cs = strdetail::code_points(s);
    for (std::size_t i = 0; i < cs.size(); ++i) {
        if (i == 0) strdetail::map_to(out, uni::to_title, uni::to_title_full, cs[0]);
        else strdetail::lower_at(out, cs, i);
    }
    return out;
}

inline std::string title(const std::string& s) {  // CPython: title case after an uncased character, else lower
    std::string out;
    const auto cs = strdetail::code_points(s);
    bool after_cased = false;
    for (std::size_t i = 0; i < cs.size(); ++i) {
        if (after_cased) strdetail::lower_at(out, cs, i);
        else strdetail::map_to(out, uni::to_title, uni::to_title_full, cs[i]);
        after_cased = strdetail::in(uni::cased, cs[i]);
    }
    return out;
}
inline std::string swapcase(const std::string& s) {
    std::string out;
    const auto cs = strdetail::code_points(s);
    for (std::size_t i = 0; i < cs.size(); ++i) {
        if (strdetail::in(uni::upper, cs[i])) strdetail::lower_at(out, cs, i);
        else if (strdetail::in(uni::lower, cs[i])) strdetail::map_to(out, uni::to_upper, uni::to_upper_full, cs[i]);
        else strdetail::encode(out, cs[i]);
    }
    return out;
}
inline std::string casefold(const std::string& s) {
    std::string out;
    for (char32_t c : strdetail::code_points(s)) strdetail::map_to(out, uni::to_fold, uni::to_fold_full, c);
    return out;
}

// strip(): Python's whitespace; strip(chars): any of those characters, from both ends
inline std::string strip_impl(const std::string& s, const std::optional<std::string>& chars, bool left, bool right) {
    const auto off = strdetail::offsets(s);
    const std::vector<char32_t> set = chars ? strdetail::code_points(*chars) : std::vector<char32_t>{};
    auto drop = [&](std::size_t i) {
        const char32_t c = strdetail::decode(s, off[i]);
        return chars ? std::find(set.begin(), set.end(), c) != set.end() : strdetail::space(c);
    };
    std::size_t a = 0, b = off.size() - 1;  // characters [a, b)
    if (left) while (a < b && drop(a)) ++a;
    if (right) while (b > a && drop(b - 1)) --b;
    return s.substr(off[a], off[b] - off[a]);
}
inline std::string strip(const std::string& s, const std::optional<std::string>& c = std::nullopt) { return strip_impl(s, c, true, true); }
inline std::string lstrip(const std::string& s, const std::optional<std::string>& c = std::nullopt) { return strip_impl(s, c, true, false); }
inline std::string rstrip(const std::string& s, const std::optional<std::string>& c = std::nullopt) { return strip_impl(s, c, false, true); }

// split(): runs of whitespace, none at the ends; split(sep): every sep, empty pieces kept
inline std::vector<std::string> split(const std::string& s, const std::optional<std::string>& sep = std::nullopt,
                                      long long maxsplit = -1) {
    std::vector<std::string> out;
    if (sep) {
        if (sep->empty()) throw std::invalid_argument("empty separator");
        std::size_t start = 0, at;
        while ((maxsplit < 0 || static_cast<long long>(out.size()) < maxsplit) && (at = s.find(*sep, start)) != std::string::npos) {
            out.push_back(s.substr(start, at - start));
            start = at + sep->size();
        }
        out.push_back(s.substr(start));
        return out;
    }
    const auto off = strdetail::offsets(s);
    std::size_t i = 0;
    const std::size_t n = off.size() - 1;
    while (true) {
        while (i < n && strdetail::space(strdetail::decode(s, off[i]))) ++i;
        if (i == n) break;
        if (maxsplit >= 0 && static_cast<long long>(out.size()) == maxsplit) {
            out.push_back(s.substr(off[i]));  // the rest, trailing whitespace and all (as CPython)
            break;
        }
        std::size_t j = i;
        while (j < n && !strdetail::space(strdetail::decode(s, off[j]))) ++j;
        out.push_back(s.substr(off[i], off[j] - off[i]));
        i = j;
    }
    return out;
}

// replace(old, new, count): old empty inserts new around every character, as Python does
inline std::string replace(const std::string& s, const std::string& old, const std::string& rep, long long count = -1) {
    std::string out;
    if (old.empty()) {
        const auto off = strdetail::offsets(s);
        long long done = 0;
        for (std::size_t i = 0; i + 1 < off.size(); ++i) {
            if (count < 0 || done < count) { out += rep; ++done; }
            out += s.substr(off[i], off[i + 1] - off[i]);
        }
        if (count < 0 || done < count) out += rep;
        return out;
    }
    std::size_t start = 0, at;
    long long done = 0;
    while ((count < 0 || done < count) && (at = s.find(old, start)) != std::string::npos) {
        out += s.substr(start, at - start);
        out += rep;
        start = at + old.size();
        ++done;
    }
    return out + s.substr(start);
}

// find / index / count / startswith / endswith: character positions, Python's start / end rules
inline long long find(const std::string& s, const std::string& sub, std::optional<long long> a = std::nullopt,
                      std::optional<long long> b = std::nullopt) {
    const auto off = strdetail::offsets(s);
    const long long n = static_cast<long long>(off.size()) - 1;
    long long lo, hi;
    strdetail::bounds(n, a, b, lo, hi);
    if (hi - lo < len(sub)) return -1;
    const std::size_t from = off[static_cast<std::size_t>(lo)], to = off[static_cast<std::size_t>(hi)];
    const std::string part = s.substr(from, to - from);
    const std::size_t at = part.find(sub);
    if (at == std::string::npos) return -1;
    return lo + len(part.substr(0, at));
}
inline long long index(const std::string& s, const std::string& sub, std::optional<long long> a = std::nullopt,
                       std::optional<long long> b = std::nullopt) {
    const long long k = find(s, sub, a, b);
    if (k < 0) throw std::invalid_argument("substring not found");
    return k;
}
inline long long count(const std::string& s, const std::string& sub, std::optional<long long> a = std::nullopt,
                       std::optional<long long> b = std::nullopt) {
    const auto off = strdetail::offsets(s);
    const long long n = static_cast<long long>(off.size()) - 1;
    long long lo, hi;
    strdetail::bounds(n, a, b, lo, hi);
    if (hi - lo < len(sub)) return 0;
    if (sub.empty()) return hi - lo + 1;
    const std::size_t from = off[static_cast<std::size_t>(lo)], to = off[static_cast<std::size_t>(hi)];
    const std::string part = s.substr(from, to - from);
    long long c = 0;
    for (std::size_t at = part.find(sub); at != std::string::npos; at = part.find(sub, at + sub.size())) ++c;
    return c;
}
inline bool startswith(const std::string& s, const std::string& p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
inline bool endswith(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
inline bool startswith(const std::string& s, std::initializer_list<std::string> ps) {  // s.startswith((a, b))
    return std::any_of(ps.begin(), ps.end(), [&](const std::string& p) { return startswith(s, p); });
}
inline bool endswith(const std::string& s, std::initializer_list<std::string> ps) {
    return std::any_of(ps.begin(), ps.end(), [&](const std::string& p) { return endswith(s, p); });
}
inline bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

// isdigit / isalpha / ...: Python's definitions, by its Unicode tables
inline bool isdecimal(const std::string& s) { return strdetail::all_chars(s, [](char32_t c) { return strdetail::in(uni::decimal, c); }); }
inline bool isdigit(const std::string& s) { return strdetail::all_chars(s, [](char32_t c) { return strdetail::in(uni::digit, c); }); }
inline bool isnumeric(const std::string& s) { return strdetail::all_chars(s, [](char32_t c) { return strdetail::in(uni::numeric, c); }); }
inline bool isalpha(const std::string& s) { return strdetail::all_chars(s, [](char32_t c) { return strdetail::in(uni::alpha, c); }); }
inline bool isalnum(const std::string& s) {
    return strdetail::all_chars(s, [](char32_t c) {
        return strdetail::in(uni::alpha, c) || strdetail::in(uni::decimal, c) || strdetail::in(uni::digit, c) ||
               strdetail::in(uni::numeric, c);
    });
}
inline bool isspace(const std::string& s) { return strdetail::all_chars(s, strdetail::space); }
inline bool isprintable(const std::string& s) {  // true for "", unlike the others
    const auto cs = strdetail::code_points(s);
    return std::all_of(cs.begin(), cs.end(), [](char32_t c) { return strdetail::in(uni::printable, c); });
}
inline bool isupper(const std::string& s) {  // a cased character, and none lowercase or titlecase
    bool cased = false;
    for (char32_t c : strdetail::code_points(s)) {
        if (strdetail::in(uni::lower, c) || strdetail::in(uni::title, c)) return false;
        if (strdetail::in(uni::upper, c)) cased = true;
    }
    return cased;
}
inline bool istitle(const std::string& s) {  // each word: one upper / title case letter, then lower case
    bool cased = false, after_cased = false;
    for (char32_t c : strdetail::code_points(s)) {
        if (strdetail::in(uni::upper, c) || strdetail::in(uni::title, c)) {
            if (after_cased) return false;
            after_cased = cased = true;
        } else if (strdetail::in(uni::lower, c)) {
            if (!after_cased) return false;
            after_cased = cased = true;
        } else {
            after_cased = false;
        }
    }
    return cased;
}
inline bool islower(const std::string& s) {
    bool cased = false;
    for (char32_t c : strdetail::code_points(s)) {
        if (strdetail::in(uni::upper, c) || strdetail::in(uni::title, c)) return false;
        if (strdetail::in(uni::lower, c)) cased = true;
    }
    return cased;
}

// padding to a width in characters: zfill keeps a sign in front; center as CPython splits the margin
inline std::string zfill(const std::string& s, long long width) {
    const long long n = len(s);
    if (n >= width) return s;
    std::string pad(static_cast<std::size_t>(width - n), '0');
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) return s.substr(0, 1) + pad + s.substr(1);
    return pad + s;
}
inline std::string fill_of(const std::string& f) {
    if (len(f) != 1) throw qt::type_error("The fill character must be exactly one character long");
    return f;
}
inline std::string times(const std::string& f, long long n) {
    std::string out;
    for (long long k = 0; k < n; ++k) out += f;
    return out;
}
inline std::string ljust(const std::string& s, long long width, const std::string& f = " ") {
    const long long n = len(s);
    return n >= width ? s : s + times(fill_of(f), width - n);
}
inline std::string rjust(const std::string& s, long long width, const std::string& f = " ") {
    const long long n = len(s);
    return n >= width ? s : times(fill_of(f), width - n) + s;
}
inline std::string center(const std::string& s, long long width, const std::string& f = " ") {
    const long long n = len(s);
    if (n >= width) return s;
    const long long marg = width - n;
    const long long left = marg / 2 + (marg & width & 1);  // CPython's pad()
    return times(fill_of(f), left) + s + times(f, marg - left);
}

// s[i], s[a:b:c], for ch in s: by character
inline std::string char_at(const std::string& s, long long i) {
    const auto off = strdetail::offsets(s);
    const long long n = static_cast<long long>(off.size()) - 1;
    if (i < 0) i += n;
    if (i < 0 || i >= n) throw std::out_of_range("string index out of range");
    const auto k = static_cast<std::size_t>(i);
    return s.substr(off[k], off[k + 1] - off[k]);
}
inline std::vector<std::string> chars(const std::string& s) {
    const auto off = strdetail::offsets(s);
    std::vector<std::string> out;
    for (std::size_t k = 0; k + 1 < off.size(); ++k) out.push_back(s.substr(off[k], off[k + 1] - off[k]));
    return out;
}
inline std::string slice(const std::string& s, std::optional<long long> a, std::optional<long long> b,
                         std::optional<long long> step = std::nullopt) {
    const std::vector<std::string> cs = slice(chars(s), a, b, step);
    std::string out;
    for (const auto& c : cs) out += c;
    return out;
}
inline std::string repeat(const std::string& s, long long n) { return times(s, n); }

// float(s) / int(s): as CPython, each Unicode digit becomes its ASCII digit and each whitespace
// character a space, any other non-ASCII character an error; then Python's syntax (spaces
// around, _ between digits, inf / nan, a sign) and the correctly rounded value. Else ValueError.
namespace strdetail {
inline std::string to_ascii_number(const std::string& text) {
    std::string out;
    for (char32_t c : code_points(text)) {
        if (c < 0x80) out += static_cast<char>(c);
        else if (space(c)) out += ' ';
        else if (const uni::Range* r = find_range(uni::decimal, c)) out += static_cast<char>('0' + (c - r->lo) % 10);
        else out += '?';
    }
    const std::size_t a = out.find_first_not_of(" \t\n\v\f\r"), b = out.find_last_not_of(" \t\n\v\f\r");
    return a == std::string::npos ? std::string() : out.substr(a, b - a + 1);
}
inline bool digit(char c) { return c >= '0' && c <= '9'; }
}  // namespace strdetail

inline double parse_float(const std::string& text) {
    const std::string t = strdetail::to_ascii_number(text);
    auto bad = [&]() { return std::invalid_argument("could not convert string to float: " + repr(text)); };
    std::size_t k = 0;
    std::string clean;
    if (k < t.size() && (t[k] == '+' || t[k] == '-')) clean += t[k++];
    std::string rest = t.substr(k);
    for (char& c : rest) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    if (rest == "inf" || rest == "infinity") return clean == "-" ? -qt::inf : qt::inf;
    if (rest == "nan") return clean == "-" ? -qt::nan : qt::nan;
    auto digits = [&](std::size_t& i) {  // digit (["_"] digit)*
        const std::size_t start = i;
        while (i < t.size() && strdetail::digit(t[i])) {
            clean += t[i++];
            if (i + 1 < t.size() && t[i] == '_' && strdetail::digit(t[i + 1])) ++i;
        }
        return i > start;
    };
    const bool lead = digits(k);
    bool frac = false;
    if (k < t.size() && t[k] == '.') {
        clean += t[k++];
        frac = digits(k);
    }
    if (!lead && !frac) throw bad();
    if (k < t.size() && (t[k] == 'e' || t[k] == 'E')) {
        clean += t[k++];
        if (k < t.size() && (t[k] == '+' || t[k] == '-')) clean += t[k++];
        if (!digits(k)) throw bad();
    }
    if (k != t.size()) throw bad();
    return std::strtod(clean.c_str(), nullptr);  // correctly rounded, as Python's
}
inline long long parse_int(const std::string& text) {
    const std::string t = strdetail::to_ascii_number(text);
    auto bad = [&]() { return std::invalid_argument("invalid literal for int() with base 10: " + repr(text)); };
    std::size_t k = 0;
    bool neg = false;
    if (k < t.size() && (t[k] == '+' || t[k] == '-')) neg = t[k++] == '-';
    if (k == t.size()) throw bad();
    unsigned long long v = 0;
    const unsigned long long limit = neg ? 9223372036854775808ULL : 9223372036854775807ULL;
    bool any = false, big = false;
    while (k < t.size()) {
        if (t[k] == '_' && any && k + 1 < t.size() && strdetail::digit(t[k + 1])) { ++k; continue; }
        if (!strdetail::digit(t[k])) throw bad();
        const unsigned d = static_cast<unsigned>(t[k++] - '0');
        if (big || v > (limit - d) / 10) big = true;  // still read on: a bad character later is ValueError
        else v = v * 10 + d;
        any = true;
    }
    if (big) throw qt::int_overflow();
    return neg ? static_cast<long long>(0ULL - v) : static_cast<long long>(v);
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

// --- dict ------------------------------------------------------------------------------------
// Python's dict: a key's position is the order it was first added (changing its value keeps the
// position; deleting and adding again puts it last), lookups by hash. Keys are str or int.
class KeyError : public std::out_of_range {
  public:
    using std::out_of_range::out_of_range;
};

template <class K>
std::string key_text(const K& k) { return repr(k); }  // KeyError's message: the key's repr

template <class K, class V>
class dict {
  public:
    using value_type = std::pair<K, V>;
    dict() = default;
    dict(std::initializer_list<value_type> items) {
        for (const auto& kv : items) (*this)[kv.first] = kv.second;  // a repeated key: the last value wins
    }

    long long size() const { return static_cast<long long>(items_.size()); }
    bool empty() const { return items_.empty(); }
    bool contains(const K& k) const { return index_.count(k) != 0; }

    // d[k]: KeyError if absent
    const V& at(const K& k) const {
        auto it = index_.find(k);
        if (it == index_.end()) throw KeyError(key_text(k));
        return items_[it->second].second;
    }
    V& at(const K& k) {
        auto it = index_.find(k);
        if (it == index_.end()) throw KeyError(key_text(k));
        return items_[it->second].second;
    }
    // d[k] = v: an existing key keeps its place, a new one goes last
    V& operator[](const K& k) {
        auto it = index_.find(k);
        if (it != index_.end()) return items_[it->second].second;
        index_.emplace(k, items_.size());
        items_.emplace_back(k, V{});
        return items_.back().second;
    }
    V get(const K& k, const V& dflt) const {
        auto it = index_.find(k);
        return it == index_.end() ? dflt : items_[it->second].second;
    }
    V& setdefault(const K& k, const V& dflt) {
        auto it = index_.find(k);
        if (it != index_.end()) return items_[it->second].second;
        return (*this)[k] = dflt;
    }
    V pop(const K& k) {
        auto it = index_.find(k);
        if (it == index_.end()) throw KeyError(key_text(k));
        return erase_at(it->second);
    }
    V pop(const K& k, const V& dflt) {
        auto it = index_.find(k);
        return it == index_.end() ? dflt : erase_at(it->second);
    }
    void erase(const K& k) { pop(k); }  // del d[k]
    void clear() { items_.clear(); index_.clear(); }
    void update(const dict& other) {
        for (const auto& kv : other.items_) (*this)[kv.first] = kv.second;
    }

    // in order: for (const auto& [k, v] : d), d.keys(), d.values()
    typename std::vector<value_type>::const_iterator begin() const { return items_.begin(); }
    typename std::vector<value_type>::const_iterator end() const { return items_.end(); }
    std::vector<K> keys() const {
        std::vector<K> out;
        out.reserve(items_.size());
        for (const auto& kv : items_) out.push_back(kv.first);
        return out;
    }
    std::vector<V> values() const {
        std::vector<V> out;
        out.reserve(items_.size());
        for (const auto& kv : items_) out.push_back(kv.second);
        return out;
    }
    bool operator==(const dict& o) const {  // Python: same keys and values, in any order
        if (size() != o.size()) return false;
        for (const auto& kv : items_) {
            if (!o.contains(kv.first)) return false;
            if constexpr (std::is_floating_point_v<V>) {
                // Python compares the values with `is` first: a NaN equals the very same NaN
                // object, which a copy may or may not be -- not reproduced
                if (std::isnan(kv.second) || std::isnan(o.at(kv.first)))
                    throw std::runtime_error("qtrans: comparing dicts holding NaN is not reproduced");
            }
            if (!(o.at(kv.first) == kv.second)) return false;
        }
        return true;
    }
    bool operator!=(const dict& o) const { return !(*this == o); }

  private:
    V erase_at(size_t i) {
        V v = std::move(items_[i].second);
        index_.erase(items_[i].first);
        items_.erase(items_.begin() + static_cast<std::ptrdiff_t>(i));
        for (size_t j = i; j < items_.size(); ++j) index_[items_[j].first] = j;
        return v;
    }
    std::vector<value_type> items_;
    std::unordered_map<K, size_t> index_;
};

// list(d.items()): its (key, value) pairs, in order
template <class K, class V>
std::vector<std::tuple<K, V>> items(const dict<K, V>& d) {
    std::vector<std::tuple<K, V>> out;
    for (const auto& [k, v] : d) out.emplace_back(k, v);
    return out;
}
// dict(pairs), dict(zip(keys, values)): a key seen again keeps its place and takes the later value
template <class K, class V>
dict<K, V> dict_from(const std::vector<std::tuple<K, V>>& pairs) {
    dict<K, V> d;
    for (const auto& [k, v] : pairs) d[k] = v;
    return d;
}

// printing: {'a': 1.0, 'b': 2}, {1: 'x'}, {}
template <class K, class V>
std::string repr(const dict<K, V>& d, bool numpy = false) {
    std::string out = "{";
    bool first = true;
    for (const auto& [k, v] : d) {
        if (!first) out += ", ";
        first = false;
        out += repr(k) + ": ";
        if constexpr (std::is_same_v<V, double> || std::is_same_v<V, long long>) {
            out += numpy ? repr(np(v)) : repr(v);
        } else {
            out += repr(v);
        }
    }
    return out + "}";
}
template <class K, class V>
std::string str(const dict<K, V>& d) { return repr(d); }

}  // namespace py
}  // namespace qt
