// qtrans_print.hpp: print(), str(), repr(), f-strings and str.format, character for character
// as Python and NumPy write them.
//
// Floats are the hard part. Python's repr(x) is the shortest decimal that reads back as x
// (0.1, not 0.1000000000000000055511151231257827); format specs (".2f", ".3e", ",", "%", "g")
// follow CPython's own rules (Python/pystrtod.c and Python/formatter_unicode.c, ported here),
// and NumPy prints an array with all its elements in one style (fixed or scientific), the
// decimal points aligned, 8 digits at most (numpy/_core/arrayprint.py, ported here). The digits
// themselves come from snprintf, which rounds correctly (glibc, macOS, the Windows UCRT), and
// the shortest form is found by trying 1, 2, ... 17 significant digits until one reads back.
// tests/test_print.py checks all of it against Python and NumPy on thousands of values.
#pragma once

#include <cctype>
#include <cmath>
#include <limits>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "qtrans_runtime.hpp"
#include "qtrans_unicode.hpp"  // the Unicode tables of Python's str methods and repr()

#include <iostream>
#include <streambuf>

#if defined(_WIN32)
// The three Windows calls console output needs, declared as <windows.h> declares them (so the
// two agree if both are included); <windows.h> itself defines macros such as min, max, small
// and near that would break ordinary code.
extern "C" {
__declspec(dllimport) void* __stdcall GetStdHandle(unsigned long);
__declspec(dllimport) int __stdcall GetConsoleMode(void*, unsigned long*);
__declspec(dllimport) int __stdcall WriteConsoleW(void*, const void*, unsigned long, unsigned long*, void*);
}
#endif

namespace qt {
namespace py {

// --- digits of a double, as David Gay's dtoa gives them to CPython ------------------------------
struct Digits {
    std::string d;  // significant digits, no leading or trailing zeros ("0" for zero)
    int decpt = 0;  // the decimal point is after d[decpt - 1]: 0.0012 is ("12", -2), 120 is ("12", 3)
    bool neg = false;
};

// "d.ddddde+-x" from snprintf -> Digits (trailing zeros dropped)
inline Digits parse_e(const char* s) {
    Digits r;
    if (*s == '-') { r.neg = true; ++s; }
    std::string digits;
    for (; *s && *s != 'e'; ++s)
        if (*s != '.') digits += *s;
    int e = *s == 'e' ? std::atoi(s + 1) : 0;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    if (digits == "0") { r.d = "0"; r.decpt = 1; return r; }
    r.d = digits;
    r.decpt = e + 1;
    return r;
}

// the closest decimal with n significant digits (mode 2 of dtoa)
inline Digits sig_digits(double x, int n) {
    std::vector<char> buf(static_cast<size_t>(n) + 32);
    std::snprintf(buf.data(), buf.size(), "%.*e", n - 1, x);
    return parse_e(buf.data());
}

// the shortest decimal that reads back as x (mode 0): the closest one with 1, 2, ... digits;
// where the closest misses (just past a power of two, the gaps either side differ), the
// neighbour one unit up in the last digit is tried too
inline Digits shortest(double x) {
    char buf[64];
    for (int n = 1; n <= 17; ++n) {
        std::snprintf(buf, sizeof buf, "%.*e", n - 1, x);
        if (std::strtod(buf, nullptr) == x) return parse_e(buf);
        if (n < 17) {
            // the decimal one unit above or below in the last place
            std::string s(buf);
            const size_t epos = s.find('e');
            std::string mant = s.substr(0, epos), ex = s.substr(epos);
            for (int dir : {1, -1}) {
                std::string m = mant;
                int i = static_cast<int>(m.size()) - 1;
                bool ok = true;
                while (i >= 0) {
                    if (m[static_cast<size_t>(i)] == '.' || m[static_cast<size_t>(i)] == '-') { --i; continue; }
                    char& c = m[static_cast<size_t>(i)];
                    if (dir > 0) { if (c == '9') { c = '0'; --i; continue; } ++c; break; }
                    if (c == '0') { c = '9'; --i; continue; }
                    --c;
                    break;
                }
                if (i < 0) ok = false;  // ran off the front (9.99 -> 10.0): not a same-length neighbour
                if (!ok) continue;
                std::string cand = m + ex;
                if (std::strtod(cand.c_str(), nullptr) == x) return parse_e(cand.c_str());
            }
        }
    }
    std::snprintf(buf, sizeof buf, "%.16e", x);
    return parse_e(buf);
}

// the closest decimal with n digits after the point (mode 3)
inline Digits fixed_digits(double x, int n) {
    std::vector<char> buf(static_cast<size_t>(n) + 400);
    std::snprintf(buf.data(), buf.size(), "%.*f", n, x);
    Digits r;
    const char* s = buf.data();
    if (*s == '-') { r.neg = true; ++s; }
    std::string ip, fp;
    for (; *s && *s != '.'; ++s) ip += *s;
    if (*s == '.') for (++s; *s; ++s) fp += *s;
    std::string all = ip + fp;
    size_t lead = 0;
    while (lead < all.size() && all[lead] == '0') ++lead;
    if (lead == all.size()) { r.d = "0"; r.decpt = 1; return r; }
    const int decpt = static_cast<int>(ip.size()) - static_cast<int>(lead);
    std::string d = all.substr(lead);
    while (d.size() > 1 && d.back() == '0') d.pop_back();
    r.d = d;
    r.decpt = decpt;
    return r;
}

// round(x, ndigits) as CPython does it (float___round__ / double_round in Objects/floatobject.c):
// the exact decimal value of x correctly rounded to ndigits places (half to even), read back
inline double round(double x, long long ndigits) {
    if (!std::isfinite(x)) return x;        // nan and inf round to themselves
    if (ndigits > 323) return x;            // NDIGITS_MAX: rounding can't change any double
    if (ndigits < -308) return 0.0 * x;     // NDIGITS_MIN: every finite double rounds to zero
    if (x == 0.0) return x;
    const int n = static_cast<int>(ndigits);
    Digits r;
    if (n >= 0) {
        r = fixed_digits(x, n);
    } else {
        // where the exact value's first digit is (not the shortest repr's: 1e+23 is 9.99...e22),
        // and 40 of its digits, which show which side of a half unit it lies
        const Digits exact = sig_digits(x, 40);
        const int keep = exact.decpt + n;  // significant digits left of the 10^-n place
        if (keep > 0) {
            r = sig_digits(x, keep);
        } else {
            r.neg = std::signbit(x);
            r.d = "0";
            r.decpt = 1;
            if (keep == 0) {
                // 0 or one unit of 10^-n: up when the exact |x| is past half a unit (a tie goes
                // to the even 0)
                const bool up = exact.d[0] > '5' || (exact.d[0] == '5' && exact.d.size() > 1);
                if (up) { r.d = "1"; r.decpt = exact.decpt + 1; }
            }
        }
    }
    std::string text = (r.neg ? "-" : "") + r.d + "e" + std::to_string(r.decpt - static_cast<int>(r.d.size()));
    const double v = std::strtod(text.c_str(), nullptr);
    if (std::isinf(v)) throw OverflowError("rounded value too large to represent");
    return v;
}

// round(x, ndigits) of a NumPy float64 (np.float64.__round__, which is np.round): x times 10^n,
// rounded half to even, divided by 10^n (n < 0: divided, rounded, multiplied). It differs from
// Python's round() where the product rounds across a half: round(np.float64(1.85), 1) is 1.8.
inline double round_numpy(double x, long long ndigits) {
    auto power_of_ten = [](long long n) {  // as NumPy computes it (calculation.c)
        static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
        if (n < 9) return p10[n];
        double r = 1e9;
        while (n-- > 9) r *= 10.;
        return r;
    };
    if (ndigits >= 0) {
        const double f = power_of_ten(ndigits);
        return std::nearbyint(x * f) / f;
    }
    const double f = power_of_ten(-ndigits);
    return std::nearbyint(x / f) * f;
}

// round(n, ndigits) of an int: an int, n rounded to a multiple of 10^-ndigits, half to even
inline long long round(long long v, long long ndigits) {
    if (ndigits >= 0) return v;
    if (ndigits < -18) return 0;  // |v| < 9.3e18 < half of 10^19
    long long unit = 1;
    for (long long k = 0; k < -ndigits; ++k) unit *= 10;
    long long q = v / unit, rem = v % unit;  // truncated toward zero
    if (rem < 0) { rem += unit; --q; }      // floor: 0 <= rem < unit
    if (rem > unit / 2 || (rem == unit / 2 && q % 2 != 0)) ++q;
    return qt::imul(q, unit);  // beyond 64 bits: OverflowError (Python's int would grow)
}

// CPython's format_float_short (Python/pystrtod.c): code 'e', 'f', 'g' or 'r' (repr)
inline std::string format_float_short(double x, char code, int precision, bool add_sign, bool add_dot_0,
                                      bool alt, bool no_neg_zero, bool upper) {
    std::string out;
    if (std::isnan(x)) return std::string(add_sign ? "+" : "") + (upper ? "NAN" : "nan");
    if (std::isinf(x)) return std::string(x < 0 ? "-" : add_sign ? "+" : "") + (upper ? "INF" : "inf");
    Digits dg;
    switch (code) {
        case 'e': dg = sig_digits(x, precision + 1); break;
        case 'f': dg = fixed_digits(x, precision); break;
        case 'g': dg = sig_digits(x, precision == 0 ? 1 : precision); if (precision == 0) precision = 1; break;
        default: dg = x == 0.0 ? Digits{"0", 1, std::signbit(x)} : shortest(x); break;  // 'r'
    }
    dg.neg = std::signbit(x);
    std::string digits = dg.d;
    int decpt = dg.decpt;
    if (code == 'f' && digits == "0") {  // dtoa gives no digits for a value that rounds to 0
        digits.clear();
        decpt = -precision;
    }
    bool neg = dg.neg;
    if (no_neg_zero && neg && (digits.empty() || digits == "0")) neg = false;
    const int dlen = static_cast<int>(digits.size());
    int vend = dlen;
    bool use_exp = false;
    switch (code) {
        case 'e': use_exp = true; vend = precision + 1; break;
        case 'f': vend = decpt + precision; break;
        case 'g':
            if (decpt <= -4 || decpt > (add_dot_0 ? precision - 1 : precision)) use_exp = true;
            if (alt) vend = precision;
            break;
        default:
            if (decpt <= -4 || decpt > 16) use_exp = true;
            break;
    }
    int exp = 0;
    if (use_exp) { exp = decpt - 1; decpt = 1; }
    const int vstart = decpt <= 0 ? decpt - 1 : 0;
    if (!use_exp && add_dot_0)
        vend = vend > decpt ? vend : decpt + 1;
    else
        vend = vend > decpt ? vend : decpt;
    if (neg) out += '-';
    else if (add_sign) out += '+';
    if (decpt <= 0) {
        out.append(static_cast<size_t>(decpt - vstart), '0');
        out += '.';
        out.append(static_cast<size_t>(-decpt), '0');
    } else {
        out.append(static_cast<size_t>(-vstart), '0');
    }
    if (0 < decpt && decpt <= dlen) {
        out += digits.substr(0, static_cast<size_t>(decpt));
        out += '.';
        out += digits.substr(static_cast<size_t>(decpt));
    } else {
        out += digits;
    }
    if (dlen < decpt) {
        out.append(static_cast<size_t>(decpt - dlen), '0');
        out += '.';
        out.append(static_cast<size_t>(vend - decpt), '0');
    } else {
        out.append(static_cast<size_t>(vend - dlen), '0');
    }
    if (out.back() == '.' && !alt) out.pop_back();
    if (use_exp) {
        char eb[16];
        std::snprintf(eb, sizeof eb, "%c%+.02d", upper ? 'E' : 'e', exp);
        out += eb;
    }
    return out;
}

// --- repr / str ----------------------------------------------------------------------------------
inline std::string repr(double x) { return format_float_short(x, 'r', 0, false, true, false, false, false); }
inline std::string repr(long long x) { return std::to_string(x); }
inline std::string repr(bool b) { return b ? "True" : "False"; }

// repr of a str, as Python writes it: single quotes unless the text has a ' and no ", backslash
// escapes for the quote, \\, \t \n \r and every character Python doesn't count as printable
// (\x.., \u...., \U........ by its Unicode tables, qtrans_unicode.hpp); the rest as it is
inline std::string repr(const std::string& s) {
    const char q = (s.find('\'') != std::string::npos && s.find('"') == std::string::npos) ? '"' : '\'';
    std::string out(1, q);
    for (std::size_t k = 0; k < s.size();) {
        std::size_t n = 1;
        const char32_t c = uni::decode(s, k, &n);
        if (c == static_cast<char32_t>(q) || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c == '\r') out += "\\r";
        else if (!uni::in(uni::printable, c)) {
            char b[16];
            const unsigned v = static_cast<unsigned>(c);
            std::snprintf(b, sizeof b, c <= 0xFF ? "\\x%02x" : c <= 0xFFFF ? "\\u%04x" : "\\U%08x", v);
            out += b;
        }
        else out.append(s, k, n);
        k += n;
    }
    out += q;
    return out;
}
inline std::string repr(const char* s) { return repr(std::string(s)); }

inline std::string str(double x) { return repr(x); }
inline std::string str(long long x) { return std::to_string(x); }
inline std::string str(bool b) { return b ? "True" : "False"; }
inline std::string str(const std::string& s) { return s; }
inline std::string str(const char* s) { return s; }
inline std::string str(std::nullopt_t) { return "None"; }
template <class T>
std::string str(const std::optional<T>& x) { return x ? str(*x) : std::string("None"); }
template <class T>
std::string repr(const std::optional<T>& x) { return x ? repr(*x) : std::string("None"); }

// a local that Python holds as an int in some assignments and a float in others (total = 0, then
// total += 0.5): the C++ keeps a double and a flag saying which Python holds now, and shows it so
inline std::string str(double x, bool as_int) {
    return as_int ? std::to_string(static_cast<long long>(x)) : str(x);
}
inline std::string repr(double x, bool as_int) { return str(x, as_int); }

// a value NumPy made (np.sum(v), v[i], v @ w, ...): print() and str() show it as Python shows a
// float, int or bool, but repr() names its type, as NumPy 2 does: np.float64(2.5), np.int64(3),
// np.True_. The translator wraps a value in np() only where Python would hold a NumPy scalar.
template <class T>
struct NumpyScalar { T v; };
template <class T>
NumpyScalar<T> np(T v) { return {v}; }
inline std::string repr(NumpyScalar<double> x) { return "np.float64(" + repr(x.v) + ")"; }
inline std::string repr(NumpyScalar<long long> x) { return "np.int64(" + repr(x.v) + ")"; }
inline std::string repr(NumpyScalar<bool> x) { return x.v ? "np.True_" : "np.False_"; }
template <class T>
std::string str(NumpyScalar<T> x) { return str(x.v); }

// arrays (defined below): declared here so a tuple holding one can show it
template <class A, class = decltype(A::ColsAtCompileTime)>
std::string str(const A& a);
template <class A, class = decltype(A::ColsAtCompileTime), class = void>
std::string repr(const A& a);

// a tuple shows the repr of each element: (1, 'a', 2.5) and (1,)
template <class... T>
std::string repr(const std::tuple<T...>& t) {
    std::string out = "(";
    size_t k = 0;
    std::apply([&](const auto&... v) { ((out += (k++ ? ", " : "") + repr(v)), ...); }, t);
    if (sizeof...(T) == 1) out += ",";
    return out + ")";
}
template <class... T>
std::string str(const std::tuple<T...>& t) { return repr(t); }

// a tuple some of whose elements are NumPy scalars: numpy[i] is 'n' for those, 'p' for the others
template <class T>
std::string repr_maybe_numpy(const T& x, bool numpy) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long long> || std::is_same_v<T, bool>) {
        if (numpy) return repr(np(x));
    }
    return repr(x);
}
template <class... T, size_t... I>
std::string repr_marked(const std::tuple<T...>& t, const char* numpy, std::index_sequence<I...>) {
    std::string out = "(";
    ((out += (I ? ", " : "") + repr_maybe_numpy(std::get<I>(t), numpy[I] == 'n')), ...);
    if (sizeof...(T) == 1) out += ",";
    return out + ")";
}
template <class... T>
std::string repr(const std::tuple<T...>& t, const char* numpy) {
    return repr_marked(t, numpy, std::index_sequence_for<T...>{});
}

// --- format(value, spec): Python's format mini-language ---------------------------------------
struct Spec {
    std::string fill = " ";
    char align = 0;      // < > ^ = or 0 (the type's default)
    char sign = '-';     // + - or space
    bool z = false, alt = false, sign_given = false, zero_pad = false;  // zero_pad: '0' set the fill and '='
    int width = -1, precision = -1;
    char grouping = 0;   // , or _
    char type = 0;
};

inline Spec parse_spec(const std::string& s) {
    Spec sp;
    size_t i = 0;
    auto is_align = [](char c) { return c == '<' || c == '>' || c == '^' || c == '='; };
    // the fill may be any character (one UTF-8 sequence) followed by an alignment
    size_t clen = 1;
    if (!s.empty()) {
        const unsigned char c0 = static_cast<unsigned char>(s[0]);
        clen = c0 < 0x80 ? 1 : c0 < 0xE0 ? 2 : c0 < 0xF0 ? 3 : 4;
    }
    if (s.size() > clen && is_align(s[clen])) { sp.fill = s.substr(0, clen); sp.align = s[clen]; i = clen + 1; }
    else if (!s.empty() && is_align(s[0])) { sp.align = s[0]; i = 1; }
    if (i < s.size() && (s[i] == '+' || s[i] == '-' || s[i] == ' ')) { sp.sign = s[i++]; sp.sign_given = true; }
    if (i < s.size() && s[i] == 'z') { sp.z = true; ++i; }
    if (i < s.size() && s[i] == '#') { sp.alt = true; ++i; }
    if (i < s.size() && s[i] == '0') {
        if (!sp.align) { sp.fill = "0"; sp.align = '='; sp.zero_pad = true; }
        ++i;
    }
    if (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        sp.width = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) sp.width = sp.width * 10 + (s[i++] - '0');
    }
    if (i < s.size() && (s[i] == ',' || s[i] == '_')) sp.grouping = s[i++];
    if (i < s.size() && s[i] == '.') {
        ++i;
        sp.precision = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) sp.precision = sp.precision * 10 + (s[i++] - '0');
    }
    if (i < s.size()) sp.type = s[i++];
    if (i != s.size()) throw ValueError("Invalid format specifier '" + s + "'");
    return sp;
}

inline size_t text_width(const std::string& s) {  // characters, not bytes
    size_t n = 0;
    for (char ch : s) n += (static_cast<unsigned char>(ch) & 0xC0) != 0x80;
    return n;
}

inline std::string repeat(const std::string& fill, size_t n) {
    std::string out;
    for (size_t k = 0; k < n; ++k) out += fill;
    return out;
}

// pad `body` (with an optional sign/prefix kept left of '=' padding) to the spec's width
inline std::string pad(const std::string& sign, const std::string& body, const Spec& sp, char default_align) {
    const char align = sp.align ? sp.align : default_align;
    const size_t len = text_width(sign) + text_width(body);
    if (sp.width < 0 || len >= static_cast<size_t>(sp.width)) return sign + body;
    const size_t n = static_cast<size_t>(sp.width) - len;
    if (align == '<') return sign + body + repeat(sp.fill, n);
    if (align == '>') return repeat(sp.fill, n) + sign + body;
    if (align == '^') return repeat(sp.fill, n / 2) + sign + body + repeat(sp.fill, n - n / 2);
    return sign + repeat(sp.fill, n) + body;  // '='
}

// insert `sep` every `every` digits from the right of the leading digit run; with '0'-fill and
// '=' alignment the padding zeros are grouped too, as CPython does
inline std::string group(const std::string& digits, char sep, int every) {
    std::string out;
    const int n = static_cast<int>(digits.size());
    for (int k = 0; k < n; ++k) {
        if (k && (n - k) % every == 0) out += sep;
        out += digits[static_cast<size_t>(k)];
    }
    return out;
}

inline std::string grouped_zero_padded(std::string digits, char sep, int every, size_t min_width) {
    // CPython fills with zeros before grouping, so separators land in the padding as well
    std::string g = group(digits, sep, every);
    while (g.size() < min_width) {
        digits = "0" + digits;
        g = group(digits, sep, every);
    }
    return g;
}

inline std::string format_number(std::string sign, std::string number, const Spec& sp, int every,
                                 bool whole = false) {
    // number: digits [. ...] [exponent] [%]: the leading decimal digits are grouped; for an
    // int (whole), every digit, hex ones included
    size_t k = whole ? number.size() : 0;
    while (k < number.size() && std::isdigit(static_cast<unsigned char>(number[k]))) ++k;
    std::string ip = number.substr(0, k), rest = number.substr(k);
    if (sp.grouping) {
        const char align = sp.align ? sp.align : '>';
        if (align == '=' && sp.fill == "0" && sp.width > 0) {
            const long need = static_cast<long>(sp.width) - static_cast<long>(text_width(sign) + text_width(rest));
            ip = grouped_zero_padded(ip, sp.grouping, every, need > 0 ? static_cast<size_t>(need) : 0);
        } else {
            ip = group(ip, sp.grouping, every);
        }
    }
    return pad(sign, ip + rest, sp, '>');
}

inline std::string format(double x, const std::string& spec) {
    Spec sp = parse_spec(spec);
    char type = sp.type;
    if (type == 'n') type = 'g';
    if (!(type == 0 || type == 'e' || type == 'E' || type == 'f' || type == 'F' || type == 'g' || type == 'G' || type == '%'))
        throw ValueError(std::string("Unknown format code '") + type + "' for object of type 'float'");
    bool add_dot_0 = false;
    char code = type;
    int precision = sp.precision;
    if (type == 0) { add_dot_0 = true; code = precision < 0 ? 'r' : 'g'; if (precision < 0) precision = 0; }
    else if (precision < 0) precision = 6;
    bool percent = type == '%';
    if (percent) { x *= 100; code = 'f'; }
    const bool upper = code == 'E' || code == 'F' || code == 'G';
    if (upper) code = static_cast<char>(code - 'A' + 'a');
    std::string s = format_float_short(x, code, precision, false, add_dot_0, sp.alt, sp.z, upper);
    std::string sign;
    if (!s.empty() && s[0] == '-') { sign = "-"; s = s.substr(1); }
    else if (sp.sign == '+') sign = "+";
    else if (sp.sign == ' ') sign = " ";
    if (percent) s += '%';
    if (s == "nan" || s == "inf" || s == "NAN" || s == "INF") return pad(sign, s, sp, '>');  // no grouping
    return format_number(sign, s, sp, 3);
}

inline std::string format(long long v, const std::string& spec) {
    Spec sp = parse_spec(spec);
    const char t = sp.type;
    if (t == 'e' || t == 'E' || t == 'f' || t == 'F' || t == 'g' || t == 'G' || t == '%')
        return format(static_cast<double>(v), spec);
    if (sp.precision >= 0) throw ValueError("Precision not allowed in integer format specifier");
    if (sp.z) throw ValueError("Negative zero coercion (z) not allowed in integer format specifier");
    if (t == 'c') {
        if (sp.sign_given) throw ValueError("Sign not allowed with integer format specifier 'c'");
        if (sp.alt) throw ValueError("Alternate form (#) not allowed with integer format specifier 'c'");
        if (v < 0 || v > 0x10FFFF) throw OverflowError("%c arg not in range(0x110000)");
        std::string ch;
        const unsigned long u = static_cast<unsigned long>(v);
        if (u < 0x80) ch = std::string(1, static_cast<char>(u));
        else if (u < 0x800) { ch += static_cast<char>(0xC0 | (u >> 6)); ch += static_cast<char>(0x80 | (u & 0x3F)); }
        else if (u < 0x10000) { ch += static_cast<char>(0xE0 | (u >> 12)); ch += static_cast<char>(0x80 | ((u >> 6) & 0x3F)); ch += static_cast<char>(0x80 | (u & 0x3F)); }
        else { ch += static_cast<char>(0xF0 | (u >> 18)); ch += static_cast<char>(0x80 | ((u >> 12) & 0x3F)); ch += static_cast<char>(0x80 | ((u >> 6) & 0x3F)); ch += static_cast<char>(0x80 | (u & 0x3F)); }
        return pad("", ch, sp, '<');
    }
    int base = 10;
    std::string prefix;
    if (t == 'b') { base = 2; prefix = "0b"; }
    else if (t == 'o') { base = 8; prefix = "0o"; }
    else if (t == 'x') { base = 16; prefix = "0x"; }
    else if (t == 'X') { base = 16; prefix = "0X"; }
    else if (!(t == 0 || t == 'd' || t == 'n'))
        throw ValueError(std::string("Unknown format code '") + t + "' for object of type 'int'");
    unsigned long long u = v < 0 ? 0ULL - static_cast<unsigned long long>(v) : static_cast<unsigned long long>(v);
    std::string digits;
    do {
        const int d = static_cast<int>(u % static_cast<unsigned long long>(base));
        digits += static_cast<char>(d < 10 ? '0' + d : (t == 'X' ? 'A' : 'a') + d - 10);
        u /= static_cast<unsigned long long>(base);
    } while (u);
    std::string num(digits.rbegin(), digits.rend());
    std::string sign = v < 0 ? "-" : sp.sign == '+' ? "+" : sp.sign == ' ' ? " " : "";
    if (sp.alt) sign += prefix;
    return format_number(sign, num, sp, base == 10 ? 3 : 4, true);
}

inline std::string format(bool b, const std::string& spec) {
    if (spec.empty()) return str(b);
    return format(static_cast<long long>(b), spec);  // bool is an int to format()
}

inline std::string format(const std::string& s, const std::string& spec) {
    Spec sp = parse_spec(spec);
    if (sp.type && sp.type != 's')
        throw ValueError(std::string("Unknown format code '") + sp.type + "' for object of type 'str'");
    if (sp.sign_given) throw ValueError("Sign not allowed in string format specifier");
    if (sp.z) throw ValueError("Negative zero coercion (z) not allowed in string format specifier");
    if (sp.alt) throw ValueError("Alternate form (#) not allowed in string format specifier");
    if (sp.grouping) throw ValueError(std::string("Cannot specify '") + sp.grouping + "' with 's'.");
    if (sp.align == '=') {
        if (!sp.zero_pad) throw ValueError("'=' alignment not allowed in string format specifier");
        sp.align = '<';  // "05" pads a string with zeros on the right
    }
    std::string body = s;
    if (sp.precision >= 0) {  // at most `precision` characters
        size_t chars = 0, k = 0;
        for (; k < body.size(); ++k) {
            if ((static_cast<unsigned char>(body[k]) & 0xC0) != 0x80) {
                if (chars == static_cast<size_t>(sp.precision)) break;
                ++chars;
            }
        }
        body = body.substr(0, k);
    }
    return pad("", body, sp, '<');
}

template <class T>
std::string format(const std::optional<T>& x, const std::string& spec) {
    if (x) return format(*x, spec);
    if (spec.empty()) return "None";
    throw TypeError("unsupported format string passed to NoneType.__format__");
}

inline std::string format(double x, const std::string& spec, bool as_int) {  // see str(x, as_int)
    return as_int ? format(static_cast<long long>(x), spec) : format(x, spec);
}

template <class T>
std::string format_any(const T& x, const std::string& spec) {
    if (spec.empty()) return str(x);
    return format(x, spec);
}

// --- NumPy arrays: str(a) and repr(a), as numpy/_core/arrayprint.py writes float64 ------------
// (default print options: precision 8, floatmode 'maxprec', 75 columns, summary past 1000
// elements with 3 at each edge)

// dragon4_positional / dragon4_scientific with unique=True: the shortest digits that read back,
// but no more than `precision` after the point (then correctly rounded to that many)
inline std::string np_positional(double x, int precision, char trim, int min_digits, int pad_left, int pad_right) {
    Digits dg = x == 0.0 ? Digits{"0", 1, std::signbit(x)} : shortest(x);
    dg.neg = std::signbit(x);
    const int need = std::max(0, static_cast<int>(dg.d.size()) - dg.decpt);
    if (precision >= 0 && need > precision) { dg = fixed_digits(x, precision); dg.neg = std::signbit(x); }
    std::string ip, fp;
    if (dg.decpt <= 0) {
        ip = "0";
        fp = std::string(static_cast<size_t>(-dg.decpt), '0') + dg.d;
        if (dg.d == "0") fp.clear();
    } else if (static_cast<size_t>(dg.decpt) >= dg.d.size()) {
        ip = dg.d + std::string(static_cast<size_t>(dg.decpt) - dg.d.size(), '0');
    } else {
        ip = dg.d.substr(0, static_cast<size_t>(dg.decpt));
        fp = dg.d.substr(static_cast<size_t>(dg.decpt));
    }
    while (static_cast<int>(fp.size()) < min_digits) fp += '0';
    if (trim == '.' || trim == '-' || trim == '0') {
        while (!fp.empty() && fp.back() == '0' && static_cast<int>(fp.size()) > min_digits) fp.pop_back();
    }
    std::string head = (dg.neg ? "-" : "") + ip;
    if (static_cast<int>(head.size()) < pad_left) head = std::string(static_cast<size_t>(pad_left) - head.size(), ' ') + head;
    std::string tail = fp;
    if (static_cast<int>(tail.size()) < pad_right) tail += std::string(static_cast<size_t>(pad_right) - tail.size(), ' ');
    return head + "." + tail;
}

inline std::string np_scientific(double x, int precision, char trim, int min_digits, int pad_left, int exp_digits) {
    Digits dg = x == 0.0 ? Digits{"0", 1, std::signbit(x)} : shortest(x);
    dg.neg = std::signbit(x);
    if (precision >= 0 && static_cast<int>(dg.d.size()) - 1 > precision) { dg = sig_digits(x, precision + 1); dg.neg = std::signbit(x); }
    std::string ip = dg.d.substr(0, 1), fp = dg.d.substr(1);
    const int e = x == 0.0 ? 0 : dg.decpt - 1;
    while (static_cast<int>(fp.size()) < min_digits) fp += '0';
    if (trim == '.' || trim == '-' || trim == '0') {
        while (!fp.empty() && fp.back() == '0' && static_cast<int>(fp.size()) > min_digits) fp.pop_back();
    }
    std::string head = (dg.neg ? "-" : "") + ip;
    if (static_cast<int>(head.size()) < pad_left) head = std::string(static_cast<size_t>(pad_left) - head.size(), ' ') + head;
    std::string ex = std::to_string(e < 0 ? -e : e);
    const int want = std::max(exp_digits, 2);
    while (static_cast<int>(ex.size()) < want) ex = "0" + ex;
    return head + "." + fp + "e" + (e < 0 ? "-" : "+") + ex;
}

// FloatingFormat: one style for all of an array's elements
struct FloatFormat {
    bool exp_format = false;
    int precision = 8, min_digits = 0, pad_left = 0, pad_right = 0, exp_size = -1;
    char trim = '.';

    explicit FloatFormat(const std::vector<double>& data) {
        std::vector<double> fin;
        for (double v : data) if (std::isfinite(v)) fin.push_back(v);
        double mx = 0, mn = std::numeric_limits<double>::infinity();
        bool any_nz = false;
        for (double v : fin) if (v != 0) { any_nz = true; mx = std::max(mx, std::fabs(v)); mn = std::min(mn, std::fabs(v)); }
        if (any_nz && (mx >= 1e8 || mn < 0.0001 || mx / mn > 1000.)) exp_format = true;
        if (fin.empty()) {
            pad_left = pad_right = 0; trim = '.'; exp_size = -1; min_digits = -1;
        } else if (exp_format) {
            int fr = 0, il = 0, es = 0;
            for (double v : fin) {
                const std::string t = np_scientific(v, 8, '.', 0, 0, -1);
                const size_t dot = t.find('.'), e = t.find('e');
                il = std::max(il, static_cast<int>(dot));
                fr = std::max(fr, static_cast<int>(e - dot - 1));
                es = std::max(es, static_cast<int>(t.size() - e - 1));
            }
            exp_size = es - 1;
            trim = 'k';
            precision = fr;
            min_digits = fr;
            pad_left = il;
            pad_right = exp_size + 2 + precision;
        } else {
            int fr = 0, il = 0;
            for (double v : fin) {
                const std::string t = np_positional(v, 8, '.', 0, 0, 0);
                const size_t dot = t.find('.');
                il = std::max(il, static_cast<int>(dot));
                fr = std::max(fr, static_cast<int>(t.size() - dot - 1));
            }
            pad_left = il;
            pad_right = fr;
            exp_size = -1;
            trim = '.';
            min_digits = 0;
        }
        if (fin.size() != data.size()) {
            bool neginf = false;
            for (double v : data) if (std::isinf(v) && v < 0) neginf = true;
            const int offset = pad_right + 1;
            pad_left = std::max({pad_left, 3 - offset, 3 + (neginf ? 1 : 0) - offset});
        }
    }

    std::string operator()(double x) const {
        if (!std::isfinite(x)) {
            const std::string r = std::isnan(x) ? "nan" : (x < 0 ? "-inf" : "inf");
            const int n = pad_left + pad_right + 1 - static_cast<int>(r.size());
            return std::string(static_cast<size_t>(std::max(n, 0)), ' ') + r;
        }
        if (exp_format) return np_scientific(x, precision, trim, min_digits, pad_left, exp_size);
        return np_positional(x, precision, trim, min_digits, pad_left, pad_right);
    }
};

inline std::string rstrip(const std::string& s) {
    size_t n = s.size();
    while (n && s[n - 1] == ' ') --n;
    return s.substr(0, n);
}

// _extendLine: a word onto the line, wrapping at line_width
inline void extend_line(std::string& s, std::string& line, const std::string& word, size_t width, const std::string& next_prefix) {
    bool wrap = line.size() + word.size() > width;
    if (line.size() <= next_prefix.size()) wrap = false;
    if (wrap) { s += rstrip(line) + "\n"; line = next_prefix; }
    line += word;
}

// _formatArray for rank 1 and 2. get(i, j) reads an element; cols < 0 means a vector.
template <class Get, class Fmt>
std::string format_array(Get get, long rows, long cols, const Fmt& fmt, const std::string& sep,
                         size_t line_width, const std::string& next_prefix, bool summary) {
    const long edge = 3;
    auto row_text = [&](long r, const std::string& hanging, size_t width, long n) {
        const size_t elem_width = width - std::max<size_t>(rstrip(sep).size(), 1);
        std::string s, line = hanging;
        const bool show = summary && 2 * edge < n;
        const long lead = show ? edge : 0, trail = show ? edge : n;
        for (long i = 0; i < lead; ++i) { extend_line(s, line, fmt(get(r, i)), elem_width, hanging); line += sep; }
        if (show) { extend_line(s, line, "...", elem_width, hanging); line += sep; }
        for (long i = trail; i > 1; --i) { extend_line(s, line, fmt(get(r, n - i)), elem_width, hanging); line += sep; }
        extend_line(s, line, fmt(get(r, n - 1)), elem_width, hanging);
        s += line;
        return "[" + s.substr(hanging.size()) + "]";
    };
    if (cols < 0) return row_text(0, next_prefix, line_width, rows);
    const std::string hanging = next_prefix, next_hanging = hanging + " ";
    const size_t next_width = line_width - 1;
    const std::string line_sep = rstrip(sep) + "\n";
    std::string s;
    const bool show = summary && 2 * edge < rows;
    const long lead = show ? edge : 0, trail = show ? edge : rows;
    for (long i = 0; i < lead; ++i) s += hanging + row_text(i, next_hanging, next_width, cols) + line_sep;
    if (show) s += hanging + "..." + line_sep;
    for (long i = trail; i > 1; --i) s += hanging + row_text(rows - i, next_hanging, next_width, cols) + line_sep;
    s += hanging + row_text(rows - 1, next_hanging, next_width, cols);
    return "[" + s.substr(hanging.size()) + "]";
}

// the elements FloatingFormat sees: all of them, or the corners kept by the summary
template <class Get>
std::vector<double> format_sample(Get get, long rows, long cols, bool summary) {
    std::vector<long> ri, ci;
    const long edge = 3;
    auto pick = [&](long n, std::vector<long>& out) {
        if (summary && n > 2 * edge) { for (long i = 0; i < edge; ++i) out.push_back(i); for (long i = n - edge; i < n; ++i) out.push_back(i); }
        else for (long i = 0; i < n; ++i) out.push_back(i);
    };
    pick(rows, ri);
    if (cols < 0) { std::vector<double> v; for (long i : ri) v.push_back(get(0, i)); return v; }
    pick(cols, ci);
    std::vector<double> v;
    for (long i : ri) for (long j : ci) v.push_back(get(i, j));
    return v;
}

// IntegerFormat: every element right-aligned to the widest of the largest and smallest
struct IntFormat {
    size_t width = 0;
    explicit IntFormat(const std::vector<long long>& data) {  // (the values the summary keeps)
        if (data.empty()) return;
        long long mx = std::numeric_limits<long long>::min(), mn = std::numeric_limits<long long>::max();
        for (long long v : data) { mx = std::max(mx, v); mn = std::min(mn, v); }
        width = std::max(std::to_string(mx).size(), std::to_string(mn).size());
    }
    std::string operator()(long long x) const {
        std::string t = std::to_string(x);
        return t.size() < width ? std::string(width - t.size(), ' ') + t : t;
    }
};

template <class Fmt, class Get, class Sample>
std::string array_text_with(Get get, Sample sample, long rows, long cols, bool as_repr, const char* dtype) {
    const long size = cols < 0 ? rows : rows * cols;
    const bool summary = size > 1000;
    std::string lst = "[]";
    if (size) {
        Fmt fmt(sample(summary));
        lst = as_repr ? format_array(get, rows, cols, fmt, ", ", 74, std::string(7, ' '), summary)
                      : format_array(get, rows, cols, fmt, " ", 75, " ", summary);
    }
    if (!as_repr) return lst;
    // array_repr: shape and dtype where the text doesn't show them
    const std::string shape = cols < 0 ? "(" + std::to_string(rows) + ",)"
                                       : "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    std::vector<std::string> extras;
    if ((size == 0 && cols >= 0) || summary) extras.push_back("shape=" + shape);
    if (size == 0) extras.push_back(std::string("dtype=") + dtype);
    if (extras.empty()) return "array(" + lst + ")";
    const std::string arr = "array(" + lst + ",";
    std::string extra;
    for (size_t k = 0; k < extras.size(); ++k) extra += (k ? ", " : "") + extras[k];
    extra += ")";
    const size_t nl = arr.rfind('\n');
    const size_t last = arr.size() - (nl == std::string::npos ? 0 : nl + 1);
    const std::string spacer = last + extra.size() + 1 > 75 ? "\n" + std::string(6, ' ') : " ";
    return arr + spacer + extra;
}

template <class Get>
std::string array_text(Get get, long rows, long cols, bool as_repr) {
    return array_text_with<FloatFormat>(get, [&](bool summary) { return format_sample(get, rows, cols, summary); },
                                        rows, cols, as_repr, "float64");
}

// a qt::Vec (an Eigen column) or a qt::Mat, or any Eigen expression of one (evaluated once);
// a qt::IVec prints as NumPy prints int64 arrays: [ 0  1 10]
template <class A>
std::string array_str_or_repr(const A& expr, bool as_repr) {
    const auto a = expr.eval();
    if constexpr (std::is_integral_v<typename A::Scalar> && !std::is_same_v<typename A::Scalar, bool>) {
        const long n = static_cast<long>(a.size());
        auto get = [&](long, long i) { return static_cast<long long>(a(i)); };
        auto sample = [&](bool summary) {
            std::vector<long long> v;
            auto add = [&](long i) { v.push_back(static_cast<long long>(a(i))); };
            if (summary && n > 6) { for (long i = 0; i < 3; ++i) add(i); for (long i = n - 3; i < n; ++i) add(i); }
            else for (long i = 0; i < n; ++i) add(i);
            return v;
        };
        return array_text_with<IntFormat>(get, sample, n, -1, as_repr, "int64");
    } else if constexpr (A::ColsAtCompileTime == 1) {
        return array_text([&](long, long i) { return static_cast<double>(a(i)); }, static_cast<long>(a.size()), -1, as_repr);
    } else {
        return array_text([&](long i, long j) { return static_cast<double>(a(i, j)); }, static_cast<long>(a.rows()),
                          static_cast<long>(a.cols()), as_repr);
    }
}
template <class A, class>
std::string str(const A& a) { return array_str_or_repr(a, false); }
template <class A, class, class>
std::string repr(const A& a) { return array_str_or_repr(a, true); }

// the pieces of a line of text, worked out in order (a braced list's elements are, left to right)
inline std::string concat_text(std::initializer_list<std::string> parts) {
    std::string out;
    for (const std::string& p : parts) out += p;
    return out;
}

// --- Python's float errors, where Python raises them ----------------------------------------------
// A C++ double carries on where a Python float raises: 1.0 / 0.0 is inf, math.sqrt(-1) NaN,
// math.exp(1000) inf. Since 1.0 the translation uses these for Python's floats (not in a function
// marked `# qtrans: ieee`): they raise what CPython raises, with this Python's message
// (qtrans_python.hpp), and otherwise return the very same double. (NumPy's numbers don't raise,
// so they are never used for those.) Each check is one or two compares marked unlikely, and the
// throw itself is a separate cold function, so the code around it stays as fast as it can.
#if defined(__GNUC__) || defined(__clang__)
#define QT_PY_COLD __attribute__((noinline, cold))
#define QT_PY_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#define QT_PY_COLD __declspec(noinline)
#define QT_PY_UNLIKELY(x) (x)
#else
#define QT_PY_COLD
#define QT_PY_UNLIKELY(x) (x)
#endif
namespace pyerr {
inline std::string domain_text(const msg::DomainText* t, double x, bool from_int);
[[noreturn]] QT_PY_COLD inline void raise_domain(const msg::DomainText* t, double x, bool from_int) {
    throw ValueError(domain_text(t, x, from_int));
}
[[noreturn]] QT_PY_COLD inline void raise_range() { throw OverflowError(msg::math_range); }
[[noreturn]] QT_PY_COLD inline void raise_zero_division(const char* text) { throw ZeroDivisionError(text); }
[[noreturn]] QT_PY_COLD inline void raise_overflow(const char* text) { throw OverflowError(text); }
[[noreturn]] QT_PY_COLD inline void raise_unbound(const char* name, bool top_level) {
    if (top_level) throw NameError(std::string(msg::name_error_prefix) + name + msg::name_error_suffix);
    throw UnboundLocalError(std::string(msg::unbound_local_prefix) + name + msg::unbound_local_suffix);
}
// a math function gave inf or NaN: Python's error, if it raises one here (NaN in, NaN out doesn't)
QT_PY_COLD inline void math1_check(double r, double x, const msg::DomainText* text, bool can_overflow, bool from_int) {
    if (std::isnan(r) && !std::isnan(x)) raise_domain(text, x, from_int);
    if (std::isinf(r) && std::isfinite(x)) {
        if (can_overflow) raise_range();
        raise_domain(text, x, from_int);
    }
}
// x ** y gave inf: 0.0 ** negative is ZeroDivisionError, a finite x and y overflowing OverflowError
QT_PY_COLD inline void power_check(double x, double y, bool ints) {
    if (std::isfinite(y) && x == 0.0 && y < 0.0)
        raise_zero_division(ints ? msg::int_zero_negative_power : msg::zero_negative_power);
    if (std::isfinite(x) && std::isfinite(y)) raise_overflow(msg::float_pow_overflow);
}
inline std::string domain_text(const msg::DomainText* t, double x, bool from_int) {
    const int cls = x == 0 ? 1 : x == -qt::inf ? 2 : x == qt::inf ? 3 : 0;
    const msg::DomainText& d = t[cls];
    std::string out = d.prefix;
    if (d.has_value) out += d.as_int && from_int ? std::to_string(static_cast<long long>(x)) : repr(x);
    return out + d.suffix;
}
// CPython's math_1: NaN from a non-NaN argument is a domain error; an infinity from a finite one
// is an overflow where the function can overflow, else a domain error (log(0.0), atanh(1.0))
template <class F>
double math1(F f, double x, const msg::DomainText* text, bool can_overflow, bool from_int) {
    const double r = f(x);
    if (QT_PY_UNLIKELY(!std::isfinite(r))) math1_check(r, x, text, can_overflow, from_int);  // one compare when all is well
    return r;
}
// gamma and lgamma are CPython's own (Modules/mathmodule.c: Lanczos' approximation, with the
// same coefficients and the same order of operations), not the C library's, so the values are
// Python's on every platform: the C libraries differ (Windows' lgamma(5e-324) is inf, Python's
// 744.44). A domain error at 0, -1, -2, ... (and -inf for gamma)
namespace cpython_gamma {
inline constexpr double pi = 3.141592653589793238462643383279502884197;
inline constexpr double lanczos_g = 6.024680040776729583740234375;
inline constexpr double lanczos_g_minus_half = 5.524680040776729583740234375;
inline constexpr double lanczos_num[13] = {
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408};
inline constexpr double lanczos_den[13] = {0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0,
                                           13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0};
inline constexpr double factorials[23] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0,
    6227020800.0, 87178291200.0, 1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
    121645100408832000.0, 2432902008176640000.0, 51090942171709440000.0, 1124000727777607680000.0};

inline double lanczos_sum(double x) {
    double num = 0.0, den = 0.0;
    if (x < 5.0) {
        for (int i = 13; --i >= 0;) {
            num = num * x + lanczos_num[i];
            den = den * x + lanczos_den[i];
        }
    } else {
        for (int i = 0; i < 13; i++) {
            num = num / x + lanczos_num[i];
            den = den / x + lanczos_den[i];
        }
    }
    return num / den;
}

inline double sinpi(double x) {  // sin(pi * x), exact at the integers and half-integers
    const double y = std::fmod(std::fabs(x), 2.0);
    const int n = static_cast<int>(std::round(2.0 * y));
    double r = 0.0;
    switch (n) {
        case 0: r = std::sin(pi * y); break;
        case 1: r = std::cos(pi * (y - 0.5)); break;
        case 2: r = std::sin(pi * (1.0 - y)); break;
        case 3: r = -std::cos(pi * (y - 1.5)); break;
        default: r = std::sin(pi * (y - 2.0)); break;
    }
    return std::copysign(1.0, x) * r;
}

inline double tgamma(double x) {  // (the domain errors are checked before)
    if (!std::isfinite(x)) return x;
    if (x == std::floor(x) && x <= 23.0) return factorials[static_cast<int>(x) - 1];
    const double absx = std::fabs(x);
    if (absx < 1e-20) return 1.0 / x;
    if (absx > 200.0) return x < 0.0 ? 0.0 / sinpi(x) : std::numeric_limits<double>::infinity();
    const double y = absx + lanczos_g_minus_half;
    double z;
    if (absx > lanczos_g_minus_half) {
        const double q = y - absx;
        z = q - lanczos_g_minus_half;
    } else {
        const double q = y - lanczos_g_minus_half;
        z = q - absx;
    }
    z = z * lanczos_g / y;
    double r;
    if (x < 0.0) {
        r = -pi / sinpi(absx) / absx * std::exp(y) / lanczos_sum(absx);
        r -= z * r;
        if (absx < 140.0) {
            r /= std::pow(y, absx - 0.5);
        } else {
            const double sqrtpow = std::pow(y, absx / 2.0 - 0.25);
            r /= sqrtpow;
            r /= sqrtpow;
        }
    } else {
        r = lanczos_sum(absx) / std::exp(y);
        r += z * r;
        if (absx < 140.0) {
            r *= std::pow(y, absx - 0.5);
        } else {
            const double sqrtpow = std::pow(y, absx / 2.0 - 0.25);
            r *= sqrtpow;
            r *= sqrtpow;
        }
    }
    return r;
}

inline double lgamma(double x) {  // (the domain errors are checked before)
    if (!std::isfinite(x)) return std::isnan(x) ? x : std::numeric_limits<double>::infinity();
    if (x == std::floor(x) && x <= 2.0) return 0.0;  // lgamma(1) = lgamma(2) = 0
    const double absx = std::fabs(x);
    if (absx < 1e-20) return -std::log(absx);
    double r = std::log(lanczos_sum(absx)) - lanczos_g;
    r += (absx - 0.5) * (std::log(absx + lanczos_g - 0.5) - 1);
    if (x < 0.0) r = std::log(pi) - std::log(std::fabs(sinpi(absx))) - std::log(absx) - r;  // reflection
    return r;
}
}  // namespace cpython_gamma

template <class F>
double gamma_like(F f, double x, const msg::DomainText* text, bool is_gamma, bool from_int) {
    if ((x <= 0 && std::isfinite(x) && x == std::floor(x)) || (is_gamma && x == -qt::inf))
        throw ValueError(domain_text(text, x, from_int));
    const double r = f(x);
    if (std::isinf(r) && std::isfinite(x)) throw OverflowError(msg::math_range);
    return r;
}
}  // namespace pyerr

#define QT_PY_MATH(name, call, ovf)                                                                  \
    inline double math_##name(double x) {                                                           \
        return pyerr::math1([](double v) { return call(v); }, x, msg::name##_float, ovf, false);     \
    }                                                                                               \
    inline double math_##name(long long x) {                                                        \
        return pyerr::math1([](double v) { return call(v); }, static_cast<double>(x), msg::name##_int, ovf, true); \
    }
QT_PY_MATH(sqrt, std::sqrt, false)
QT_PY_MATH(exp, std::exp, true)
QT_PY_MATH(log, std::log, false)
QT_PY_MATH(log10, std::log10, false)
QT_PY_MATH(log2, std::log2, false)
QT_PY_MATH(log1p, std::log1p, false)
QT_PY_MATH(expm1, std::expm1, true)
QT_PY_MATH(sin, std::sin, false)
QT_PY_MATH(cos, std::cos, false)
QT_PY_MATH(tan, std::tan, false)
QT_PY_MATH(asin, std::asin, false)
QT_PY_MATH(acos, std::acos, false)
QT_PY_MATH(atan, std::atan, false)
QT_PY_MATH(sinh, std::sinh, true)
QT_PY_MATH(cosh, std::cosh, true)
QT_PY_MATH(tanh, std::tanh, false)
QT_PY_MATH(asinh, std::asinh, false)
QT_PY_MATH(acosh, std::acosh, false)
QT_PY_MATH(atanh, std::atanh, false)
QT_PY_MATH(erf, std::erf, false)
QT_PY_MATH(erfc, std::erfc, false)
QT_PY_MATH(cbrt, std::cbrt, false)
QT_PY_MATH(exp2, std::exp2, true)
QT_PY_MATH(fabs, std::fabs, false)
#undef QT_PY_MATH
inline double math_gamma(double x) { return pyerr::gamma_like([](double v) { return pyerr::cpython_gamma::tgamma(v); }, x, msg::gamma_float, true, false); }
inline double math_gamma(long long x) {
    return pyerr::gamma_like([](double v) { return pyerr::cpython_gamma::tgamma(v); }, static_cast<double>(x), msg::gamma_int, true, true);
}
inline double math_lgamma(double x) { return pyerr::gamma_like([](double v) { return pyerr::cpython_gamma::lgamma(v); }, x, msg::lgamma_float, false, false); }
inline double math_lgamma(long long x) {
    return pyerr::gamma_like([](double v) { return pyerr::cpython_gamma::lgamma(v); }, static_cast<double>(x), msg::lgamma_int, false, true);
}

// math.pow(x, y): CPython's rules (errors only for finite x and y)
inline double math_pow(double x, double y) {
    if (msg::math_pow_zero_to_minus_inf_raises && x == 0.0 && y == -qt::inf) throw ValueError(msg::math_domain_pow);
    const double r = qt::pow(x, y);
    if (std::isfinite(x) && std::isfinite(y)) {
        if (std::isnan(r)) throw ValueError(msg::math_domain_pow);
        if (std::isinf(r)) {
            if (x == 0.0) throw ValueError(msg::math_domain_pow);
            throw OverflowError(msg::math_range);
        }
    }
    return r;
}
// math.fmod(x, y)
inline double math_fmod(double x, double y) {
    if (std::isinf(y) && std::isfinite(x)) return x;
    const double r = std::fmod(x, y);
    if (std::isnan(r) && !std::isnan(x) && !std::isnan(y)) throw ValueError(msg::math_domain_fmod);
    return r;
}
// a variable Python may read before it is set (where qtrans can't show it always is): its value,
// or UnboundLocalError (NameError at a program's top level) when no assignment has set it yet
template <class T>
const T& bound(const T& value, bool set, const char* name, bool top_level = false) {
    if (QT_PY_UNLIKELY(!set)) pyerr::raise_unbound(name, top_level);
    return value;
}
inline void check_bound(bool set, const char* name, bool top_level = false) {
    if (QT_PY_UNLIKELY(!set)) pyerr::raise_unbound(name, top_level);
}
// a / b of two Python numbers: ZeroDivisionError for b == 0 (ints and floats have their own words)
inline double truediv(double a, double b) {
    if (QT_PY_UNLIKELY(b == 0.0)) pyerr::raise_zero_division(msg::float_truediv_zero);
    return a / b;
}
inline double truediv(long long a, long long b) {
    if (QT_PY_UNLIKELY(b == 0)) pyerr::raise_zero_division(msg::int_truediv_zero);
    return static_cast<double>(a) / static_cast<double>(b);
}
// x ** y of Python floats (CPython's float_pow): 0.0 ** negative raises ZeroDivisionError, an
// overflow OverflowError; a negative x to a fractional power is complex in Python (NaN here)
inline double power(double x, double y, bool ints = false) {
    // (pow(x, 0.0) is 1.0 for every x, as in Python; pow(0.0, y < 0) is inf, so both errors show as inf)
    const double r = qt::pow(x, y);
    if (QT_PY_UNLIKELY(std::isinf(r))) pyerr::power_check(x, y, ints);
    return r;
}

// --- text on a Windows console -----------------------------------------------------------------
// Python writes to a Windows console window as characters (WriteConsoleW), so print() of Greek
// or Arabic text shows it as it is, whatever the console's code page. std::cout writes bytes, which the console would
// read in its code page (437, 1252, ...) and show as other characters. In main():
//     qt::py::windows_console console;
// makes std::cout and std::cerr do what Python does there: on a console, characters, written at
// once (nothing is held back, so std::exit loses nothing); anywhere else (a file, a pipe) and on
// every other system nothing changes, and the text goes out as UTF-8 bytes, as Python writes it
// in UTF-8 mode (python -X utf8, and the default from Python 3.15).

// UTF-8 to UTF-16: the complete characters at the start of [p, p + n), appended to out; the
// bytes used (a character cut off at the end waits for the rest)
inline std::size_t utf8_to_utf16(const char* p, std::size_t n, std::u16string& out) {
    std::size_t k = 0;
    while (k < n) {
        const unsigned char c = static_cast<unsigned char>(p[k]);
        const std::size_t len = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (k + len > n) break;
        char32_t v = len == 1 ? c : (c & (0x7Fu >> len));
        for (std::size_t i = 1; i < len; ++i) v = (v << 6) | (static_cast<unsigned char>(p[k + i]) & 0x3Fu);
        if (v >= 0x10000) {
            v -= 0x10000;
            out += static_cast<char16_t>(0xD800 + (v >> 10));
            out += static_cast<char16_t>(0xDC00 + (v & 0x3FF));
        } else {
            out += static_cast<char16_t>(v);
        }
        k += len;
    }
    return k;
}

#if defined(_WIN32)
class console_buf : public std::streambuf {
public:
    explicit console_buf(void* handle) : handle_(handle) {}
    ~console_buf() override { flush_all(); }

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::not_eof(ch);
        pending_ += traits_type::to_char_type(ch);
        flush_all();
        return ch;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        pending_.append(s, static_cast<std::size_t>(n));
        flush_all();
        return n;
    }
    int sync() override {
        flush_all();
        return 0;
    }

private:
    void flush_all() {
        std::u16string text;
        const std::size_t used = utf8_to_utf16(pending_.data(), pending_.size(), text);
        pending_.erase(0, used);
        const char16_t* at = text.data();
        std::size_t left = text.size();
        while (left > 0) {
            unsigned long done = 0;
            const unsigned long part = left > 8192 ? 8192ul : static_cast<unsigned long>(left);
            if (!WriteConsoleW(handle_, at, part, &done, nullptr) || done == 0) break;
            at += done;
            left -= done;
        }
    }
    void* handle_;
    std::string pending_;
};
#endif

class windows_console {
public:
    windows_console() {
#if defined(_WIN32)
        attach(std::cout, static_cast<unsigned long>(-11), out_, old_out_);  // STD_OUTPUT_HANDLE
        attach(std::cerr, static_cast<unsigned long>(-12), err_, old_err_);  // STD_ERROR_HANDLE
#endif
    }
    ~windows_console() {
#if defined(_WIN32)
        std::cout.flush();
        std::cerr.flush();
        if (out_) std::cout.rdbuf(old_out_);
        if (err_) std::cerr.rdbuf(old_err_);
        delete out_;
        delete err_;
#endif
    }
    windows_console(const windows_console&) = delete;
    windows_console& operator=(const windows_console&) = delete;

private:
#if defined(_WIN32)
    static void attach(std::ostream& os, unsigned long which, console_buf*& buf, std::streambuf*& old) {
        void* h = GetStdHandle(which);
        unsigned long mode = 0;
        if (h == nullptr || h == reinterpret_cast<void*>(-1) || !GetConsoleMode(h, &mode)) return;  // not a console
        os.flush();
        buf = new console_buf(h);
        old = os.rdbuf(buf);
    }
    console_buf* out_ = nullptr;
    console_buf* err_ = nullptr;
    std::streambuf* old_out_ = nullptr;
    std::streambuf* old_err_ = nullptr;
#endif
};

}  // namespace py
}  // namespace qt
