// npmath_probe.cpp: the runtime's NumPy maths (qtrans_npmath.hpp) on values read from stdin, for
// tests/test_npmath.py and tools/npmath_versions.py to compare with NumPy bit for bit.
//
//   paths                      -> one line per function: "<name> <svml_ha|svml_la|numpy_tang|numpy_own|libm>"
//   unary <fn> <n> <x...>      -> f(x) over the whole array (NumPy's loop)
//   scalar <fn> <n> <x...>     -> f(x) one value at a time (np.f on a number)
//   binary <fn> <n> <x...> <y...>   -> np.power / np.arctan2 of two arrays
//   power <op|ufunc> <s> <n> <x...> -> x ** s (op) or np.power(x, s) (ufunc)
//   base <b> <n> <x...>        -> b ** x
// Values travel as raw little-endian doubles (stdin binary after the text header line).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "qtrans_npmath.hpp"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

using qt::npm::Fn;

static const char* const kNames[] = {"exp", "log", "exp2", "log2", "log10", "expm1", "log1p", "cbrt", "tan",
                                     "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh",
                                     "arctanh", "sin", "cos", "power", "arctan2"};

static Fn fn_of(const std::string& s) {
    for (int i = 0; i < static_cast<int>(Fn::count_); ++i)
        if (s == kNames[i]) return static_cast<Fn>(i);
    std::fprintf(stderr, "unknown function %s\n", s.c_str());
    std::exit(2);
}

static std::vector<double> read(long n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    if (n > 0 && std::fread(v.data(), 8, static_cast<std::size_t>(n), stdin) != static_cast<std::size_t>(n)) std::exit(3);
    return v;
}

static double scalar(Fn f, double x) {
    switch (f) {
        case Fn::exp: return qt::npm::exp(x);
        case Fn::log: return qt::npm::log(x);
        case Fn::exp2: return qt::npm::exp2(x);
        case Fn::log2: return qt::npm::log2(x);
        case Fn::log10: return qt::npm::log10(x);
        case Fn::expm1: return qt::npm::expm1(x);
        case Fn::log1p: return qt::npm::log1p(x);
        case Fn::cbrt: return qt::npm::cbrt(x);
        case Fn::tan: return qt::npm::tan(x);
        case Fn::arcsin: return qt::npm::arcsin(x);
        case Fn::arccos: return qt::npm::arccos(x);
        case Fn::arctan: return qt::npm::arctan(x);
        case Fn::sinh: return qt::npm::sinh(x);
        case Fn::cosh: return qt::npm::cosh(x);
        case Fn::tanh: return qt::npm::tanh(x);
        case Fn::arcsinh: return qt::npm::arcsinh(x);
        case Fn::arccosh: return qt::npm::arccosh(x);
        case Fn::arctanh: return qt::npm::arctanh(x);
        case Fn::sin: return qt::npm::sin(x);
        case Fn::cos: return qt::npm::cos(x);
        default: return 0.0;
    }
}

int main() {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);  // raw doubles, not text
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    char line[256];
    if (!std::fgets(line, sizeof line, stdin)) return 1;
    char a[64] = {0}, b[64] = {0};
    double s = 0.0;
    long n = 0;
    std::vector<double> out;
    if (std::sscanf(line, "paths%c", a) == 1 || std::strncmp(line, "paths", 5) == 0) {
        for (int i = 0; i < static_cast<int>(Fn::count_); ++i)
            std::printf("%s %s\n", kNames[i], qt::npm::path_name(static_cast<Fn>(i)));
        std::printf("numpy_version %ld\n", qt::npm::numpy_version);
        return 0;
    } else if (std::sscanf(line, "unary %63s %ld", a, &n) == 2) {
        std::vector<double> x = read(n);
        out.resize(x.size());
        qt::npm::apply(fn_of(a), out.data(), x.data(), n);
    } else if (std::sscanf(line, "scalar %63s %ld", a, &n) == 2) {
        std::vector<double> x = read(n);
        const Fn f = fn_of(a);
        for (double v : x) out.push_back(scalar(f, v));
    } else if (std::sscanf(line, "binary %63s %ld", a, &n) == 2) {
        std::vector<double> x = read(n), y = read(n);
        out.resize(x.size());
        qt::npm::apply2(fn_of(a), out.data(), x.data(), y.data(), n);
    } else if (std::sscanf(line, "power %63s %lf %ld", a, &s, &n) == 3) {
        std::vector<double> x = read(n);
        out.resize(x.size());
        qt::npm::power_scalar(out.data(), x.data(), s, n, std::string(a) == "op");
    } else if (std::sscanf(line, "base %lf %ld", &s, &n) == 2) {
        std::vector<double> x = read(n);
        out.resize(x.size());
        qt::npm::power_base(out.data(), s, x.data(), n);
    } else {
        std::fprintf(stderr, "bad request: %s", line);
        return 2;
    }
    (void)b;
    std::fwrite(out.data(), 8, out.size(), stdout);
    return 0;
}
