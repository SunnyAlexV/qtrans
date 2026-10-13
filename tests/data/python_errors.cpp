#include "qtrans_print.hpp"
#include <cstdio>
#include <iostream>
using namespace qt;
template <class F> std::string run(F f) {
    try { return py::repr(f()); } catch (const Exception& e) { return std::string(e.type_name()) + ": " + e.what(); }
}
int main() {
    double v;
    std::string name, tv, tw;
    while (std::cin >> name >> tv) {
        v = std::strtod(tv.c_str(), nullptr);
        std::string r;
#define U(n) if (name == #n) r = run([&] { return py::math_##n(v); }); if (name == #n "_i") r = run([&] { return py::math_##n(static_cast<long long>(v)); });
        U(sqrt) U(exp) U(log) U(log10) U(log2) U(log1p) U(expm1) U(sin) U(cos) U(tan) U(asin) U(acos) U(atan)
        U(sinh) U(cosh) U(tanh) U(asinh) U(acosh) U(atanh) U(erf) U(erfc) U(cbrt) U(exp2) U(fabs) U(gamma) U(lgamma)
        double w;
        if (name == "pow" || name == "mpow" || name == "fmod" || name == "div" || name == "ipow") {
            std::cin >> tw;
            w = std::strtod(tw.c_str(), nullptr);
            if (name == "pow") r = run([&] { return py::power(v, w); });
            if (name == "ipow") r = run([&] { return py::power(v, w, true); });
            if (name == "mpow") r = run([&] { return py::math_pow(v, w); });
            if (name == "fmod") r = run([&] { return py::math_fmod(v, w); });
            if (name == "div") r = run([&] { return py::truediv(v, w); });
        }
        std::cout << r << "\n";
    }
}
