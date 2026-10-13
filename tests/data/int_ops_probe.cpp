// Probes qt::IVec arithmetic and printing against NumPy (tests/test_int_arrays.py): each line is
// an operation and two int arrays; the answer is the result printed as NumPy prints it.
#include <iostream>
#include <sstream>
#include <string>
#include "qtrans_arrays.hpp"
#include "qtrans_print.hpp"

static qt::IVec read(std::istringstream& in) {
    long long n; in >> n;
    qt::IVec v(n);
    for (long long i = 0; i < n; ++i) in >> v(i);
    return v;
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string op;
        in >> op;
        const qt::IVec a = read(in), b = read(in);
        long long k; in >> k;
        try {
            std::string r;
            if (op == "add") r = qt::py::str(qt::iadd(a, b));
            else if (op == "sub") r = qt::py::str(qt::isub(a, b));
            else if (op == "mul") r = qt::py::str(qt::imul(a, b));
            else if (op == "floordiv") r = qt::py::str(qt::ifloordiv(a, b));
            else if (op == "mod") r = qt::py::str(qt::imod(a, b));
            else if (op == "floordivk") r = qt::py::str(qt::ifloordiv(a, k));
            else if (op == "modk") r = qt::py::str(qt::imod(a, k));
            else if (op == "pow") r = qt::py::str(qt::ipow(a, b));
            else if (op == "powk") r = qt::py::str(qt::ipow(a, k));
            else if (op == "kpow") r = qt::py::str(qt::ipow(k, a));
            else if (op == "neg") r = qt::py::str(qt::ineg(a));
            else if (op == "abs") r = qt::py::str(qt::iabs(a));
            else if (op == "sum") r = std::to_string(qt::isum(a));
            else if (op == "cumsum") r = qt::py::str(qt::icumsum(a));
            else if (op == "max") r = std::to_string(qt::imax(a));
            else if (op == "argmax") r = std::to_string(qt::iargmax(a));
            else if (op == "repr") r = qt::py::repr(a);
            else if (op == "str") r = qt::py::str(a);
            else if (op == "arange") r = qt::py::str(qt::iarange(a(0), a(1), a(2)));
            std::string flat;  // (one line per answer: NumPy wraps long arrays)
            for (char c : r) flat += c == '\n' ? '|' : c;
            std::cout << flat << "\n";
        } catch (const qt::Exception& e) {
            std::cout << e.type_name() << ": " << e.what() << "\n";
        }
    }
}
