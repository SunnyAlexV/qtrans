// Probes qt::py::fsum against math.fsum (tests/test_fsum.py): one list of hex floats per line.
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qtrans_runtime.hpp"

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::vector<double> xs;
        std::string t;
        while (in >> t) xs.push_back(std::strtod(t.c_str(), nullptr));
        try {
            std::printf("%a\n", qt::py::fsum(xs));
        } catch (const qt::Exception& e) {
            std::printf("%s: %s\n", e.type_name(), e.what());
        }
    }
}
