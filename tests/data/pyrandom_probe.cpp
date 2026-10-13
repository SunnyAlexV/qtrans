// Probes qt::py::Random against Python's random module (tests/test_pyrandom.py): one call per
// input line, one result per output line (numbers as hex floats or ints, lists space separated,
// errors as "Class: message").
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qtrans_pyrandom.hpp"

static std::vector<long long> iota_list(long long n) {
    std::vector<long long> v;
    for (long long i = 0; i < n; ++i) v.push_back(i);
    return v;
}

static std::string ints(const std::vector<long long>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s.empty() ? "-" : s;
}

static std::string hex(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%a", x);
    return b;
}

int main() {
    qt::py::Random& g = qt::py::random_module();
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string op;
        in >> op;
        try {
            if (op == "seed") { long long a; in >> a; g.seed(a); std::cout << "ok\n"; }
            else if (op == "random") std::cout << hex(g.random()) << "\n";
            else if (op == "getrandbits") { int k; in >> k; std::cout << g.getrandbits(k) << "\n"; }
            else if (op == "randrange1") { long long a; in >> a; std::cout << g.randrange(a) << "\n"; }
            else if (op == "randrange2") { long long a, b; in >> a >> b; std::cout << g.randrange(a, b) << "\n"; }
            else if (op == "randrange3") { long long a, b, s; in >> a >> b >> s; std::cout << g.randrange(a, b, s) << "\n"; }
            else if (op == "randint") { long long a, b; in >> a >> b; std::cout << g.randint(a, b) << "\n"; }
            else if (op == "uniform") { double a, b; in >> a >> b; std::cout << hex(g.uniform(a, b)) << "\n"; }
            else if (op == "gauss") { double m, s; in >> m >> s; std::cout << hex(g.gauss(m, s)) << "\n"; }
            else if (op == "normalvariate") { double m, s; in >> m >> s; std::cout << hex(g.normalvariate(m, s)) << "\n"; }
            else if (op == "lognormvariate") { double m, s; in >> m >> s; std::cout << hex(g.lognormvariate(m, s)) << "\n"; }
            else if (op == "expovariate") { double l; in >> l; std::cout << hex(g.expovariate(l)) << "\n"; }
            else if (op == "triangular2") { double a, b; in >> a >> b; std::cout << hex(g.triangular(a, b)) << "\n"; }
            else if (op == "triangular3") { double a, b, m; in >> a >> b >> m; std::cout << hex(g.triangular(a, b, m)) << "\n"; }
            else if (op == "choice") { long long n; in >> n; std::cout << g.choice(iota_list(n)) << "\n"; }
            else if (op == "shuffle") { long long n; in >> n; auto v = iota_list(n); g.shuffle(v); std::cout << ints(v) << "\n"; }
            else if (op == "sample") { long long n, k; in >> n >> k; std::cout << ints(g.sample(iota_list(n), k)) << "\n"; }
            else if (op == "choices") { long long n, k; in >> n >> k; std::cout << ints(g.choices(iota_list(n), k)) << "\n"; }
            else if (op == "wchoices") {
                long long n, k; in >> n >> k;
                std::vector<double> w;
                for (long long i = 0; i < n; ++i) { double x; in >> x; w.push_back(x); }
                std::cout << ints(g.choices(iota_list(n), w, k)) << "\n";
            }
            else if (op == "state") {  // what verify compares: the next 32 bits, gauss's saved value
                std::cout << g.peek32() << " " << (g.gauss_next() ? hex(*g.gauss_next()) : std::string("None")) << "\n";
            }
            else std::cout << "?\n";
        } catch (const qt::Exception& e) {
            std::cout << e.type_name() << ": " << e.what() << "\n";
        }
    }
}
