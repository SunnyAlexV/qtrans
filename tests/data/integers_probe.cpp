#include "qtrans_random.hpp"
#include <cstdio>
#include <iostream>
#include <string>
int main() {
    std::string op;
    long long seed;
    while (std::cin >> op >> seed) {
        qt::Rng g(seed);
        long long lo, hi, n; std::cin >> lo >> hi >> n;
        try {
            if (n < 0) std::printf("%lld", g.integers(lo, hi));
            else { auto v = g.integers(lo, hi, n); for (long i = 0; i < v.size(); ++i) std::printf("%lld ", v(i)); }
            const double r = g.random(); const unsigned u = g.next_uint32(); std::printf("| %.17g %u\n", r, u);
        } catch (const qt::Exception& e) { std::printf("%s: %s\n", e.type_name(), e.what()); }
    }
}
