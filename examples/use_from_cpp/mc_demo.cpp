// Running translated Monte Carlo code from C++. With the same seed, qt::Rng draws exactly the
// numbers np.random.default_rng draws, so these prices equal the Python ones.
//
// Build (from this folder):
//   g++ -std=c++17 -O2 -I../cpp_out -I/usr/include/eigen3 mc_demo.cpp -o mc_demo

#include <cstdio>

#include "monte_carlo.hpp"

int main() {
    auto [eu, eu_se] = monte_carlo::european_mc(100.0, 100.0, 0.05, 0.2, 1.0, 200000, 42);
    std::printf("European call (antithetic)   %.10f  +- %.6f\n", eu, eu_se);

    auto [asian, asian_se] = monte_carlo::asian_call_mc(100.0, 100.0, 0.05, 0.3, 1.0, 12, 100000, 9);
    std::printf("Asian call (control variate) %.10f  +- %.6f\n", asian, asian_se);

    std::printf("Down-and-out call, B = 90    %.10f\n",
                monte_carlo::down_and_out_call_mc(100.0, 100.0, 90.0, 0.05, 0.25, 1.0, 50, 100000, 5));
    std::printf("Heston call                  %.10f\n",
                monte_carlo::heston_call_mc(100.0, 100.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 100, 50000, 13));

    // a generator shared across calls, as a Python Generator object would be
    qt::Rng rng(2026);
    qt::Mat paths = monte_carlo::gbm_paths(100.0, 0.05, 0.2, 1.0, 4, 3, rng);
    std::printf("first simulated path         %.6f %.6f %.6f %.6f %.6f\n",
                paths(0, 0), paths(0, 1), paths(0, 2), paths(0, 3), paths(0, 4));
    std::printf("generator's next raw output  %llu\n", static_cast<unsigned long long>(rng.next_uint64()));
    return 0;
}
