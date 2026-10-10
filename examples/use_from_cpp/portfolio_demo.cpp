// Using translated array code from C++: build a covariance matrix, then compute the
// minimum-variance portfolio, its risk contributions, and Black-Litterman weights.
//
// Build (from this folder; Eigen's include folder must be on the path):
//   g++ -std=c++17 -O2 -I../cpp_out -I/usr/include/eigen3 portfolio_demo.cpp -o portfolio_demo
// Windows + conda:   cl /std:c++17 /O2 /EHsc /I..\cpp_out /I%CONDA_PREFIX%\Library\include\eigen3 portfolio_demo.cpp

#include <cstdio>

#include "portfolio.hpp"  // includes qtrans_arrays.hpp (Eigen) first

static void print(const char* label, const qt::Vec& v) {
    std::printf("%-26s", label);
    for (qt::Index i = 0; i < v.size(); ++i) std::printf(" %9.5f", v(i));
    std::printf("\n");
}

int main() {
    // annualised covariance of four assets (vols 15%, 20%, 25%, 10%)
    const qt::Mat cov = qt::mat({{0.0225, 0.0120, 0.0075, 0.0030},
                                 {0.0120, 0.0400, 0.0150, 0.0040},
                                 {0.0075, 0.0150, 0.0625, 0.0050},
                                 {0.0030, 0.0040, 0.0050, 0.0100}});
    const qt::Vec w_mkt = qt::vec({0.35, 0.25, 0.15, 0.25});

    const qt::Vec w_mv = portfolio::min_variance_weights(cov);
    print("min-variance weights", w_mv);
    print("risk contributions", portfolio::risk_contributions(w_mv, cov));
    std::printf("%-26s %9.5f\n", "min-variance volatility", portfolio::portfolio_vol(w_mv, cov));

    // one view: asset 3 outperforms asset 1 by 2% a year, with 1% standard deviation
    const qt::Mat P = qt::mat({{-1.0, 0.0, 1.0, 0.0}});
    const qt::Vec Q = qt::vec({0.02});
    const qt::Vec omega = qt::vec({0.0001});
    auto [mu, post_cov] = portfolio::black_litterman(cov, w_mkt, P, Q, omega);
    print("Black-Litterman returns", mu);
    print("Black-Litterman weights", portfolio::bl_weights(cov, w_mkt, P, Q, omega));
    std::printf("%-26s %9.5f\n", "posterior vol of asset 3", std::sqrt(post_cov(2, 2)));

    // NumPy raises ValueError for a negative view variance; so does the C++
    try {
        portfolio::black_litterman(cov, w_mkt, P, Q, qt::vec({-0.0001}));
    } catch (const std::invalid_argument& e) {
        std::printf("negative view variance     rejected: %s\n", e.what());
    }
    return 0;
}
