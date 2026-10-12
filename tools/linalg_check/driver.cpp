// The per-operation driver of tools/linalg_check: reads one operation per line,
//   <op> <rows> <cols> <entries row by row...> [<rows> <cols> <entries...>]   (the second: solve's b)
// runs qt's numpy.linalg routine on it with the audit compiled in (QTRANS_LINALG_AUDIT), and prints
//   OK <audited> <worst ratio> <values...>      the result, flattened (eigh: w then V; det: one value)
//   EXC <LinAlgError | ShapeError | Error>
// followed on the same line by " | <operation the worst ratio came from>".
#include <cstdio>
#include <iostream>
#include <string>

#include "qtrans_arrays.hpp"

static qt::Mat read_mat(std::istream& in) {
    long r = 0, c = 0;
    in >> r >> c;
    qt::Mat m(r, c);
    for (long i = 0; i < r; ++i)
        for (long j = 0; j < c; ++j) {
            std::string t;
            in >> t;
            m(i, j) = std::stod(t);
        }
    return m;
}

static void put(const qt::Mat& m) {
    for (Eigen::Index i = 0; i < m.rows(); ++i)
        for (Eigen::Index j = 0; j < m.cols(); ++j) std::printf(" %.17g", m(i, j));
}

int main() {
    std::ios::sync_with_stdio(false);
    std::string op;
    while (std::cin >> op) {
        const qt::Mat a = read_mat(std::cin);
        qt::Mat b;
        if (op == "solve") b = read_mat(std::cin);
        qt::audit::reset();
        std::string out;
        try {
            if (op == "solve") {
                const qt::Mat x = qt::solve(a, b);
                std::printf("OK %ld %.17g", qt::audit::state().count, qt::audit::state().worst);
                put(x);
            } else if (op == "inv") {
                const qt::Mat x = qt::inv(a);
                std::printf("OK %ld %.17g", qt::audit::state().count, qt::audit::state().worst);
                put(x);
            } else if (op == "det") {
                const double d = qt::det(a);
                std::printf("OK %ld %.17g %.17g", qt::audit::state().count, qt::audit::state().worst, d);
            } else if (op == "cholesky") {
                const qt::Mat l = qt::cholesky(a);
                std::printf("OK %ld %.17g", qt::audit::state().count, qt::audit::state().worst);
                put(l);
            } else if (op == "eigh") {
                const auto [w, v] = qt::eigh(a);
                std::printf("OK %ld %.17g", qt::audit::state().count, qt::audit::state().worst);
                put(qt::Mat(w.matrix().transpose().array()));
                put(v);
            } else if (op == "pinv") {
                const qt::Mat x = qt::pinv(a);
                std::printf("OK %ld %.17g", qt::audit::state().count, qt::audit::state().worst);
                put(x);
            } else {
                std::printf("EXC UnknownOperation");
            }
        } catch (const qt::linalg_error&) {
            std::printf("EXC LinAlgError");
        } catch (const qt::shape_error&) {
            std::printf("EXC ShapeError");
        } catch (const std::exception&) {
            std::printf("EXC Error");
        }
        std::printf(" | %s\n", qt::audit::state().op[0] ? qt::audit::state().op : "-");
        std::fflush(stdout);
    }
}
