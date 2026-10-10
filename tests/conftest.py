"""Shared pytest settings.

Without a C++ compiler or Eigen, the compiled tests skip, so a run can pass while checking
none of the C++. CI sets QTRANS_REQUIRE_CXX=1, which turns any skipped test into a failure.
"""
import os

_skipped = []


def pytest_runtest_logreport(report):
    if report.skipped:
        _skipped.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("QTRANS_REQUIRE_CXX") and _skipped and exitstatus == 0:
        print(f"\nQTRANS_REQUIRE_CXX is set but {len(_skipped)} test(s) were skipped, "
              f"e.g. {_skipped[0]}")
        session.exitstatus = 1
