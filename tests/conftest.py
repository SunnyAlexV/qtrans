"""Shared pytest settings.

Without a C++ compiler or Eigen, the compiled tests skip, so a run can pass while checking
none of the C++. CI sets QTRANS_REQUIRE_CXX=1, which turns any skipped test into a failure.
Tests that drive a real browser (Playwright) count only where QTRANS_REQUIRE_BROWSER=1 too: the
reference job installs one; the other jobs skip them.
"""
import os

_skipped = []


def pytest_runtest_logreport(report):
    if report.skipped:
        if "in_a_browser" in report.nodeid and not os.environ.get("QTRANS_REQUIRE_BROWSER"):
            return
        _skipped.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("QTRANS_REQUIRE_CXX") and _skipped and exitstatus == 0:
        print(f"\nQTRANS_REQUIRE_CXX is set but {len(_skipped)} test(s) were skipped, "
              f"e.g. {_skipped[0]}")
        session.exitstatus = 1
