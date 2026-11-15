"""A wire whose built segments have a zero, subnormal or NaN length is refused
by name, never solved.

Without the check, OpenNEC gives such a segment a 0/0 = NaN length (its Newton
square root), NaN passes its own "too small" test, and over Sommerfeld ground
the fill reaches romberg_integrate_sommerfeld's assert(s >= 0): the PROCESS
aborts. Elsewhere it returns a NaN answer silently. Each case runs in a
subprocess, so a regression shows as a failed test, not a dead test runner.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

GROUNDS = {
    "free": "None",
    "perfect": '"perfect"',
    "reflection": '("reflection", 10.0, 0.01)',
    "sommerfeld": '("sommerfeld", 10.0, 0.01)',
}

# (wire tuple as Python source, what makes its segments degenerate)
CASES = {
    # |p1 - p0|**2 underflows to 0, so every segment's length is 0/0
    "subnormal": "(1, 2, (0.0, 0.0, 1.0), (2.3e-162, 0.0, 1.0), 1e-170)",
    # the segment midpoint rounds onto an end: one segment exactly 0 long
    "nextafter-2": f"(1, 2, (0.0, 0.0, 1.0), (0.0, 0.0, {float(np.nextafter(1.0, 2.0))!r}), 1e-6)",
    "nextafter-3": f"(1, 3, (0.0, 0.0, 1.0), (0.0, 0.0, {float(np.nextafter(1.0, 2.0))!r}), 1e-6)",
}


def _run(wire_src: str, ground_src: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        f"""
        from pyopennec import porty
        try:
            st = porty.Structure([{wire_src}], ports=[(1, 1)], ground={ground_src})
            sol = st.solve([14.0])
            print("SOLVED", complex(sol.y[0, 0, 0]))
        except Exception as ex:
            print("REFUSED", type(ex).__name__, ex)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("ground", sorted(GROUNDS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_degenerate_segment_is_refused_by_name(case, ground):
    r = _run(CASES[case], GROUNDS[ground])
    assert r.returncode == 0, (r.returncode, r.stderr[-400:])  # no abort, no signal
    out = r.stdout.strip()
    assert out.startswith("REFUSED"), out
    assert "must be finite and > 0" in out, out


def test_an_ordinary_short_wire_still_solves():
    # The check refuses degenerate segments, not short ones: 1 mm in 3 segments.
    r = _run("(1, 3, (0.0, 0.0, 1.0), (0.0, 0.0, 1.001), 1e-5)", GROUNDS["sommerfeld"])
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.startswith("SOLVED"), r.stdout
