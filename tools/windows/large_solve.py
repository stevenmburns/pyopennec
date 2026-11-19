"""A whip-sized solve through the extension, against onec.exe on the same deck.

    python large_solve.py onec.exe

4,400 unknowns on python.exe's own stack and libgomp's worker stacks: the size
of the W8IO whip, where a stack or address-space limit of the DLL would show
and a 100-segment unit test would not.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import nec_output  # noqa: E402
from pyopennec import porty  # noqa: E402

WIRES = 44
SEGS = 100
FREQ = 146.0


def main(onec: str) -> int:
    wires = [(k + 1, SEGS, (0.25 * k, 0.0, -0.5), (0.25 * k, 0.0, 0.5), 0.002) for k in range(WIRES)]
    feed = SEGS // 2
    z = complex(1.0 / porty.Structure(wires, ports=[(1, feed)]).solve([FREQ]).y[0, 0, 0])
    gw = "\n".join(f"GW {t} {n} {a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]} {r}" for t, n, a, b, r in wires)
    deck = f"CM large\nCE\n{gw}\nGE 0\nEX 0 1 {feed} 0 1 0\nFR 0 1 0 0 {FREQ}\nXQ\nEN\n"
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "large.nec").write_text(deck)
        subprocess.run([onec, "-i", str(d / "large.nec"), "-o", str(d / "large.out")], check=True, timeout=1200)
        rows = nec_output.input_parameters((d / "large.out").read_text(errors="replace"))
    ref = rows[0][5]
    err = abs(z - ref) / abs(ref)
    print(f"{WIRES * SEGS} unknowns: porty {z:.5f}, onec {ref:.5f}, |dZ|/|Z| = {err:.2e}")
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
