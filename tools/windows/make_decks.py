"""Two synthetic decks the size of the W8IO benchmark structures.

    python make_decks.py OUTDIR

synth_cage.nec  4,392 segments: a whip made of a 24-wire cage (free space).
synth_lpy.nec   1,353 segments: a 27-element log-periodic with 25 crossed TL
                cards and the extended kernel.

Invented geometry, generated here, so nothing private is in the repository.
They set the matrix size and the shape of the work (fill, factor, TL network,
frequency loop); their answers mean nothing as antennas.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_DECKS = ["example5.nec", "6mYagi-Onec.nec", "GH8n2.nec"]


def cage() -> str:
    wires = 24
    per_wire = 182
    feed = 4392 - wires * per_wire
    radius = 0.04
    bottom, top = 0.03, 0.55
    lines = ["CM synthetic 24-wire cage whip, 4392 segments", "CE"]
    tag = 1
    lines.append(f"GW {tag} {feed} 0 0 0 0 0 {bottom} 0.001")
    for k in range(wires):
        a = 2 * math.pi * k / wires
        x, y = radius * math.cos(a), radius * math.sin(a)
        tag += 1
        lines.append(f"GW {tag} {per_wire // 3} 0 0 {bottom} {x:.6f} {y:.6f} {bottom + 0.05} 0.0005")
        tag += 1
        lines.append(
            f"GW {tag} {per_wire - 2 * (per_wire // 3)} {x:.6f} {y:.6f} {bottom + 0.05} "
            f"{x:.6f} {y:.6f} {top - 0.05} 0.0005"
        )
        tag += 1
        lines.append(f"GW {tag} {per_wire // 3} {x:.6f} {y:.6f} {top - 0.05} 0 0 {top} 0.0005")
    lines += ["GE 0", f"EX 0 1 {feed // 2 + 1} 0 1 0", "FR 0 3 0 0 300 50", "XQ", "EN"]
    return "\n".join(lines) + "\n"


def lpy() -> str:
    tau = 0.965
    half = 0.40
    x = 0.0
    lines = ["CM synthetic log-periodic, 1353 segments", "CE"]
    tags = []
    tag = 0
    for k in range(27):
        tag += 1
        tags.append(tag)
        nseg = 27 if k == 0 else 51
        lines.append(f"GW {tag} {nseg} {x:.5f} {-half:.5f} 0 {x:.5f} {half:.5f} 0 0.002")
        x += 0.14 * 2 * half
        half *= tau
    lines += ["GE 0", "EK"]
    for a, b in zip(tags[1:-1], tags[2:], strict=True):
        lines.append(f"TL {a} 26 {b} 26 -100 0 0 0 0 0")
    lines += [f"EX 0 {tags[-1]} 26 0 1 0", "FR 0 9 0 0 200 40", "RP 0 1 37 1000 90 0 0 10", "XQ", "EN"]
    return "\n".join(lines) + "\n"


def main(out: str) -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "synth_cage.nec").write_text(cage(), newline="\n")
    (d / "synth_lpy.nec").write_text(lpy(), newline="\n")
    # OpenNEC's own public decks: its golden matrix and three examples
    for f in sorted((ROOT / "opennec" / "tests" / "golden_tests" / "decks").glob("*.nec")):
        shutil.copy(f, d / f.name)
    for name in REPO_DECKS:
        shutil.copy(ROOT / "opennec" / "examples" / name, d / name)


if __name__ == "__main__":
    main(sys.argv[1])
