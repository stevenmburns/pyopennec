"""Read the numbers out of onec / nec2c .out reports, for the CLI cross-checks.

Only the test suite parses text; the package itself never does.
"""
import cmath
import math
import re

NUM = r"[-+]?\d*\.\d+E[-+]\d+"
FREQ = re.compile(r"FREQUENCY[=:]\s*([0-9.E+-]+)\s*MHZ", re.I)


def _nums(line):
    return [float(v) for v in re.findall(NUM, line)]


def input_parameters(text):
    """[(freq_mhz, tag, seg, V, I, Z, Y, P)] -- every ANTENNA INPUT PARAMETERS row."""
    rows, freq, inside = [], None, False
    for line in text.splitlines():
        m = FREQ.search(line)
        if m:
            freq = float(m.group(1))
        if "ANTENNA INPUT PARAMETERS" in line:
            inside = True
            continue
        if inside:
            m = re.match(r"^\s*(\d+)\s+(\d+)\s*([-+ ]?\d\.\d+E[-+]\d+.*)$", line)
            if m:
                v = _nums(m.group(3))
                rows.append((freq, int(m.group(1)), int(m.group(2)), complex(v[0], v[1]),
                             complex(v[2], v[3]), complex(v[4], v[5]), complex(v[6], v[7]), v[8]))
            elif rows and line.strip() == "" and rows[-1][0] == freq:
                inside = False
    return rows


def currents(text):
    """[[(seg, tag, I)]] -- one list per CURRENTS AND LOCATION block."""
    blocks, cur, inside = [], None, False
    for line in text.splitlines():
        if "CURRENTS AND LOCATION" in line:
            cur, inside = [], True
            blocks.append(cur)
            continue
        if inside:
            m = re.match(r"^\s*(\d+)\s+(\d+)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[\d.]+\s+(" + NUM + r")\s*(" + NUM + ")", line)
            if m:
                cur.append((int(m.group(1)), int(m.group(2)), complex(float(m.group(3)), float(m.group(4)))))
            elif cur and line.strip() == "":
                inside = False
    return blocks


def patterns(text):
    """[[(theta, phi, vert, hor, total)]] -- one list per RADIATION PATTERNS block."""
    blocks, cur, inside = [], None, False
    for line in text.splitlines():
        if "RADIATION PATTERNS" in line:
            cur, inside = [], True
            blocks.append(cur)
            continue
        if inside:
            m = re.match(r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+[\d.]+\s+[-\d.]+", line)
            if m:
                cur.append(tuple(float(m.group(k)) for k in range(1, 6)))
            elif cur and line.strip() == "":
                inside = False
    return blocks


NF_ROW = re.compile(r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)" + r"\s+([-\d.]+E[-+]\d+)\s+([-\d.]+)" * 3 + r"\s*$")


def near_fields(text):
    """[(freq_mhz, nfeh, [(x, y, z, Fx, Fy, Fz)])] -- every NEAR ELECTRIC /
    MAGNETIC FIELDS block in print order (nfeh 0 = E, 1 = H)."""
    blocks, cur, freq = [], None, None
    for line in text.splitlines():
        m = FREQ.search(line) or re.search(r"FREQUENCY\s*:\s*([0-9.E+-]+)\s*MHz", line)
        if m:
            freq = float(m.group(1))
        if "NEAR ELECTRIC FIELDS" in line or "NEAR MAGNETIC FIELDS" in line:
            cur = []
            blocks.append((freq, 1 if "MAGNETIC" in line else 0, cur))
        elif cur is not None:
            m = NF_ROW.match(line)
            if m:
                v = [float(x) for x in m.groups()]
                cur.append((v[0], v[1], v[2], *[cmath.rect(v[3 + 2 * k], math.radians(v[4 + 2 * k]))
                                                 for k in range(3)]))
    return blocks
