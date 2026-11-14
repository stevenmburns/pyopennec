"""porty against oracles that need no recording: the golden pins, onec, nec2c.

* Golden pins (nec2c 1.3.1, 1e-4): every deck under
  opennec/tests/golden_tests that porty can state -- GW (plus GX reflection
  written out), GE, GN, LD, EK, EX 0 at 1 V, FR -- and the TL / NT decks
  with the network applied to porty's port Y by the caller, the way a
  circuit layer uses it.
* onec, built from the same submodule commit by tools/build_onec.sh: decks
  written from porty's own cases, compared at print precision (half a unit
  in the last printed digit of %12.5E impedances, %11.4E currents, %7.2f
  gains).  onec is the engine's own CLI and shares porty's source, so this
  proves the port-solve wiring, not the engine.
* nec2c (when on PATH), a separate C translation of NEC-2: impedances at
  1e-4 and near fields at 2e-3 of the peak, the tolerances the card-deck
  module was held to.
"""

import json
import math
import shutil
import subprocess

import numpy as np
import pytest

import nec_output
from pyopennec import porty

CVEL = 299.8  # NEC-2's speed of light, m/us
NEC2C = shutil.which("nec2c")

# --- golden decks --------------------------------------------------------


def parse_deck(text):
    """A deck porty can state, as Structure kwargs + freqs + the network
    the caller applies on Y; None for anything else."""
    wires, loads, ports, net = [], [], [], None
    ground, gpflag, ek, freqs = None, 0, False, None
    for line in text.splitlines():
        f = line.split()
        if not f or f[0] in ("CM", "CE", "XQ", "EN"):
            continue
        card, v = f[0], [float(x) for x in f[1:]]
        if card == "GW":
            tag, nseg = int(v[0]), int(v[1])
            wires.append((tag, nseg, tuple(v[2:5]), tuple(v[5:8]), v[8]))
        elif card == "GX":
            inc, axes = int(v[0]), f"{int(v[1]):03d}"
            if axes != "100":
                return None
            wires += [(t + inc, n, (-a[0], a[1], a[2]), (-b[0], b[1], b[2]), r) for t, n, a, b, r in wires]
        elif card == "GE":
            gpflag = int(v[0])
        elif card == "GN":
            kind = int(v[0])
            ground = "perfect" if kind == 1 else (("reflection", "", "sommerfeld")[kind], v[4], v[5])
        elif card == "LD":
            loads.append((int(v[0]), int(v[1]), int(v[2]), int(v[3]), *(v[4:7] + [0.0] * (7 - len(v)))[:3]))
        elif card == "EK":
            ek = True
        elif card == "EX":
            if int(v[0]) != 0 or v[4:6] != [1.0, 0.0][: len(v[4:6])]:
                return None
            p = (int(v[1]), int(v[2]))
            if p in ports:
                ports.remove(p)
            ports.insert(0, p)
        elif card == "FR":
            if int(v[0]) != 0:
                return None
            freqs = [v[4] + k * (v[5] if len(v) > 5 else 0.0) for k in range(int(v[1]))]
        elif card in ("TL", "NT"):
            a, b = (int(v[0]), int(v[1])), (int(v[2]), int(v[3]))
            ports += [p for p in (a, b) if p not in ports]
            net = (card, a, b, v[4:])
        else:
            return None
    # porty sets the GE ground-plane flag exactly when there is a ground;
    # the decks' GE value only matters for a wire end on z = 0
    on_plane = any(a[2] == 0 or b[2] == 0 for _, _, a, b, _ in wires)
    if on_plane and gpflag != (ground is not None):
        return None
    return dict(wires=wires, ports=ports, ground=ground, loads=tuple(loads), ek=ek, freqs=freqs, net=net)


def network_y(net, ports, f_mhz):
    """The NT / TL two-port as an admittance matrix over porty's ports."""
    card, a, b, p = net
    if card == "NT":
        y11, y12, y22 = complex(p[0], p[1]), complex(p[2], p[3]), complex(p[4], p[5])
    else:  # TL: lossless line; a negative Z0 is NEC's crossed line
        z0, length = p[0], p[1]
        bl = 2 * math.pi * f_mhz / CVEL * length
        y11 = y22 = -1j / (math.tan(bl) * abs(z0))
        y12 = math.copysign(1.0, z0) * 1j / (abs(z0) * math.sin(bl))
    yn = np.zeros((len(ports), len(ports)), dtype=complex)
    i, j = ports.index(a), ports.index(b)
    yn[i, i], yn[j, j], yn[i, j], yn[j, i] = y11, y22, y12, y12
    return yn


def input_impedance(y_ant, y_net):
    """1 V on port 0, every other port loaded only by the network: the other
    ports' voltages make the total port current vanish there.  NEC reports
    the source current as the segment current plus the network's."""
    y = y_ant + y_net
    v = np.zeros(len(y), dtype=complex)
    v[0] = 1.0
    if len(y) > 1:
        v[1:] = np.linalg.solve(y[1:, 1:], -y[1:, 0])
    return 1.0 / (y @ v)[0]


def test_golden_pins(golden, record_property):
    spec = json.loads((golden / "expected.json").read_text())
    worst, n, skipped = 0.0, 0, []
    for entry in spec["decks"]:
        case = parse_deck((golden / "decks" / entry["deck"]).read_text())
        if case is None:
            skipped.append(entry["deck"])
            continue
        st = porty.Structure(case["wires"], ports=case["ports"], ground=case["ground"],
                             loads=case["loads"], extended_kernel=case["ek"])
        sol = st.solve(case["freqs"])
        assert len(entry["points"]) == sol.nf, entry["deck"]
        for k, pt in enumerate(entry["points"]):
            assert abs(sol.freqs_mhz[k] - pt["freq_mhz"]) < 1e-9
            assert case["ports"][0] == (pt["tag"], pt["seg"])
            yn = network_y(case["net"], case["ports"], pt["freq_mhz"]) if case["net"] else 0.0
            z = input_impedance(sol.y[k], yn)
            zw = complex(pt["r"], pt["x"])
            r = abs(z - zw) / abs(zw)
            worst = max(worst, r)
            n += 1
            assert r <= spec["tolerance"], (entry["deck"], z, zw, r)
    record_property("worst_rel_dZ", worst)
    print(f"\nporty vs golden pins: {n} points over {len(spec['decks']) - len(skipped)} decks, "
          f"worst |dZ|/|Z| = {worst:.2e} (tolerance {spec['tolerance']:.0e}); skipped {skipped}")
    # every deck in the matrix is one porty can state today; a new one that
    # is not must be looked at, not silently dropped
    assert skipped == []


# --- onec at print precision ----------------------------------------------

FREQS = [7.1, 14.2, 28.4]  # FR 1 3 0 0 7.1 2: multiplicative steps, exact doublings
GROUNDS = {
    "free": None,
    "perfect": "perfect",
    "gn0": ("reflection", 13.0, 0.005),
    "gn2": ("sommerfeld", 13.0, 0.005),
}
DIPOLE = [(1, 21, (-5.2, 0.0, 8.0), (5.2, 0.0, 8.0), 0.001)]
ARRAY = [
    (1, 11, (-5.0, 0.0, 8.0), (5.0, 0.0, 8.0), 0.002),
    (2, 11, (-5.3, 3.0, 8.0), (5.3, 3.0, 8.0), 0.002),
    (3, 9, (-4.6, -2.5, 8.5), (4.6, -2.5, 7.5), 0.0015),
]
ARRAY_PORTS = [(1, 6), (2, 6), (3, 3)]
MONOPOLE = [(1, 15, (0.0, 0.0, 0.0), (0.0, 0.0, 5.0), 0.004)]
MIXED = (
    (0, 1, 3, 3, 10.0, 1.0e-6, 0.0),
    (1, 1, 5, 5, 1000.0, 2.0e-6, 100e-12),
    (4, 1, 15, 17, 50.0, -30.0, 0.0),
    (5, 1, 0, 0, 5.8e7, 0.0, 0.0),
)
V3 = np.array([1.0 + 0.5j, -0.3 + 0.8j, 0.7 - 0.2j])
RP = "RP 0 19 12 1000 0 0 10 30"  # theta 0..180 step 10, phi 0..330 step 30


def deck(wires, ground=None, loads=(), ek=False, sources=(), rp=False, near=None):
    lines = ["CE"]
    lines += [f"GW {t} {n} {a[0]!r} {a[1]!r} {a[2]!r} {b[0]!r} {b[1]!r} {b[2]!r} {r!r}" for t, n, a, b, r in wires]
    lines.append(f"GE {0 if ground is None else 1}")
    lines += ["LD " + " ".join(repr(x) for x in ld) for ld in loads]
    if ground == "perfect":
        lines.append("GN 1")
    elif ground is not None:
        kind, eps, sig = ground
        lines.append(f"GN {0 if kind == 'reflection' else 2} 0 0 0 {eps!r} {sig!r}")
    if ek:
        lines.append("EK 0")
    lines += [f"EX 0 {t} {s} 0 {complex(v).real!r} {complex(v).imag!r}" for t, s, v in sources]
    lines.append(f"FR 1 {len(FREQS)} 0 0 {FREQS[0]!r} 2")
    lines.append(RP if rp else "XQ")
    if near:
        lines.append(near)
    lines.append("EN")
    return "\n".join(lines) + "\n"


def half_quantum(x, digits, slack=1e-12):
    """Half a unit in the last place of a %.{digits}E print, plus a relative
    slack for porty's own ~1e-12 distance from the engine's deck path."""
    if x == 0:
        return 0.5 * 10.0 ** -digits
    return 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - digits) + slack * abs(x)


@pytest.fixture(scope="module")
def run_onec(onec, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("onec")
    count = [0]

    def run(text):
        count[0] += 1
        d = tmp / f"d{count[0]}.nec"
        d.write_text(text)
        out = tmp / f"d{count[0]}.out"
        r = subprocess.run([onec, "-i", str(d), "-o", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return out.read_text(errors="replace")

    return run


ONE_PORT = {
    f"dipole-{g}-{lname}-{'ek' if ek else 'thin'}": (DIPOLE, [(1, 11)], g, loads, ek)
    for g in GROUNDS for lname, loads in (("none", ()), ("mixed", MIXED)) for ek in (False, True)
} | {f"monopole-{g}": (MONOPOLE, [(1, 1)], g, (), False) for g in ("perfect", "gn0", "gn2")}


def test_impedance_vs_onec(run_onec, record_property):
    worst = 0.0
    for name, (wires, ports, g, loads, ek) in ONE_PORT.items():
        sol = porty.Structure(wires, ports=ports, ground=GROUNDS[g], loads=loads, extended_kernel=ek).solve(FREQS)
        rows = nec_output.input_parameters(run_onec(deck(wires, GROUNDS[g], loads, ek, [(*ports[0], 1.0)])))
        assert len(rows) == len(FREQS), name
        for k, row in enumerate(rows):
            assert math.isclose(row[0], FREQS[k], rel_tol=1e-6)
            z, zp = 1.0 / sol.y[k, 0, 0], row[5]
            assert abs(z.real - zp.real) <= half_quantum(zp.real, 5), (name, k, z, zp)
            assert abs(z.imag - zp.imag) <= half_quantum(zp.imag, 5), (name, k, z, zp)
            worst = max(worst, abs(z - zp) / abs(zp))
    record_property("worst_rel_dZ", worst)
    print(f"\nporty vs onec: {len(ONE_PORT) * len(FREQS)} impedances within %12.5E half-quantum, "
          f"worst |dZ|/|Z| = {worst:.2e}")


def _currents_vs_onec(run_onec, sol, wires, ground, sources, v):
    blocks = nec_output.currents(run_onec(deck(wires, ground, (), False, sources)))
    assert len(blocks) == len(FREQS)
    worst = 0.0
    for k, block in enumerate(blocks):
        got = sol.currents(k, v)
        assert len(block) == len(got)
        for (seg, tag, ip), i in zip(block, got):
            assert tag == sol.segments.tags[seg - 1]
            for a, b in ((i.real, ip.real), (i.imag, ip.imag)):
                assert abs(a - b) <= half_quantum(b, 4), (seg, i, ip)
        ref = np.array([b[2] for b in block])
        worst = max(worst, np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    return worst


@pytest.mark.parametrize("gname", list(GROUNDS))
def test_y_columns_vs_onec(run_onec, gname):
    """Each port driven alone in onec: all segment currents equal porty's
    basis currents for that port, so every Y[j, i] is checked, in order."""
    ground = GROUNDS[gname]
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=ground).solve(FREQS)
    for i, (t, s) in enumerate(ARRAY_PORTS):
        e = np.zeros(3, dtype=complex)
        e[i] = 1.0
        _currents_vs_onec(run_onec, sol, ARRAY, ground, [(t, s, 1.0)], e)


@pytest.mark.parametrize("gname", list(GROUNDS))
def test_superposition_and_far_field_vs_onec(run_onec, gname, record_property):
    ground = GROUNDS[gname]
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=ground).solve(FREQS)
    src = [(t, s, v) for (t, s), v in zip(ARRAY_PORTS, V3)]
    worst_i = _currents_vs_onec(run_onec, sol, ARRAY, ground, src, V3)
    blocks = nec_output.patterns(run_onec(deck(ARRAY, ground, (), False, src, rp=True)))
    assert len(blocks) == len(FREQS)
    worst_db = 0.0
    for k, block in enumerate(blocks):
        rows = np.array(block)
        ff = porty.far_field(sol, k, V3, rows[:, 0], rows[:, 1])
        live = ff.valid & (rows[:, 4] > -99.0)
        assert live.sum() > len(rows) // 3
        # gains print as %7.2f: half a unit in the second decimal
        np.testing.assert_allclose(ff.gain_total[live], rows[live, 4], atol=0.005 + 1e-9)
        worst_db = max(worst_db, np.max(np.abs(ff.gain_total[live] - rows[live, 4])))
    record_property("worst_rel_dI", worst_i)
    print(f"\nporty vs onec ({gname}): currents worst {worst_i:.1e} of peak, gains worst {worst_db:.4f} dB")


# --- nec2c -----------------------------------------------------------------


def run_nec2c(text, tmp_path):
    # nec2c aborts on long -o paths and writes <name>.out next to the deck
    (tmp_path / "d.nec").write_text(text)
    subprocess.run([NEC2C, "-i", "d.nec"], cwd=tmp_path, check=True, capture_output=True)
    return (tmp_path / "d.out").read_text(errors="replace")


@pytest.mark.skipif(NEC2C is None, reason="nec2c not installed")
@pytest.mark.parametrize("name", ["dipole-free-mixed-thin", "dipole-gn2-none-ek", "monopole-gn2", "dipole-gn0-mixed-thin"])
def test_impedance_vs_nec2c(name, tmp_path):
    wires, ports, g, loads, ek = ONE_PORT[name]
    sol = porty.Structure(wires, ports=ports, ground=GROUNDS[g], loads=loads, extended_kernel=ek).solve(FREQS)
    rows = nec_output.input_parameters(run_nec2c(deck(wires, GROUNDS[g], loads, ek, [(*ports[0], 1.0)]), tmp_path))
    assert len(rows) == len(FREQS)
    for k, row in enumerate(rows):
        r = abs(1.0 / sol.y[k, 0, 0] - row[5]) / abs(row[5])
        print(f" {name} {FREQS[k]} MHz: |dZ|/|Z| = {r:.1e}", end="")
        assert r <= 1e-4


@pytest.mark.skipif(NEC2C is None, reason="nec2c not installed")
@pytest.mark.parametrize("gname", ["free", "gn2"])
def test_near_field_vs_nec2c(gname, tmp_path):
    """nec2c evaluates an NE/NH after XQ at the last frequency only."""
    ground = GROUNDS[gname]
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=ground).solve(FREQS)
    src = [(t, s, v) for (t, s), v in zip(ARRAY_PORTS, V3)]
    grid = "0 3 2 2 -2.0 -1.0 2.0 2.0 2.5 3.0"
    text = deck(ARRAY, ground, (), False, src, near=f"NE {grid}\nNH {grid}")
    blocks = nec_output.near_fields(run_nec2c(text, tmp_path))
    assert len(blocks) == 2
    for freq, nfeh, block in blocks:
        assert math.isclose(freq, FREQS[-1], rel_tol=1e-6)
        ref = np.array(block)
        got = porty.near_field(sol, len(FREQS) - 1, V3, ref[:, :3].real, kind="EH"[nfeh])
        err = np.max(np.abs(got - ref[:, 3:6])) / np.max(np.abs(ref[:, 3:6]))
        print(f" near {gname} {'EH'[nfeh]}: {err:.1e}", end="")
        assert err <= 2e-3
