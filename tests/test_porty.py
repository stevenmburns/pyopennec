"""Gates for pyopennec.porty against recorded card-deck runs on the same source.

The oracle is tests/fixtures/porty_oracle.json, recorded during private
development from pyopennec's former card-deck module (removed before 0.2.0;
itself parity-tested against nec2c and onec) on the same OpenNEC sources:
one deck run per frequency and per excitation.  porty solves every
frequency and every port in one call, so each gate compares the structural
solve's outputs with what the deck path reported for the equivalent EX set.
Floats are stored with repr(), so the tolerances below are those the live
comparison used.  The print-precision oracles that need no recording are in
test_porty_vs_cli.py.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pyopennec import _porty, porty

FREQS = [7.1, 14.2, 28.4]

GROUNDS = {
    "free": None,
    "perfect": "perfect",
    "gn0": ("reflection", 13.0, 0.005),
    "gn2": ("sommerfeld", 13.0, 0.005),
}

# A horizontal dipole 8 m up with an off-centre stub, so loads and ports have
# somewhere to sit off the symmetry plane.
DIPOLE = [(1, 21, (-5.2, 0.0, 8.0), (5.2, 0.0, 8.0), 0.001)]
# Three parallel wires, unequal lengths; port 3 off-centre.
ARRAY = [
    (1, 11, (-5.0, 0.0, 8.0), (5.0, 0.0, 8.0), 0.002),
    (2, 11, (-5.3, 3.0, 8.0), (5.3, 3.0, 8.0), 0.002),
    (3, 9, (-4.6, -2.5, 8.5), (4.6, -2.5, 7.5), 0.0015),
]
ARRAY_PORTS = [(1, 6), (2, 6), (3, 3)]
# Ground-mounted monopole: the GE flag joins its base to the image.
MONOPOLE = [(1, 15, (0.0, 0.0, 0.0), (0.0, 0.0, 5.0), 0.004)]

LOADS = {
    "none": (),
    "mixed": (
        (0, 1, 3, 3, 10.0, 1.0e-6, 0.0),  # series RL
        (1, 1, 5, 5, 1000.0, 2.0e-6, 100e-12),  # parallel RLC
        (4, 1, 15, 17, 50.0, -30.0, 0.0),  # fixed Z on a range
        (5, 1, 0, 0, 5.8e7, 0.0, 0.0),  # copper, whole tag
    ),
}
Y_LOADS = ((0, 2, 2, 2, 5.0, 0.5e-6, 0.0),)
V3 = np.array([1.0 + 0.5j, -0.3 + 0.8j, 0.7 - 0.2j])
RP = (19, 12, 0, 0, 0, 0, 0.0, 0.0, 10.0, 30.0, 0.0, 0.0)  # nth, nph, XNDA, th0, ph0, dth, dph, ...
NEAR_GRID = (3, 2, 2, -2.0, -1.0, 2.0, 2.0, 2.5, 3.0)  # NE/NH: nx, ny, nz, x0, y0, z0, dx, dy, dz

ORACLE = json.loads((Path(__file__).resolve().parent / "fixtures" / "porty_oracle.json").read_text())


def cx(d):
    return np.asarray(d["re"], dtype=np.float64) + 1j * np.asarray(d["im"], dtype=np.float64)


def _lists(x):
    return json.loads(json.dumps(x))


def test_oracle_records_these_cases():
    """The fixture was recorded for exactly the inputs the gates below use."""
    rec = ORACLE["inputs"]
    mine = dict(freqs=FREQS, grounds=GROUNDS, dipole=DIPOLE, array=ARRAY, array_ports=ARRAY_PORTS,
                monopole=MONOPOLE, loads=LOADS, y_loads=Y_LOADS, rp=RP, near_grid=NEAR_GRID)
    for k, v in mine.items():
        assert rec[k] == _lists(v), k
    np.testing.assert_array_equal(cx(rec["v3"]), V3)


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


# --- gate 1: one port ------------------------------------------------------


@pytest.mark.parametrize("ek", [False, True], ids=["thin", "ek"])
@pytest.mark.parametrize("loads", list(LOADS), ids=list(LOADS))
@pytest.mark.parametrize("gname", list(GROUNDS))
def test_one_port_impedance(gname, loads, ek):
    ground = GROUNDS[gname]
    st = porty.Structure(DIPOLE, ports=[(1, 11)], ground=ground, loads=LOADS[loads], extended_kernel=ek)
    sol = st.solve(FREQS)
    z_ref = cx(ORACLE["one_port_z"][f"{gname}-{loads}-{'ek' if ek else 'thin'}"])
    for k, f in enumerate(FREQS):
        z = 1.0 / sol.y[k, 0, 0]
        assert abs(z - z_ref[k]) / abs(z_ref[k]) < 1e-9, (gname, loads, ek, f, z, z_ref[k])


@pytest.mark.parametrize("gname", ["perfect", "gn0", "gn2"])
def test_one_port_ground_mounted(gname):
    ground = GROUNDS[gname]
    sol = porty.Structure(MONOPOLE, ports=[(1, 1)], ground=ground).solve(FREQS)
    z_ref = cx(ORACLE["ground_mounted_z"][gname])
    for k in range(len(FREQS)):
        assert abs(1.0 / sol.y[k, 0, 0] - z_ref[k]) / abs(z_ref[k]) < 1e-9


# --- gate 2: two and three ports ------------------------------------------


@pytest.mark.parametrize("nports", [2, 3])
@pytest.mark.parametrize("gname", ["free", "gn2"])
def test_y_matrix(nports, gname):
    ground = GROUNDS[gname]
    wires, ports = ARRAY[:nports], ARRAY_PORTS[:nports]
    sol = porty.Structure(wires, ports=ports, ground=ground, loads=Y_LOADS).solve(FREQS[:2])
    # ref[k, j, i]: the deck run's current on port j's segment with 1 V on port i
    ref = cx(ORACLE["y_matrix"][f"{nports}-{gname}"]).reshape(2, nports, nports)
    for k in range(2):
        for i in range(nports):
            for j in range(nports):
                assert abs(sol.y[k, j, i] - ref[k, j, i]) / abs(ref[k, j, i]) < 1e-9, (k, i, j)  # I_j / V_i


# --- gate 3: superposition -------------------------------------------------


@pytest.mark.parametrize("gname", ["free", "gn0"])
def test_superposition(gname):
    ground = GROUNDS[gname]
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=ground).solve(FREQS)
    ref = cx(ORACLE["superposition_currents"][gname]).reshape(len(FREQS), sol.nseg)
    for k in range(len(FREQS)):
        assert rel(sol.currents(k, V3), ref[k]) < 1e-9


def test_y_reads_as_i_equals_y_v():
    # NEC-2's Y is not symmetric (collocation), so this pins the orientation:
    # the port currents of a superposed excitation are Y @ V, not Y.T @ V.
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS).solve(FREQS[:1])
    y = sol.y[0]
    assert np.max(np.abs(y - y.T)) > 1e-6 * np.max(np.abs(y))  # the probe can tell
    i_ports = sol.currents(0, V3)[sol.port_segments - 1]
    assert rel(y @ V3, i_ports) < 1e-12
    assert rel(y.T @ V3, i_ports) > 1e-6
    # and against the recorded deck run, not only porty's own currents
    ref = cx(ORACLE["superposition_currents"]["free"]).reshape(len(FREQS), sol.nseg)[0]
    assert rel(y @ V3, ref[sol.port_segments - 1]) < 1e-9


# --- gate 4: far field -----------------------------------------------------


@pytest.mark.parametrize("gname", ["free", "perfect", "gn0", "gn2"])
def test_far_field(gname):
    ground = GROUNDS[gname]
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=ground).solve(FREQS[1:])
    nth, nph = RP[0], RP[1]
    for k, p in enumerate(ORACLE["far_field"][gname]):
        # NEC's flat order: phi outer, theta inner
        th = np.tile(p["theta"], nph)
        ph = np.repeat(p["phi"], nth)
        ff = porty.far_field(sol, k, V3, th, ph)
        np.testing.assert_array_equal(ff.valid, [c == "1" for c in p["valid"]])
        ref = np.asarray(p["gain_tot"])
        live = ff.valid & (ref > -60.0)
        assert live.sum() > nth * nph // 3
        assert np.max(np.abs(ff.gain_total[live] - ref[live])) < 1e-6
        for mine, theirs in ((ff.gain_vert, p["gain_vert"]), (ff.gain_horiz, p["gain_horiz"])):
            theirs = np.asarray(theirs)
            ok = live & (theirs > -60.0)
            assert np.max(np.abs(mine[ok] - theirs[ok])) < 1e-6
        assert rel(ff.e_theta[live], cx(p["e_theta"])[live]) < 1e-8


@pytest.mark.parametrize("kind", ["E", "H"])
@pytest.mark.parametrize("gname", ["free", "gn2"])
def test_near_field(gname, kind):
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=GROUNDS[gname]).solve([14.2])
    rec = ORACLE["near_field"][f"{gname}-{kind}"]
    pts = np.asarray(rec["points"]).reshape(-1, 3)
    assert len(pts) == NEAR_GRID[0] * NEAR_GRID[1] * NEAR_GRID[2]
    ref = cx(rec["field"]).reshape(-1, 3)
    got = porty.near_field(sol, 0, V3, pts, kind=kind)
    assert rel(got, ref) < 1e-9


def test_near_field_after_frequency_switch():
    """somnec() refills the Sommerfeld tables but OpenNEC's interpolator keeps
    the last cell's coefficients; without porty's reset the first points after
    a frequency switch read the old grid (0.3 % here).  Must equal a fresh
    single-frequency solution exactly."""
    wire = [(1, 21, (-5.2, 0.0, 0.6), (5.2, 0.0, 0.6), 0.001)]
    st = porty.Structure(wire, ports=[(1, 11)], ground=GROUNDS["gn2"])
    pts = np.array([[0.3, 0.2, 0.4], [0.31, 0.2, 0.4]])
    both = st.solve([14.0, 14.5])
    porty.near_field(both, 0, [1.0], pts)
    after = porty.near_field(both, 1, [1.0], pts)
    fresh = porty.near_field(st.solve([14.5]), 0, [1.0], pts)
    np.testing.assert_array_equal(after, fresh)


# --- gate 5: reciprocity ---------------------------------------------------
#
# NEC-2 point-matches at segment centres (collocation, not Galerkin), so its
# Y is reciprocal only to discretisation error: ~0.2 % on the off-diagonal
# terms of ARRAY as meshed, and the oracle shows the same asymmetry (gate 2
# compares Y[i, j] and Y[j, i] separately).  What can be pinned: exact
# symmetry where the geometry forces it, and the asymmetry shrinking roughly
# as the square of the segment length.


@pytest.mark.parametrize("gname", list(GROUNDS))
def test_reciprocity_mirror_pair(gname):
    pair = [(1, 21, (-5, 0, 8), (5, 0, 8), 0.002), (2, 21, (-5, 2, 8), (5, 2, 8), 0.002)]
    sol = porty.Structure(pair, ports=[(1, 11), (2, 11)], ground=GROUNDS[gname],
                          loads=((4, 0, 0, 0, 20.0, 5.0, 0.0),)).solve(FREQS)
    for k in range(sol.nf):
        y = sol.y[k]
        assert np.max(np.abs(y - y.T)) / np.max(np.abs(y)) < 1e-8


@pytest.mark.parametrize("gname", ["free", "gn0"])
def test_reciprocity_converges(gname):
    asym = []
    for m in (1, 3, 9):
        wires = [(t, n * m, a, b, r) for t, n, a, b, r in ARRAY]
        ports = [(1, 5 * m + 1), (2, 5 * m + 1), (3, 2 * m + 1 if m > 1 else 3)]
        y = porty.Structure(wires, ports=ports, ground=GROUNDS[gname]).solve([14.2]).y[0]
        off = ~np.eye(3, dtype=bool)
        asym.append(np.max(np.abs(y - y.T)[off] / np.abs(y)[off]))
    assert asym[0] < 1e-2
    assert asym[1] < asym[0] / 4 and asym[2] < asym[1] / 4, asym


# --- gate 7: cancel, progress, GIL, validation ----------------------------


def test_progress_and_cancel_callable():
    st = porty.Structure(ARRAY, ports=ARRAY_PORTS)
    seen = []
    with pytest.raises(porty.Cancelled):
        st.solve([10.0, 12.0, 14.0, 16.0], progress=lambda i, n: seen.append((i, n)), cancel=lambda: len(seen) >= 2)
    assert seen == [(1, 4), (2, 4)]
    ok = st.solve([10.0, 12.0], progress=lambda i, n: seen.append((i, n)), cancel=lambda: False)
    assert ok.nf == 2 and seen[-2:] == [(1, 2), (2, 2)]


def test_cancel_event():
    ev = threading.Event()
    ev.set()
    with pytest.raises(porty.Cancelled, match="before frequency 1 of 1"):
        porty.Structure(DIPOLE, ports=[(1, 11)]).solve([14.0], cancel=ev)


def test_progress_exception_propagates():
    def boom(i, n):
        raise KeyError("from progress")

    with pytest.raises(KeyError, match="from progress"):
        porty.Structure(DIPOLE, ports=[(1, 11)]).solve([14.0, 15.0], progress=boom)


def test_gil_released_during_solve():
    """A Python thread keeps running while another thread is inside solve()."""
    wires = [(1, 600, (-5.2, 0, 8), (5.2, 0, 8), 0.001)]
    st = porty.Structure(wires, ports=[(1, 300)], ground=GROUNDS["gn0"])
    ticks, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            ticks[0] += 1

    t = threading.Thread(target=spin)
    t.start()
    try:
        before = ticks[0]
        t0 = time.perf_counter()
        sol = st.solve([14.0, 14.1])
        dt = time.perf_counter() - t0
        during = ticks[0] - before
    finally:
        stop.set()
        t.join()
    assert sol.nf == 2
    assert during > 1000, (during, dt)


def test_threads_agree():
    st = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=GROUNDS["gn2"])
    ref = st.solve(FREQS).y
    out = [None] * 4

    def run(i):
        out[i] = st.solve(FREQS).y

    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for y in out:
        np.testing.assert_array_equal(y, ref)


def test_field_queries_share_a_solution_across_threads():
    """Field queries rescale the Solution's one engine context; its lock
    serialises them, so interleaved frequencies still read their own."""
    sol = porty.Structure(ARRAY, ports=ARRAY_PORTS, ground=GROUNDS["gn2"]).solve(FREQS)
    th, ph = np.linspace(0, 90, 7), np.full(7, 30.0)
    pts = np.array([[0.5, 0.5, 9.0]])
    ref = [(porty.far_field(sol, k, V3, th, ph).gain_total, porty.near_field(sol, k, V3, pts)) for k in range(3)]
    errors = []

    def run(k):
        for _ in range(20):
            g = porty.far_field(sol, k, V3, th, ph).gain_total
            e = porty.near_field(sol, k, V3, pts)
            if not (np.array_equal(g, ref[k][0]) and np.array_equal(e, ref[k][1])):
                errors.append(k)

    ts = [threading.Thread(target=run, args=(k % 3,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []


W = [(1, 9, (0, 0, -2.5), (0, 0, 2.5), 0.005), (2, 5, (1, 0, -1), (1, 0, 1), 0.005)]
BAD_STRUCTURES = {
    "no wires": (dict(wires=[], ports=[(1, 5)]), "wires: at least one"),
    "nseg zero": (dict(wires=[W[0], (2, 0, (1, 0, -1), (1, 0, 1), 0.005)], ports=[(1, 5)]), r"wires\[1\].nseg"),
    "radius": (dict(wires=[(1, 9, (0, 0, -2.5), (0, 0, 2.5), 0.0)], ports=[(1, 5)]), r"wires\[0\].radius"),
    "end coord": (dict(wires=[(1, 9, (0, "a", -2.5), (0, 0, 2.5), 0.005)], ports=[(1, 5)]), r"wires\[0\].end1.y"),
    "nan coord": (dict(wires=[(1, 9, (0, 0, float("nan")), (0, 0, 2.5), 0.005)], ports=[(1, 5)]), r"wires\[0\].end1.z"),
    "zero length": (dict(wires=[(1, 9, (0, 0, 1), (0, 0, 1), 0.005)], ports=[(1, 5)]), r"wires\[0\].end2"),
    "short record": (dict(wires=[(1, 9, (0, 0, 1), (0, 0, 2))], ports=[(1, 5)]), r"wires\[0\]: expected a sequence of 5"),
    "float tag": (dict(wires=[(1.5, 9, (0, 0, 1), (0, 0, 2), 0.01)], ports=[(1, 5)]), r"wires\[0\].tag"),
    "port past wire": (dict(wires=W, ports=[(1, 5), (2, 6)]), r"ports\[1\].seg: no segment 6 on tag 2"),
    "port missing tag": (dict(wires=W, ports=[(7, 1)]), r"ports\[0\].tag: no wire has tag 7"),
    "port seg 0": (dict(wires=W, ports=[(1, 0)]), r"ports\[0\].seg: must be >= 1"),
    "port duplicate": (dict(wires=W, ports=[(1, 5), (0, 5)]), r"ports\[1\]: same segment as ports\[0\]"),
    "no ports": (dict(wires=W, ports=[]), "ports: at least one"),
    "load type": (dict(wires=W, ports=[(1, 5)], loads=[(9, 1, 1, 1, 50, 0, 0)]), r"loads\[0\].type"),
    "load range": (dict(wires=W, ports=[(1, 5)], loads=[(0, 1, 1, 1, 1, 0, 0), (0, 2, 3, 9, 1, 0, 0)]), r"loads\[1\].seg_to"),
    "load from": (dict(wires=W, ports=[(1, 5)], loads=[(0, 2, 6, 0, 1, 0, 0)]), r"loads\[0\].seg_from"),
    "load tag": (dict(wires=W, ports=[(1, 5)], loads=[(0, 4, 0, 0, 1, 0, 0)]), r"loads\[0\].tag"),
    "load sigma": (dict(wires=W, ports=[(1, 5)], loads=[(5, 1, 0, 0, 0.0, 0, 0)]), r"loads\[0\].f1"),
    "load field": (dict(wires=W, ports=[(1, 5)], loads=[(0, 1, 0, 0, 1, None, 0)]), r"loads\[0\].f2"),
    "ground eps": (dict(wires=W, ports=[(1, 5)], ground=("reflection", 0.5, 0.01)), "ground.epsr"),
    "ground sigma": (dict(wires=W, ports=[(1, 5)], ground=("sommerfeld", 13, -1.0)), "ground.sigma"),
    "ground kind": (dict(wires=W, ports=[(1, 5)], ground="lossy"), "ground: unknown kind"),
    "below ground": (dict(wires=[(1, 9, (0, 0, -2.5), (0, 0, 2.5), 0.005)], ports=[(1, 5)], ground="perfect"), "geometry:"),
}


@pytest.mark.parametrize("case", list(BAD_STRUCTURES))
def test_validation_structure(case):
    kw, msg = BAD_STRUCTURES[case]
    with pytest.raises(ValueError, match=msg):
        porty.Structure(kw.pop("wires"), **kw)


def test_validation_solve_and_fields():
    st = porty.Structure(W, ports=[(1, 5), (2, 3)])
    with pytest.raises(ValueError, match=r"freqs_mhz\[1\]"):
        st.solve([14.0, -1.0])
    with pytest.raises(ValueError, match="freqs_mhz: at least one"):
        st.solve([])
    with pytest.raises(ValueError, match="cancel"):
        st.solve([14.0], cancel=3)
    sol = st.solve([14.0])
    with pytest.raises(ValueError, match="port_voltages: expected 2"):
        porty.far_field(sol, 0, [1.0], [90.0], [0.0])
    with pytest.raises(ValueError, match="i_f: 1 is outside 0..0"):
        porty.far_field(sol, 1, [1.0, 0.0], [90.0], [0.0])
    with pytest.raises(ValueError, match=r"port_voltages\[1\]: must be finite"):
        porty.far_field(sol, 0, [1.0, np.nan], [90.0], [0.0])
    with pytest.raises(ValueError, match="deliver no input power"):
        porty.far_field(sol, 0, [0.0, 0.0], [90.0], [0.0])
    with pytest.raises(ValueError, match="points"):
        porty.near_field(sol, 0, [1.0, 0.0], [1.0, 2.0])
    with pytest.raises(TypeError):
        porty.Solution.__new__(_porty.Solution)


def test_segments_and_wavelength_scaling():
    sol = porty.Structure(W, ports=[(1, 5)]).solve([14.0])
    s = sol.segments
    assert s.centres.shape == (14, 3)
    np.testing.assert_allclose(s.lengths[:9], 5.0 / 9, rtol=1e-12)
    np.testing.assert_array_equal(s.tags, [1] * 9 + [2] * 5)
    np.testing.assert_array_equal(s.numbers, list(range(1, 10)) + list(range(1, 6)))
    assert sol.port_segments.tolist() == [5]
    # tag numbers agree with the engine's tag_nums, a tag split over two wires counts on
    split = [W[0], W[1], (1, 4, (2, 0, -1), (2, 0, 1), 0.005)]
    s2 = porty.Structure(split, ports=[(1, 11)]).solve([14.0])
    np.testing.assert_array_equal(s2.segments.tags, [1] * 9 + [2] * 5 + [1] * 4)
    np.testing.assert_array_equal(s2.segments.numbers, list(range(1, 10)) + list(range(1, 6)) + [10, 11, 12, 13])
    assert s2.port_segments.tolist() == [16]
    # 1 V on a half-wave-ish dipole: tens of mA, not mA/lambda
    assert 1e-3 < abs(sol.basis_currents[0, 0, 4]) < 1.0
