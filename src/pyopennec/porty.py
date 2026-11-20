"""OpenNEC behind an N-port admittance interface.

The structure is solved once per frequency for the port admittance matrix
``Y`` (``Y[i, j]`` = current at port ``i`` with 1 V on port ``j`` and every
other port shorted, so ``I = Y @ V``) and the segment currents each port drives.  Circuits --
transmission lines, networks, phasing, feed voltages -- are the caller's, on
``Y``; fields for a set of port voltages are computed on demand from the
stored basis currents without re-solving::

    from pyopennec import porty
    st = porty.Structure([(1, 21, (0, 0, -5.2), (0, 0, 5.2), 0.001)], ports=[(1, 11)])
    sol = st.solve([14.0, 14.2])
    z = 1 / sol.y[:, 0, 0]
    ff = porty.far_field(sol, 0, [1.0], theta_deg=[90.0], phi_deg=[0.0])

The engine work lives in the C extension ``pyopennec._porty`` (plain C on the
CPython C API, no numpy C API); this module only turns its byte buffers into
numpy arrays and its low-level arguments into a friendlier signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from . import _porty
from ._porty import Cancelled, PortyError

__all__ = ["Structure", "Solution", "FarField", "far_field", "near_field", "Cancelled", "PortyError"]

GroundSpec = Union[None, str, Tuple[str, float, float]]

_GROUND_KINDS = {"free": -1, "reflection": 0, "perfect": 1, "sommerfeld": 2}


def _ground_args(ground: GroundSpec) -> dict:
    """None/"free", "perfect", ("reflection", epsr, sigma) (GN 0) or
    ("sommerfeld", epsr, sigma) (GN 2)."""
    if ground is None:
        return dict(ground=-1)
    if isinstance(ground, str):
        kind = _GROUND_KINDS.get(ground)
        if kind not in (-1, 1):
            raise ValueError(f"ground: {ground!r} needs (kind, epsr, sigma)" if kind is not None
                             else f"ground: unknown kind {ground!r}")
        return dict(ground=kind)
    try:
        name, epsr, sigma = ground
    except (TypeError, ValueError):
        raise ValueError("ground: expected None, 'perfect', or (kind, epsr, sigma)") from None
    kind = _GROUND_KINDS.get(name)
    if kind not in (0, 2):
        raise ValueError(f"ground.kind: {name!r} takes no constants; use 'reflection' or 'sommerfeld'")
    return dict(ground=kind, epsr=float(epsr), sigma=float(sigma))


class Structure:
    """Straight wires, structural loads, ground and kernel, with ports.

    wires: ``(tag, nseg, (x1, y1, z1), (x2, y2, z2), radius)``, metres.
    ports: ``(tag, seg)``, seg 1-based within the tag (tag 0: absolute).
    loads: ``(type, tag, seg_from, seg_to, f1, f2, f3)``, an LD card's
    fields, types 0-5.
    ground: None (free space), "perfect", ("reflection", epsr, sigma) for
    GN 0 or ("sommerfeld", epsr, sigma) for GN 2.  A ground also sets the
    GE ground-plane flag, so wire ends on z = 0 connect to their image.
    fill_reuse: None leaves it to ``OPENNEC_FILL_REUSE`` and then the build
    (on wherever OpenMP meets OpenBLAS or Accelerate); True or False sets it
    for this structure's solves alone, whatever the environment says.
    """

    def __init__(self, wires, *, ports, ground: GroundSpec = None, loads=(), extended_kernel: bool = False,
                 fill_reuse: Optional[bool] = None):
        if fill_reuse is not None and not isinstance(fill_reuse, bool):
            raise TypeError(f"fill_reuse: None, True or False, got {fill_reuse!r}")
        self._c = _porty.Structure(wires, ports, loads, extended_kernel=bool(extended_kernel),
                                   fill_reuse=0 if fill_reuse is None else (1 if fill_reuse else -1),
                                   **_ground_args(ground))

    @property
    def nseg(self) -> int:
        return self._c.nseg

    @property
    def nports(self) -> int:
        return self._c.nports

    def solve(self, freqs_mhz, cancel=None, progress: Optional[Callable[[int, int], None]] = None) -> "Solution":
        """Fill, factor and solve once per frequency (GIL released).

        ``cancel``: a ``threading.Event`` or a callable returning bool,
        checked before each frequency; a set cancel raises ``Cancelled``.
        ``progress(done, total)`` is called after each frequency."""
        f = np.ascontiguousarray(np.atleast_1d(np.asarray(freqs_mhz, dtype=np.float64)))
        if f.ndim != 1:
            raise ValueError("freqs_mhz: expected a 1-D sequence")
        return Solution(self._c.solve(f, cancel=cancel, progress=progress))


@dataclass(frozen=True)
class Segments:
    centres: np.ndarray  # (nseg, 3) m
    lengths: np.ndarray  # (nseg,) m
    radii: np.ndarray  # (nseg,) m
    tags: np.ndarray  # (nseg,) int32
    numbers: np.ndarray  # (nseg,) int32, 1-based within the tag


class Solution:
    """Per-frequency port admittances and per-port segment currents."""

    def __init__(self, c):
        self._c = c
        nf, p, n = c.nf, c.nports, c.nseg
        self.freqs_mhz = np.frombuffer(c.freqs_mhz, dtype=np.float64)
        self.y = np.frombuffer(c.y, dtype=np.complex128).reshape(nf, p, p)
        self.basis_currents = np.frombuffer(c.basis_currents, dtype=np.complex128).reshape(nf, p, n)
        self.port_segments = np.frombuffer(c.port_segments, dtype=np.int32)
        xyz, ln, rad, tag, num = c.segments
        self.segments = Segments(
            np.frombuffer(xyz, dtype=np.float64).reshape(n, 3),
            np.frombuffer(ln, dtype=np.float64),
            np.frombuffer(rad, dtype=np.float64),
            np.frombuffer(tag, dtype=np.int32),
            np.frombuffer(num, dtype=np.int32),
        )

    @property
    def nf(self) -> int:
        return self._c.nf

    @property
    def nports(self) -> int:
        return self._c.nports

    @property
    def nseg(self) -> int:
        return self._c.nseg

    def currents(self, i_f: int, port_voltages) -> np.ndarray:
        """Segment-centre currents (A) for port voltages, by superposition."""
        return np.asarray(port_voltages, dtype=np.complex128) @ self.basis_currents[i_f]


@dataclass(frozen=True)
class FarField:
    gain_total: np.ndarray  # dBi, power gain (RP XNDA 0)
    gain_vert: np.ndarray
    gain_horiz: np.ndarray
    e_theta: np.ndarray  # r*E, volts
    e_phi: np.ndarray
    valid: np.ndarray  # False below the horizon over ground (-999.99 dB there)
    power_in: float  # W delivered by the port voltages


def _voltages(sol: Solution, port_voltages) -> np.ndarray:
    v = np.ascontiguousarray(np.atleast_1d(np.asarray(port_voltages, dtype=np.complex128)))
    if v.shape != (sol.nports,):
        raise ValueError(f"port_voltages: expected {sol.nports} values, got shape {v.shape}")
    return v


def far_field(sol: Solution, i_f: int, port_voltages, theta_deg, phi_deg) -> FarField:
    """Gains and fields at (theta, phi) pairs (broadcast together)."""
    v = _voltages(sol, port_voltages)
    th, ph = np.broadcast_arrays(np.asarray(theta_deg, dtype=np.float64), np.asarray(phi_deg, dtype=np.float64))
    shape = th.shape
    gains, fields, valid, pin = sol._c._far_field(int(i_f), v, np.ascontiguousarray(th.ravel()),
                                                  np.ascontiguousarray(ph.ravel()))
    g = np.frombuffer(gains, dtype=np.float64).reshape(-1, 3)
    e = np.frombuffer(fields, dtype=np.complex128).reshape(-1, 2)
    ok = np.frombuffer(valid, dtype=np.uint8).astype(bool)
    return FarField(g[:, 0].reshape(shape), g[:, 1].reshape(shape), g[:, 2].reshape(shape),
                    e[:, 0].reshape(shape), e[:, 1].reshape(shape), ok.reshape(shape), pin)


def near_field(sol: Solution, i_f: int, port_voltages, points, kind: str = "E") -> np.ndarray:
    """E (V/m) or H (A/m) at points (..., 3) in metres; returns (..., 3) complex."""
    v = _voltages(sol, port_voltages)
    if kind not in ("E", "H"):
        raise ValueError("kind: 'E' or 'H'")
    p = np.asarray(points, dtype=np.float64)
    if p.shape[-1:] != (3,):
        raise ValueError("points: expected shape (..., 3)")
    out = sol._c._near_field(int(i_f), v, np.ascontiguousarray(p.reshape(-1, 3)), 0 if kind == "E" else 1)
    return np.frombuffer(out, dtype=np.complex128).reshape(p.shape)
