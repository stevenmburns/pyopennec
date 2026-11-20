# pyopennec

[OpenNEC](https://github.com/maurymarkowitz/OpenNEC) (a C rewrite of NEC-2)
behind an N-port admittance interface.  A structure of straight wires is
solved once per frequency for its port admittance matrix `Y` and the segment
currents each port drives; everything a circuit does -- feed voltages,
transmission lines, matching networks, phasing -- is the caller's, on `Y`.
Radiation and near fields for any set of port voltages are computed from the
stored solution without solving again.

The extension is plain C on the CPython Limited API, so one wheel per
platform (tagged `cp312-abi3`) installs on every CPython from 3.12 on.

## A dipole

```python
from pyopennec import porty

st = porty.Structure(
    [(1, 21, (0.0, 0.0, -5.2), (0.0, 0.0, 5.2), 0.001)],  # tag, segments, end 1, end 2, radius (m)
    ports=[(1, 11)],                                        # segment 11 of tag 1
)
sol = st.solve([13.8, 14.2, 14.6])                          # MHz
z = 1.0 / sol.y[:, 0, 0]                                    # one port: Z = 1 / Y
ff = porty.far_field(sol, 1, [1.0], theta_deg=[90.0], phi_deg=[0.0])
print(z, ff.gain_total)
```

## API

All of it is `pyopennec.porty`.  Lengths are metres, frequencies MHz,
angles degrees.

**`Structure(wires, *, ports, ground=None, loads=(), extended_kernel=False)`**

* `wires`: `(tag, nseg, (x1, y1, z1), (x2, y2, z2), radius)` per straight
  wire (NEC's GW).  Wires that share an end are joined as NEC joins them.
* `ports`: `(tag, seg)` per port, `seg` 1-based within the tag (tag 0
  counts segments over the whole structure).  A port is an applied-field
  voltage source on that segment (NEC's EX 0).
* `ground`: `None` (free space), `"perfect"` (GN 1), `("reflection", epsr,
  sigma)` (GN 0) or `("sommerfeld", epsr, sigma)` (GN 2), sigma in S/m.  A
  ground also sets NEC's GE ground-plane flag, so a wire end on z = 0
  connects to its image.
* `loads`: an LD card's fields, `(type, tag, seg_from, seg_to, f1, f2, f3)`,
  types 0-5 (series / parallel RLC, per-metre series / parallel RLC, fixed
  impedance, wire conductivity), NEC's units.
* `extended_kernel`: NEC's extended thin-wire kernel (EK 0).
* `.nseg`, `.nports`.

**`Structure.solve(freqs_mhz, cancel=None, progress=None) -> Solution`** fills,
factors and solves once per frequency, with the GIL released.  `cancel` is a
`threading.Event` or a callable returning `bool`, checked before each
frequency; `progress(done, total)` is called after each.

**`Solution`**

* `y`: `(nf, nports, nports)` complex, siemens.  `Y[i, j]` is the current on
  port `i`'s segment with 1 V on port `j` and every other port shorted, so
  **`I = Y @ V`**.  NEC-2 point-matches at segment centres, so `Y` is
  reciprocal only to discretisation error (about 0.2 % off-diagonal at
  modest meshes): the orientation matters, and `Y.T @ V` is not the same
  thing.
* `basis_currents`: `(nf, nports, nseg)` complex, amperes -- the segment
  currents for 1 V on each port.  `currents(i_f, port_voltages)` superposes
  them.
* `freqs_mhz`, `port_segments` (1-based, over the whole structure) and
  `segments` (`centres` (nseg, 3), `lengths`, `radii` in metres; `tags`,
  `numbers` within the tag).

**`far_field(sol, i_f, port_voltages, theta_deg, phi_deg) -> FarField`** at
the broadcast `(theta, phi)` pairs: `gain_total`, `gain_vert`, `gain_horiz`
(dBi, power gain), `e_theta`, `e_phi` (r·E, volts), `valid` (False below
the horizon over ground, where the gains read -999.99) and `power_in` (W
delivered by the port voltages).

**`near_field(sol, i_f, port_voltages, points, kind="E")`**: E (V/m) or H
(A/m) at `points` of shape `(..., 3)`, returned complex `(..., 3)`.

Field queries on one `Solution` may come from several threads; they are
serialised on that solution.  Separate solves run concurrently.

**Refusals.**  Invalid input raises `ValueError` naming the field
(`wires[1].nseg`, `ports[0].seg: no segment 6 on tag 2`, `loads[0].type`,
`ground.epsr`, `port_voltages: expected 2 values`, a geometry below a
ground, port voltages that deliver no power, ...) before anything reaches
the engine.  An engine failure after validation raises `porty.PortyError`; a
set cancel raises `porty.Cancelled`.  Out of scope by design: surface
patches, helices and arcs (write them as straight wires), plane-wave and
current-source excitation, NT / TL (do them on `Y`), GN 3 and NGF files.

`pyopennec.OPENNEC_VERSION`, `pyopennec.BACKEND` (`"openblas"`,
`"scipy-openblas32"`, `"accelerate"` or `"original"`) and `pyopennec.OPENMP`
say which build you have.

**Fill reuse.** From OpenNEC 2.4.2, a build with OpenMP and OpenBLAS or
Accelerate (every wheel) fills each repeated wire-pair geometry once and
copies it. The answers move by about 1e-10 relative at most (pairs match
on quantised geometry), far below a printed digit.
`Structure(..., fill_reuse=False)` turns it off for that structure's solves
alone (`True` turns it on); `None`, the default, leaves it to
`OPENNEC_FILL_REUSE` and then the build. Prefer the argument in a program
that solves several structures at once: the environment is process-wide.

Why not PyNEC's card-by-card API: a circuit solver wants the port
admittances and nothing else, and an interface of its own keeps pyopennec
free of anything derived from PyNEC (GPL).

## Install

Release wheels (one per platform, `cp312-abi3`: any CPython >= 3.12) are
attached to each [GitHub Release](https://github.com/stevenmburns/pyopennec/releases);
PyPI comes later.  Install one by URL,
`https://github.com/stevenmburns/pyopennec/releases/download/v<ver>/<wheel>`:

* **Linux x86_64** (glibc >= 2.27):
  `pyopennec-<ver>-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`.
  LAPACK is OpenBLAS from the scipy-openblas32 wheel, vendored.  OpenMP is
  the system's GCC runtime, `libgomp.so.1`, deliberately not vendored
  (present on every glibc distribution; `apt install libgomp1` if not).
* **macOS arm64** (macOS >= 14):
  `pyopennec-<ver>-cp312-abi3-macosx_14_0_arm64.whl`.  LAPACK is Apple's
  Accelerate.  OpenMP is Homebrew's libomp, deliberately not vendored, so
  first `brew install libomp`.
* **Windows**: no wheel yet.  OpenNEC is C99 `complex double`, which MSVC
  does not have; `.github/workflows/windows-mingw.yml` builds one with
  MinGW-w64, but releases do not carry it yet.

**onec.exe for Windows.**  From v0.3.0 each release also carries
`onec-windows-x64.zip`: OpenNEC's console engine as one signed program per
CPU level (`onec.exe` for any x86-64, `onec-avx2.exe` for AVX2), built with
MinGW-w64 with OpenBLAS linked in, so it needs nothing beside it on
Windows 10 or later.  `onec.exe -i antenna.nec -o antenna.out`; its
`README.txt` covers threads, and its licence files and
`THIRD-PARTY-NOTICES.txt` the libraries linked in.

The exact filenames for a release are on its page; `pyopennec.BACKEND`
(`"scipy-openblas32"` / `"accelerate"`) and `pyopennec.OPENMP` say what you
got.

**One process with momwire and PyNEC.**  pyopennec loads beside
[momwire](https://pypi.org/project/momwire/) and
[pynec-accel](https://pypi.org/project/pynec-accel/) (PyNEC), all three
OpenMP-threaded, with no environment workarounds.  That works because none
of the three wheels ships a private OpenMP runtime: they all share one image
(the system `libgomp` on Linux, Homebrew's `libomp` on macOS).  A private copy
is what breaks it -- on Linux a vendored libgomp fails to load once another
extension loaded the system one ("cannot allocate memory in static TLS
block"), on macOS two libomp copies abort with OMP Error #15 or deadlock.
`tools/cross_engine.py` checks this in every import order (CI runs it against
the built wheel): accelerators loaded, exactly one OpenMP runtime mapped,
dipole impedances agreeing.

### From source

```
git clone --recursive https://github.com/stevenmburns/pyopennec
cd pyopennec
python3.12 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

or build the wheel once and install it into any CPython >= 3.12:

```
python3.12 -m pip wheel . --no-deps -w dist   # dist/pyopennec-*-cp312-abi3-linux_x86_64.whl
python3.14 -m pip install dist/pyopennec-*.whl
```

The build needs a C compiler and CMake >= 3.26 (no C++).  It compiles every
OpenNEC library source (all of `opennec/src` except `main.c`) with OpenMP.
The LAPACK is chosen by the CMake cache variable `PYOPENNEC_BLAS` (pass it as
`CMAKE_ARGS="-DPYOPENNEC_BLAS=..."`):

* `auto` (the default): a system OpenBLAS + LAPACK when `pkg-config
  openblas` finds them (Debian/Ubuntu: `apt install libopenblas-dev
  liblapack-dev gfortran`), else Accelerate on macOS, else OpenNEC's own
  matrix code (`BACKEND == "original"`, much slower).
* `openblas`, `scipy-openblas32` (with `-DSCIPY_OPENBLAS32_DIR=<its package
  dir>`, or importable by the build interpreter), `accelerate`, `original`:
  exactly that one, and the configure fails if it cannot be had.

`-DPYOPENNEC_REQUIRE_OPENMP=ON` makes a missing OpenMP fatal too (macOS: `brew
install libomp` and add `-DOpenMP_ROOT=$(brew --prefix libomp)`).  The release
wheels are built with both, so a wheel can never silently ship the fallback.

## How it works

Per frequency, porty runs the steps of OpenNEC's sequential processor
without its deck reader or report writer: scale the geometry to
wavelengths, set the ground (and the Sommerfeld tables for GN 2), apply the
loads, fill the interaction matrix, LU-factor it, and solve for every port at
once as one multi-right-hand-side solve.  Fields for given port voltages
superpose the stored per-port solutions and call OpenNEC's own pattern and
near-field routines, so a field query never refactors.  No text is parsed or
printed and no Python is called from inside OpenNEC.

## Tests

```
.venv/bin/python -m pytest
```

* `test_porty.py` holds porty to recorded card-deck runs of the same
  OpenNEC sources (`tests/fixtures/porty_oracle.json`, every float stored
  exactly): Z, Y, superposition and near fields to 1e-9, gains to 1e-6 dB.
  The fixture was recorded during private development from an earlier
  card-deck module built on the same OpenNEC sources (opennec `2531311`);
  this tree cannot regenerate it, so after a submodule bump that moves the
  numbers the print-precision oracles below are the gate.
* `test_porty_vs_cli.py` needs no recording: all twelve golden decks of
  `opennec/tests/golden_tests` (nec2c pins, 1e-4; the TL and NT decks with
  the network applied on `Y`), the onec CLI built from the same submodule
  (`tools/build_onec.sh`) at print precision, and nec2c when it is on `PATH`.
* Reciprocity, threads, cancellation, validation messages and RSS growth.

## Licence and credits

MIT (see `LICENSE`).  The engine is OpenNEC by Maury Markowitz, MIT
(`opennec/LICENSE`), itself a rewrite of NEC-2, the Numerical
Electromagnetics Code by G. J. Burke and A. J. Poggio of Lawrence Livermore
National Laboratory.
