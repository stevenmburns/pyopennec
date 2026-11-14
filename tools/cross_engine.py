"""Three OpenMP engines in one Python process: pyopennec, momwire, PyNEC.

pyopennec is meant to load beside momwire and pynec-accel (PyNEC), all three
OpenMP-threaded, with no OpenMP runtime clash.  The clash has two known
shapes, both caused by an extension shipping a PRIVATE copy of the runtime:

* Linux: a vendored libgomp (auditwheel's ``<pkg>.libs/libgomp-<hash>.so``)
  uses initial-exec TLS and fails to dlopen ("cannot allocate memory in
  static TLS block") once another extension loaded the system libgomp, so
  that engine's accelerator silently drops out.
* macOS: two libomp images abort with OMP Error #15 or deadlock at a join
  barrier.

So the check is structural, not just "it did not crash": after importing the
engines and running a threaded solve on each, exactly ONE OpenMP runtime
image may be mapped in the process.

  python tools/cross_engine.py            # driver: every order, each in a child
  python tools/cross_engine.py --engines pyopennec,momwire,pynec   # one order, in-process

The driver runs each engine alone, every ordered pair and every order of all
three (so pyopennec both first and last), each in a fresh interpreter with
OMP_NUM_THREADS=4 and a timeout (a deadlock fails instead of hanging), and
exits non-zero if any run fails.  Each run asserts:

* the engine's accelerator loaded (``momwire.accelerated``;
  ``pyopennec.OPENMP`` and a LAPACK backend; PyNEC has no flag, so its
  extension must be the image that mapped the runtime when it runs alone);
* the runtime images mapped, deduplicated by realpath, number exactly one,
  and its ``omp_get_max_threads()`` is 4;
* every result is finite, and when more than one engine ran, the dipole
  impedances agree within 5 % (|Za - Zb| / |Zb|) -- three different
  discretisations of one wire, so this is a sanity bound, not a parity gate.

How the runtime images are counted:

* Linux: ``/proc/self/maps`` is the kernel's list of what is mapped into the
  process; every path whose basename is libgomp* / libomp* / libiomp*
  (system or vendored, any soname mangling) counts.
* macOS: there is no /proc.  dyld's own image list (``_dyld_image_count`` /
  ``_dyld_get_image_name`` in libSystem, via ctypes) is the authoritative,
  in-process list of loaded Mach-O images -- the same thing ``vmmap`` reports,
  without spawning a tool that needs task_for_pid rights, and unlike ``otool``
  it describes what actually loaded rather than what a binary asks for.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

THREADS = 4
FREQ_MHZ = 14.2
# a 10 m dipole, 1 mm radius, centre-fed: near resonance at 14.2 MHz, so all
# three engines read it in the same few-percent neighbourhood
HALF = 5.0
RADIUS = 0.001
NSEG = 51
ENGINES = ("pyopennec", "momwire", "pynec")
RUNTIME_RE = re.compile(r"^lib(gomp|omp|iomp)[0-9]*([.-].*)?\.(so[.0-9]*|dylib)$")


# --- the engines --------------------------------------------------------------


def run_pyopennec():
    import pyopennec
    from pyopennec import porty

    assert pyopennec.OPENMP is True, "pyopennec built without OpenMP"
    assert pyopennec.BACKEND != "original", (
        f"pyopennec has no LAPACK backend ({pyopennec.BACKEND})"
    )
    st = porty.Structure(
        [(1, NSEG, (0.0, 0.0, -HALF), (0.0, 0.0, HALF), RADIUS)],
        ports=[(1, NSEG // 2 + 1)],
    )
    sol = st.solve([FREQ_MHZ])
    return complex(
        1.0 / sol.y[0, 0, 0]
    ), f"pyopennec {pyopennec.__version__} {pyopennec.BACKEND}"


def run_momwire():
    import momwire
    from momwire.deck import build_solver, parse

    assert momwire.accelerated, "momwire accelerator did not load"
    deck = (
        "CM cross-engine dipole\nCE\n"
        f"GW 1 {NSEG} 0. 0. {-HALF} 0. 0. {HALF} {RADIUS}\n"
        "GE 0\n"
        f"EX 0 1 {NSEG // 2 + 1} 0 1. 0.\n"
        f"FR 0 1 0 0 {FREQ_MHZ}\nXQ\nEN\n"
    )
    built = build_solver(parse(deck), basis="bspline")
    y = built.solver.compute_port_solution().y
    port = built.ports.feed_ports[0]
    try:
        from importlib.metadata import version

        ver = version("momwire")
    except Exception:  # noqa: BLE001 -- the version is decoration only
        ver = "?"
    return complex(1.0 / y[port, port]), f"momwire {ver} bspline"


def run_pynec():
    import PyNEC

    nec = PyNEC.nec_context()
    geo = nec.get_geometry()
    geo.wire(1, NSEG, 0.0, 0.0, -HALF, 0.0, 0.0, HALF, RADIUS, 1.0, 1.0)
    nec.geometry_complete(0)
    nec.fr_card(0, 1, FREQ_MHZ, 0.0)
    nec.ex_card(0, 1, NSEG // 2 + 1, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nec.xq_card(0)
    z = nec.get_input_parameters(0).get_impedance()[0]
    try:
        from importlib.metadata import version

        ver = version("pynec-accel")
    except Exception:  # noqa: BLE001 -- the version is decoration only
        ver = "?"
    return complex(z), f"pynec-accel {ver}"


RUNNERS = {"pyopennec": run_pyopennec, "momwire": run_momwire, "pynec": run_pynec}
MODULES = {"pyopennec": "pyopennec", "momwire": "momwire", "pynec": "PyNEC"}


# --- the OpenMP runtime images ------------------------------------------------


def _loaded_images():
    if sys.platform.startswith("linux"):
        paths = set()
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    paths.add(parts[5].strip())
        return sorted(paths)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        return [
            libc._dyld_get_image_name(i).decode()
            for i in range(libc._dyld_image_count())
        ]
    raise SystemExit(f"unsupported platform {sys.platform}")


def openmp_runtimes():
    """Distinct OpenMP runtime images mapped in this process (by realpath)."""
    found = {}
    for p in _loaded_images():
        if RUNTIME_RE.match(os.path.basename(p)):
            found.setdefault(os.path.realpath(p), p)
    return sorted(found)


def max_threads(runtime):
    lib = ctypes.CDLL(runtime)
    lib.omp_get_max_threads.restype = ctypes.c_int
    return lib.omp_get_max_threads()


# --- one order, in this process -----------------------------------------------


def run_order(engines):
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise SystemExit(f"run with OMP_NUM_THREADS={THREADS} set before Python starts")
    report = {"engines": engines, "platform": sys.platform, "z": {}, "who": {}}
    for e in engines:  # import in the stated order, before any solve
        importlib.import_module(MODULES[e])
    if engines == ["pynec"]:
        # PyNEC has no accelerator flag: alone, its extension must be what
        # mapped the runtime, or it was built without OpenMP.
        assert openmp_runtimes(), "PyNEC alone mapped no OpenMP runtime"
    t0 = time.perf_counter()
    for rnd in range(2):  # twice, interleaved: a region after another engine's
        for e in engines:
            z, who = RUNNERS[e]()
            assert math.isfinite(z.real) and math.isfinite(z.imag), f"{e}: Z = {z}"
            if rnd:
                assert abs(
                    z - report["z"][e][0] - 1j * report["z"][e][1]
                ) <= 1e-6 * abs(z), f"{e}: second solve differs: {z}"
            report["z"][e] = [z.real, z.imag]
            report["who"][e] = who
    report["seconds"] = round(time.perf_counter() - t0, 3)
    rts = openmp_runtimes()
    report["runtimes"] = rts
    assert len(rts) == 1, (
        f"expected exactly one OpenMP runtime image, found {len(rts)}: {rts}"
    )
    report["omp_max_threads"] = max_threads(rts[0])
    assert report["omp_max_threads"] == THREADS, report["omp_max_threads"]
    worst = 0.0
    for a, b in itertools.combinations(engines, 2):
        za, zb = complex(*report["z"][a]), complex(*report["z"][b])
        worst = max(worst, abs(za - zb) / abs(zb))
    report["worst_rel_dz"] = worst
    assert worst < 0.05, f"dipole Z disagree by {worst:.3%}: {report['z']}"
    if sys.platform.startswith("linux"):
        report["threads_in_process"] = len(os.listdir("/proc/self/task"))
    print("CROSS-ENGINE " + json.dumps(report), flush=True)


# --- the negative control ------------------------------------------------------


def negative_control():
    """The counter must see a private copy: import pyopennec, then dlopen a
    copy of the runtime it mapped under a vendored-looking name (what
    auditwheel / delocate would ship), and require a count of two.  Without
    this, "exactly one" could be a check that never sees a second image.
    Nothing calls into the copy, so it never initialises."""
    import shutil
    import tempfile

    import pyopennec  # noqa: F401

    (rt,) = openmp_runtimes()
    d = tempfile.mkdtemp(prefix="pyopennec.libs-")
    stem, ext = os.path.basename(rt).split(".", 1)
    copy = os.path.join(d, f"{stem}-deadbeef.{ext}")
    shutil.copy(rt, copy)
    ctypes.CDLL(copy)
    seen = openmp_runtimes()
    assert len(seen) == 2, f"the counter missed a private runtime copy: {seen}"
    print(
        f"NEGATIVE-CONTROL ok: a vendored copy reads as {len(seen)} runtimes {seen}",
        flush=True,
    )


# --- the driver ----------------------------------------------------------------


def orders(engines):
    out = [[e] for e in engines]
    out += [list(p) for p in itertools.permutations(engines, 2)]
    out += [list(p) for p in itertools.permutations(engines, 3)]
    return out


def drive(engines, timeout):
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS))
    # a workaround in the environment would hide exactly what this checks
    for k in ("KMP_DUPLICATE_LIB_OK", "GLIBC_TUNABLES"):
        env.pop(k, None)
    failed = 0
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--negative-control"],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    print((r.stdout + r.stderr).strip(), flush=True)
    if r.returncode:
        print(
            "FAIL negative control: the runtime counter cannot be trusted", flush=True
        )
        failed += 1
    for order in orders(engines):
        name = ",".join(order)
        cmd = [sys.executable, os.path.abspath(__file__), "--engines", name]
        try:
            r = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"FAIL {name}: timed out after {timeout} s (deadlock?)", flush=True)
            failed += 1
            continue
        line = next(
            (ln for ln in r.stdout.splitlines() if ln.startswith("CROSS-ENGINE ")), None
        )
        if r.returncode or line is None:
            failed += 1
            print(
                f"FAIL {name}: exit {r.returncode}\n{r.stdout}\n{r.stderr}", flush=True
            )
            continue
        rep = json.loads(line[len("CROSS-ENGINE ") :])
        zs = "  ".join(f"{e}={complex(*rep['z'][e]):.2f}" for e in order)
        print(
            f"ok   {name:28s} runtimes={len(rep['runtimes'])} {rep['runtimes'][0]}"
            f"  threads={rep['omp_max_threads']}  worst dZ={rep['worst_rel_dz']:.2%}  {zs}",
            flush=True,
        )
        if len(order) == 3:
            print(f"     {rep['who']}", flush=True)
    total = len(orders(engines)) + 1
    print(
        f"{total - failed}/{total} runs passed (negative control + {total - 1} orders)",
        flush=True,
    )
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engines", help="comma list: run this order in-process")
    ap.add_argument(
        "--only", default=",".join(ENGINES), help="driver: the engines to combine"
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--negative-control", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.negative_control:
        negative_control()
        return 0
    if a.engines:
        run_order(a.engines.split(","))
        return 0
    return drive(a.only.split(","), a.timeout)


if __name__ == "__main__":
    sys.exit(main())
