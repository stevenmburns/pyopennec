"""pyopennec beside momwire and PyNEC in one Windows process, measured.

On Linux and macOS the three share one OpenMP runtime, and
tools/cross_engine.py holds them to that. On Windows they cannot: momwire and
pynec-accel bring their own runtime DLLs, and the MinGW _porty.pyd has libgomp
linked in. So this does not count runtimes. It measures what a second runtime
could break:

* every import order of the engines (alone, pairs, triples, and numpy/scipy
  first and last), each in a fresh process with a timeout: no crash, no hang;
* each engine's answer is stable and the engines agree on a dipole;
* each engine's solve time beside the others, against that engine alone in its
  own process (a pool spinning idle after its region shows up here);
* all engines solving AT ONCE from Python threads (true oversubscription);
* --soak: fresh processes that import everything, solve and exit (exit hangs).

    python coexist.py                 # the driver
    python coexist.py --soak 300
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
import statistics
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cross_engine  # noqa: E402

THREADS = cross_engine.THREADS
ENGINES = ("pyopennec", "momwire", "pynec")
MODULES = dict(cross_engine.MODULES, scipy="scipy.linalg")
RUNTIME = re.compile(r"(vcomp|libomp|libiomp|libgomp|openblas|winpthread|flang|mkl)", re.I)
ARRAY = 12  # parallel dipoles
SEGS = 85  # segments each: 1,020 unknowns


def deck_wires():
    return [(k + 1, SEGS, (0.8 * k, 0.0, -5.0), (0.8 * k, 0.0, 5.0), 0.001) for k in range(ARRAY)]


def heavy_pyopennec():
    from pyopennec import porty

    st = porty.Structure(deck_wires(), ports=[(1, SEGS // 2 + 1)])
    return complex(1.0 / st.solve([14.2]).y[0, 0, 0])


def heavy_momwire():
    from momwire.deck import build_solver, parse

    gw = "\n".join(
        f"GW {t} {n} {a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]} {r}" for t, n, a, b, r in deck_wires()
    )
    deck = f"CM array\nCE\n{gw}\nGE 0\nEX 0 1 {SEGS // 2 + 1} 0 1. 0.\nFR 0 1 0 0 14.2\nXQ\nEN\n"
    built = build_solver(parse(deck), basis="bspline")
    y = built.solver.compute_port_solution().y
    p = built.ports.feed_ports[0]
    return complex(1.0 / y[p, p])


def heavy_pynec():
    import PyNEC

    nec = PyNEC.nec_context()
    geo = nec.get_geometry()
    for t, n, a, b, r in deck_wires():
        geo.wire(t, n, *a, *b, r, 1.0, 1.0)
    nec.geometry_complete(0)
    nec.fr_card(0, 1, 14.2, 0.0)
    nec.ex_card(0, 1, SEGS // 2 + 1, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nec.xq_card(0)
    return complex(nec.get_input_parameters(0).get_impedance()[0])


def heavy_scipy():
    import numpy as np
    import scipy.linalg as sl

    rng = np.random.default_rng(1)
    n = 1200
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal(n) + 0j
    x = sl.lu_solve(sl.lu_factor(a), b)
    y = np.linalg.solve(a, b)
    res = max(np.linalg.norm(a @ x - b), np.linalg.norm(a @ y - b)) / np.linalg.norm(b)
    assert res < 1e-8, f"scipy/numpy residual {res}"
    return complex(float(np.linalg.norm(x)), 0.0)


HEAVY = {"pyopennec": heavy_pyopennec, "momwire": heavy_momwire, "pynec": heavy_pynec, "scipy": heavy_scipy}


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


def thread_count() -> int:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
    e = THREADENTRY32()
    e.dwSize = ctypes.sizeof(e)
    n = 0
    pid = os.getpid()
    ok = k32.Thread32First(snap, ctypes.byref(e))
    while ok:
        n += e.th32OwnerProcessID == pid
        ok = k32.Thread32Next(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    return n


def runtime_modules() -> list[str]:
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    # the pseudo-handle is -1: without argtypes ctypes cannot pass it
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    h = k32.GetCurrentProcess()
    mods = (wintypes.HMODULE * 4096)()
    need = wintypes.DWORD()
    psapi.EnumProcessModules(h, mods, ctypes.sizeof(mods), ctypes.byref(need))
    names = []
    buf = ctypes.create_unicode_buffer(1024)
    for m in mods[: need.value // ctypes.sizeof(wintypes.HMODULE)]:
        psapi.GetModuleFileNameExW(h, m, buf, 1024)
        base = os.path.basename(buf.value)
        if RUNTIME.search(base):
            names.append(base)
    return sorted(set(names))


def child(order: list[str], rounds: int) -> None:
    rep: dict = {"order": order, "z": {}, "times": {}, "threads": {}}
    for e in order:
        importlib.import_module(MODULES[e])
    rep["modules_after_import"] = runtime_modules()
    rep["threads_after_import"] = thread_count()
    for e in order:  # the small dipole, cross_engine's own check
        if e in cross_engine.RUNNERS:
            rep["z"][e] = list(map(float, (lambda z: (z.real, z.imag))(cross_engine.RUNNERS[e]()[0])))
    names = [e for e in order if e in cross_engine.RUNNERS]
    for a, b in itertools.combinations(names, 2):
        za, zb = complex(*rep["z"][a]), complex(*rep["z"][b])
        assert abs(za - zb) / abs(zb) < 0.05, f"{a} and {b} disagree: {za} {zb}"
    first = {}
    for e in order:
        rep["times"][e] = []
    for _ in range(rounds):  # interleaved, so each solve follows another engine's
        for e in order:
            t0 = time.perf_counter()
            z = HEAVY[e]()
            rep["times"][e].append(time.perf_counter() - t0)
            assert math.isfinite(z.real) and math.isfinite(z.imag), f"{e}: {z}"
            if e in first:
                assert abs(z - first[e]) <= 1e-6 * abs(z), f"{e}: {z} then {first[e]}"
            first[e] = z
            rep["threads"][e] = thread_count()
    rep["median_s"] = {e: round(statistics.median(v), 4) for e, v in rep["times"].items()}
    del rep["times"]
    rep["modules"] = runtime_modules()
    print("COEXIST " + json.dumps(rep), flush=True)


def concurrent(order: list[str]) -> None:
    for e in order:
        importlib.import_module(MODULES[e])
    out, errors = {}, []

    def work(e):
        try:
            t0 = time.perf_counter()
            for _ in range(3):
                HEAVY[e]()
            out[e] = round((time.perf_counter() - t0) / 3, 4)
        except Exception as ex:  # noqa: BLE001 -- reported to the driver
            errors.append(f"{e}: {ex!r}")

    t0 = time.perf_counter()
    ts = [threading.Thread(target=work, args=(e,)) for e in order]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(
        "CONCURRENT "
        + json.dumps({"order": order, "per_solve_s": out, "wall": round(time.perf_counter() - t0, 2), "errors": errors}),
        flush=True,
    )
    assert not errors, errors


def spawn(args: list[str], timeout: float) -> tuple[int | str, str, float]:
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS))
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        o, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        o, _ = p.communicate()
        return "HANG", o, timeout
    return p.returncode, o, time.perf_counter() - t0


def drive(rounds: int, timeout: float) -> int:
    failed = 0
    base = {}
    for e in (*ENGINES, "scipy"):
        rc, o, _ = spawn(["--child", e, "--rounds", str(rounds)], timeout)
        line = next((ln for ln in o.splitlines() if ln.startswith("COEXIST ")), None)
        if rc != 0 or line is None:
            print(f"FAIL alone {e}: rc={rc}\n{o[-3000:]}", flush=True)
            failed += 1
            continue
        rep = json.loads(line[8:])
        base[e] = rep["median_s"][e]
        print(f"alone {e:10s} {base[e]:7.3f} s  threads {rep['threads'][e]:3d}  runtimes {rep['modules']}", flush=True)
    orders = [list(p) for p in itertools.permutations(ENGINES, 2)]
    orders += [list(p) for p in itertools.permutations(ENGINES, 3)]
    orders += [["scipy", *ENGINES], [*ENGINES, "scipy"]]
    worst = 0.0
    for order in orders:
        name = ",".join(order)
        rc, o, wall = spawn(["--child", name, "--rounds", str(rounds)], timeout)
        line = next((ln for ln in o.splitlines() if ln.startswith("COEXIST ")), None)
        if rc != 0 or line is None:
            print(f"FAIL {name}: rc={rc} after {wall:.0f} s\n{o[-3000:]}", flush=True)
            failed += 1
            continue
        rep = json.loads(line[8:])
        ratios = {e: rep["median_s"][e] / base[e] for e in order if e in base}
        worst = max(worst, *ratios.values())
        shown = "  ".join(f"{e} x{r:.2f}" for e, r in ratios.items())
        print(f"ok   {name:36s} {shown}  threads {max(rep['threads'].values())}", flush=True)
    print(f"worst slowdown beside the others: x{worst:.2f} (each engine against itself alone)", flush=True)
    for order in (list(ENGINES), list(reversed(ENGINES))):
        name = ",".join(order)
        rc, o, wall = spawn(["--concurrent", name], timeout)
        print(next((ln for ln in o.splitlines() if ln.startswith("CONCURRENT ")), o[-2000:]), flush=True)
        if rc != 0:
            print(f"FAIL concurrent {name}: rc={rc} after {wall:.0f} s", flush=True)
            failed += 1
    # a long interleaving: every engine's pool sleeps and wakes many times
    name = ",".join(ENGINES)
    rc, o, wall = spawn(["--child", name, "--rounds", "150"], timeout)
    line = next((ln for ln in o.splitlines() if ln.startswith("COEXIST ")), None)
    if rc != 0 or line is None:
        print(f"FAIL 150 interleaved rounds: rc={rc} after {wall:.0f} s\n{o[-3000:]}", flush=True)
        failed += 1
    else:
        print(f"ok   150 interleaved rounds of {name} in {wall:.0f} s", flush=True)
    total = len(orders) + 4 + 2 + 1
    print(f"{total - failed}/{total} runs passed", flush=True)
    return 1 if failed else 0


def soak(runs: int, timeout: float) -> int:
    hangs = fails = 0
    t0 = time.perf_counter()
    order = ",".join(ENGINES)
    for k in range(runs):
        rc, o, _ = spawn(["--soak-child", order], timeout)
        if rc == "HANG":
            hangs += 1
            print(f"HANG at run {k}\n{o[-500:]}", flush=True)
        elif rc != 0 or "SOAK-OK" not in o:
            fails += 1
            print(f"run {k}: rc={rc}\n{o[-800:]}", flush=True)
    print(f"{runs} process runs in {time.perf_counter() - t0:.0f} s: {hangs} hangs, {fails} failures", flush=True)
    return 1 if hangs or fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--concurrent")
    ap.add_argument("--soak-child")
    ap.add_argument("--soak", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    if a.child:
        child(a.child.split(","), a.rounds)
        return 0
    if a.concurrent:
        concurrent(a.concurrent.split(","))
        return 0
    if a.soak_child:
        for e in a.soak_child.split(","):
            importlib.import_module(MODULES[e])
        # the larger solves: the pools must be up when the process exits, which
        # a 51-segment dipole does not guarantee
        for e in a.soak_child.split(","):
            HEAVY[e]()
        n = thread_count()
        assert n > THREADS, f"only {n} threads alive at exit: the pools never started"
        print(f"SOAK-OK threads={n}", flush=True)
        return 0
    if a.soak:
        return soak(a.soak, 120.0)
    return drive(a.rounds, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
