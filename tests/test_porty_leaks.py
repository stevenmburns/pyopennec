"""Gate: repeated porty solves, field queries and failing calls keep RSS flat.

Warm-up cycles are excluded (allocator arenas, OpenMP/OpenBLAS pools).  The
threshold is per cycle: an engine context dropped without engine_destroy()
alone is ~20 KiB (the Sommerfeld tables), a lost interaction matrix far more.
"""
import ctypes
import gc
import sys
import threading

import numpy as np
import pytest

from pyopennec import porty

WARMUP = 50
CYCLES = 200

ARRAY = [
    (1, 11, (-5.0, 0.0, 8.0), (5.0, 0.0, 8.0), 0.002),
    (2, 11, (-5.3, 3.0, 8.0), (5.3, 3.0, 8.0), 0.002),
    (3, 9, (-4.6, -2.5, 8.5), (4.6, -2.5, 7.5), 0.0015),
]
PORTS = [(1, 6), (2, 6), (3, 3)]
LOADS = ((0, 1, 3, 3, 10.0, 1e-6, 0.0), (5, 0, 0, 0, 5.8e7, 0.0, 0.0))
TH = np.linspace(0.0, 180.0, 10)
PH = np.zeros(10)
PTS = np.array([[0.0, 1.0, 9.0], [1.0, 1.0, 9.5]])


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64), ("user_time", ctypes.c_int32 * 2),
                ("system_time", ctypes.c_int32 * 2), ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32)]


def rss_kib():
    """Current resident set: /proc on Linux, the Mach task's resident_size
    (MACH_TASK_BASIC_INFO, what `ps -o rss` reads) on macOS."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None)
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        task = ctypes.c_uint32.in_dll(libc, "mach_task_self_")
        kr = libc.task_info(task, 20, ctypes.byref(info), ctypes.byref(count))  # 20: MACH_TASK_BASIC_INFO
        if kr != 0:
            raise RuntimeError(f"task_info: kern_return {kr}")
        return info.resident_size // 1024
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("no VmRSS")


def good_cycle(k):
    ground = ("sommerfeld", 13.0, 0.005) if k % 10 == 0 else ("reflection", 13.0, 0.005)
    st = porty.Structure(ARRAY, ports=PORTS, ground=ground, loads=LOADS)
    sol = st.solve([14.0 + 0.01 * (k % 5), 21.0], progress=lambda i, n: None)
    porty.far_field(sol, k % 2, [1.0, 0.5j, -0.2], TH, PH)
    porty.near_field(sol, 0, [1.0, 0.5j, -0.2], PTS)
    return sol.y[0, 0, 0]


def expect(exc, fn, *a, **kw):
    # not pytest.raises: its ExceptionInfo bookkeeping shows up in RSS
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{fn} did not raise {exc.__name__}")


def bad_cycle(k):
    for kw in (dict(ports=[(1, 99)]), dict(ports=[(1, 6)], loads=[(9, 1, 1, 1, 0, 0, 0)]),
               dict(ports=[(1, 6), (1, 6)]), dict(ports=[(1, 6)], ground=("reflection", 0.1, 0.0))):
        expect(ValueError, porty.Structure, ARRAY, **kw)
    expect(ValueError, porty.Structure, [(1, 9, (0, 0, -1), (0, 0, 1), 0.01)], ports=[(1, 5)],
           ground="perfect")  # below ground
    st = porty.Structure(ARRAY, ports=PORTS)
    ev = threading.Event()
    n = [0]

    def prog(i, total):
        n[0] += 1
        if n[0] == 2:
            ev.set()

    expect(porty.Cancelled, st.solve, [10.0, 12.0, 14.0], cancel=ev, progress=prog)
    expect(ValueError, st.solve, [10.0, float("nan")])
    sol = st.solve([14.0])
    expect(ValueError, porty.far_field, sol, 0, [0.0, 0.0, 0.0], TH, PH)
    expect(ValueError, porty.far_field, sol, 3, [1.0, 0.0, 0.0], TH, PH)


@pytest.mark.parametrize("cycle", [good_cycle, bad_cycle], ids=["solve+fields", "error-paths"])
def test_rss_flat(cycle, record_property):
    for k in range(WARMUP):
        cycle(k)
    gc.collect()
    before = rss_kib()
    for k in range(CYCLES):
        cycle(k)
    gc.collect()
    grown = rss_kib() - before
    record_property("rss_growth_kib", grown)
    print(f"\n{cycle.__name__}: RSS {before} KiB -> {before + grown} KiB over {CYCLES} cycles "
          f"({grown / CYCLES:.2f} KiB/cycle)")
    assert grown / CYCLES < 4.0
