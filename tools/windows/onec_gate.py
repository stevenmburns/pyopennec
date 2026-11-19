"""Run onec over a set of decks and compare two sets of reports.

    onec_gate.py run --exe onec.exe --decks DIR --out DIR [--threads 4] [--only a.nec,b.nec]
    onec_gate.py compare REF_DIR TEST_DIR
    onec_gate.py control REF_DIR
    onec_gate.py threads --exe onec.exe --deck a.nec
    onec_gate.py soak --exe onec.exe --deck a.nec --runs 2000

`compare` reads every ANTENNA INPUT PARAMETERS row of every report in REF_DIR.
Reports print six significant figures (%12.5E), so two correct engines can differ by
one unit in the last printed digit from rounding alone: the gate is one such
unit (with a margin for the rounding of the two sides) on R and on X.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import nec_output  # noqa: E402


def child_env(threads: int | None, blas: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"):
        env.pop(k, None)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    if blas is not None:
        env["OPENBLAS_NUM_THREADS"] = str(blas)
    return env


def watch(cmd: list[str], env: dict[str, str], cwd: str, timeout: float) -> dict:
    """Run to completion, sampling the peak thread count and the CPU time."""
    import psutil

    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ps = psutil.Process(p.pid)
    peak, cpu = 0, 0.0
    while p.poll() is None:
        if time.perf_counter() - t0 > timeout:
            p.kill()
            p.wait()
            return {"rc": "timeout", "wall": timeout, "peak_threads": peak, "cpu": cpu}
        try:
            peak = max(peak, ps.num_threads())
            c = ps.cpu_times()
            cpu = c.user + c.system
        except psutil.Error:
            pass
        time.sleep(0.02)
    err = p.stderr.read().decode(errors="replace")
    return {
        "rc": p.returncode,
        "wall": round(time.perf_counter() - t0, 3),
        "peak_threads": peak,
        "cpu": round(cpu, 2),
        "stderr": err[-500:],
    }


def cmd_run(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    decks = sorted(Path(a.decks).glob("*.nec"))
    if a.only:
        decks = [d for d in decks if d.name in a.only.split(",")]
    exe = str(Path(a.exe).resolve())
    bad = 0
    times = {}
    for d in decks:
        r = watch(
            [exe, "-i", str(d.resolve()), "-o", str(out / (d.stem + ".out"))],
            child_env(a.threads, a.threads),
            str(out),
            a.timeout,
        )
        times[d.name] = r
        print(f"{d.name:40s} rc={r['rc']} {r['wall']:9.2f} s  peak threads {r['peak_threads']}", flush=True)
        bad += r["rc"] != 0
    (out / "times.json").write_text(json.dumps(times, indent=1))
    return 1 if bad else 0


def last_digit_unit(v: float) -> float:
    return 10.0 ** (math.floor(math.log10(abs(v))) - 5) if v else 1e-300


def cmd_compare(a: argparse.Namespace) -> int:
    ref, test = Path(a.ref), Path(a.test)
    rows = bad = 0
    worst = 0.0
    for f in sorted(ref.glob("*.out")):
        other = test / f.name
        if not other.exists():
            print(f"MISSING {f.name}")
            bad += 1
            continue
        ra = nec_output.input_parameters(f.read_text(errors="replace"))
        rb = nec_output.input_parameters(other.read_text(errors="replace"))
        if len(ra) != len(rb) or not ra:
            print(f"{f.name}: {len(ra)} rows against {len(rb)}")
            bad += not (len(ra) == len(rb) == 0)
            continue
        here = 0.0
        for x, y in zip(ra, rb, strict=True):
            rows += 1
            za, zb = x[5], y[5]
            here = max(here, abs(za - zb) / abs(za))
            for u, v in ((za.real, zb.real), (za.imag, zb.imag)):
                if abs(u - v) > 1.01 * last_digit_unit(max(abs(u), abs(v))):
                    print(f"DIFF {f.name} {x[0]} MHz tag {x[1]} seg {x[2]}: {za} against {zb}")
                    bad += 1
        worst = max(worst, here)
        print(f"{f.name:40s} {len(ra):4d} rows  max |dZ|/|Z| {here:.2e}")
    print(f"{rows} impedance rows, {bad} outside one printed digit, worst |dZ|/|Z| {worst:.2e}")
    return 1 if bad or not rows else 0


def cmd_control(a: argparse.Namespace) -> int:
    """The negative control: `compare` must fail on a report whose impedance
    moved by two units in the last printed digit, and pass on an exact copy."""
    ref = Path(a.ref)
    victim = max(ref.glob("*.out"), key=lambda f: len(nec_output.input_parameters(f.read_text(errors="replace"))))
    text = victim.read_text(errors="replace")
    z = nec_output.input_parameters(text)[0][5].real
    old = f"{z:.5E}"
    if old not in text:
        print(f"control: {old} is not printed in {victim.name}")
        return 1
    moved = f"{z + 2 * last_digit_unit(z):.5E}"
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "ref").mkdir()
        (d / "same").mkdir()
        (d / "moved").mkdir()
        shutil.copy(victim, d / "ref" / victim.name)
        shutil.copy(victim, d / "same" / victim.name)
        (d / "moved" / victim.name).write_text(re.sub(re.escape(old), moved, text, count=0))
        same = cmd_compare(argparse.Namespace(ref=d / "ref", test=d / "same"))
        caught = cmd_compare(argparse.Namespace(ref=d / "ref", test=d / "moved"))
    print(f"control on {victim.name}: exact copy rc={same}, {old} moved to {moved} rc={caught}")
    return 0 if same == 0 and caught == 1 else 1


def cmd_threads(a: argparse.Namespace) -> int:
    exe = str(Path(a.exe).resolve())
    deck = Path(a.deck).resolve()
    cwd = str(deck.parent)
    out = str(deck.parent / "threads_probe.out")
    bad = 0
    for label, omp, blas, most in (
        ("OMP=1", 1, None, 2),
        ("OMP=2", 2, None, 4),
        ("OMP=4", 4, None, 8),
        ("OMP=4 OPENBLAS=1", 4, 1, 8),
        ("OMP=1 OPENBLAS=4", 1, 4, 8),
    ):
        r = watch([exe, "-i", str(deck), "-o", out], child_env(omp, blas), cwd, a.timeout)
        par = r["cpu"] / r["wall"] if r["wall"] else 0.0
        print(f"{label:20s} rc={r['rc']} wall {r['wall']:8.2f} s  cpu/wall {par:4.2f}  peak threads {r['peak_threads']}")
        bad += r["rc"] != 0 or r["peak_threads"] > most
    return 1 if bad else 0


def cmd_soak(a: argparse.Namespace) -> int:
    exe = str(Path(a.exe).resolve())
    deck = Path(a.deck).resolve()
    out = str(deck.parent / "soak.out")
    env = child_env(4, 4)
    env["ONEC_EXIT_TRACE"] = "1"
    hangs = fails = 0
    t0 = time.perf_counter()
    for k in range(a.runs):
        # stdout piped and read to the end, the shape of the published exit hang
        p = subprocess.Popen(
            [exe, "-i", str(deck), "-o", out], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            _, err = p.communicate(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            hangs += 1
            print(f"HANG at run {k}", flush=True)
            p.kill()
            p.communicate()
            continue
        if p.returncode != 0 or b"onec_exit: TerminateProcess(0)" not in err:
            fails += 1
            print(f"run {k}: rc={p.returncode} stderr={err[-300:]!r}", flush=True)
    print(f"{a.runs} runs in {time.perf_counter() - t0:.0f} s: {hangs} hangs, {fails} failures")
    return 1 if hangs or fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exe", required=True)
    r.add_argument("--decks", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--only", default="")
    r.add_argument("--timeout", type=float, default=2400)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("test")
    n = sub.add_parser("control")
    n.add_argument("ref")
    t = sub.add_parser("threads")
    t.add_argument("--exe", required=True)
    t.add_argument("--deck", required=True)
    t.add_argument("--timeout", type=float, default=600)
    s = sub.add_parser("soak")
    s.add_argument("--exe", required=True)
    s.add_argument("--deck", required=True)
    s.add_argument("--runs", type=int, default=2000)
    s.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "control": cmd_control, "threads": cmd_threads, "soak": cmd_soak}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
