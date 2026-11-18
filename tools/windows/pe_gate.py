"""What a Windows binary imports and exports, as a gate.

    python pe_gate.py exe onec.exe
    python pe_gate.py pyd _porty.pyd

exe: every imported DLL is a Windows system DLL or the UCRT.
pyd: the same plus python3.dll (the stable ABI; python312.dll would tie the
     module to one CPython), and the only export is PyInit__porty.
Post-Windows-7 imports are listed, not refused: no claim about Windows 7 is
made for these builds.
"""

from __future__ import annotations

import re
import sys

import pefile

SYSTEM = {
    "advapi32.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "kernel32.dll",
    "ntdll.dll",
    "ucrtbase.dll",
    "user32.dll",
}
UCRT = re.compile(r"^api-ms-win-crt-[a-z]+-l1-1-0\.dll$")
POST_WIN7 = {
    "GetSystemTimePreciseAsFileTime",
    "SetThreadDescription",
    "GetThreadDescription",
    "WaitOnAddress",
    "WakeByAddressSingle",
    "WakeByAddressAll",
    "GetCurrentThreadStackLimits",
    "SetThreadInformation",
    "GetSystemCpuSetInformation",
}


def main(kind: str, path: str) -> int:
    pe = pefile.PE(path)
    bad = []
    newer = []
    print(f"{path}: {kind}")
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        dll = entry.dll.decode().lower()
        names = [i.name.decode() for i in entry.imports if i.name]
        print(f"  imports {dll} ({len(names)})")
        newer += [f"{dll}!{n}" for n in names if n in POST_WIN7]
        allowed = dll in SYSTEM or UCRT.match(dll) or (kind == "pyd" and dll == "python3.dll")
        if not allowed:
            bad.append(dll)
    if getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", None):
        bad += [e.dll.decode().lower() for e in pe.DIRECTORY_ENTRY_DELAY_IMPORT]
    print(f"  post-Windows-7 imports: {newer or 'none'}")
    if kind == "pyd":
        pe.parse_data_directories()
        exports = [
            s.name.decode()
            for s in getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", [])
            if s.name
        ]
        print(f"  exports: {exports}")
        if exports != ["PyInit__porty"]:
            print(f"ERROR: unexpected exports {exports}")
            return 1
    if bad:
        print(f"ERROR: not system DLLs: {bad}")
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
