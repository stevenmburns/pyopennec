import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "opennec" / "tests" / "golden_tests"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _newest_source_mtime():
    srcs = list((ROOT / "opennec" / "src").rglob("*.c")) + list((ROOT / "opennec" / "src").rglob("*.h"))
    return max(p.stat().st_mtime for p in srcs)


@pytest.fixture(scope="session")
def onec():
    """The onec CLI built from the same submodule commit as the extension.

    PYOPENNEC_ONEC overrides; otherwise tools/build_onec.sh builds it under
    build/onec, and rebuilds when any OpenNEC source is newer than it."""
    env = os.environ.get("PYOPENNEC_ONEC")
    if env:
        return env
    exe = ROOT / "build" / "onec" / "onec"
    if not exe.exists() or exe.stat().st_mtime < _newest_source_mtime():
        subprocess.run(["bash", str(ROOT / "tools" / "build_onec.sh"), str(exe.parent)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(exe)


@pytest.fixture(scope="session")
def golden():
    if not (GOLDEN / "expected.json").exists():
        pytest.skip("opennec submodule not checked out")
    return GOLDEN
