import os
import sys

import pyopennec
from pyopennec import _porty


def test_version():
    assert pyopennec.__version__ == "0.2.2"


def test_engine_reports_its_build():
    assert pyopennec.OPENNEC_VERSION.count(".") == 2
    assert pyopennec.BACKEND in ("openblas", "scipy-openblas32", "accelerate", "original")
    assert isinstance(pyopennec.OPENMP, bool)
    assert _porty.LAPACK_DIRECT == (pyopennec.BACKEND != "original")


def test_expected_build():
    """PYOPENNEC_EXPECT_BACKEND / PYOPENNEC_EXPECT_OPENMP=1 (set by the wheel
    jobs) pin what the build under test must be, so a wheel that fell back to
    OpenNEC's own matrix code or lost OpenMP fails here."""
    want = os.environ.get("PYOPENNEC_EXPECT_BACKEND")
    if want:
        assert pyopennec.BACKEND == want
    if os.environ.get("PYOPENNEC_EXPECT_OPENMP") == "1":
        assert pyopennec.OPENMP is True


def test_one_abi3_extension():
    """The only compiled module is _porty, built on the 3.12 Limited API."""
    assert _porty.LIMITED_API == 0x030C0000
    assert _porty.__file__.endswith(".abi3.so")
    assert "pyopennec._core" not in sys.modules
    assert pyopennec.porty.Structure is not None
