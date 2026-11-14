import sys

import pyopennec
from pyopennec import _porty


def test_version():
    assert pyopennec.__version__ == "0.2.0.dev0"


def test_engine_reports_its_build():
    assert pyopennec.OPENNEC_VERSION.count(".") == 2
    assert pyopennec.BACKEND in ("openblas", "original")
    assert isinstance(pyopennec.OPENMP, bool)
    assert _porty.LAPACK_DIRECT == (pyopennec.BACKEND == "openblas")


def test_one_abi3_extension():
    """The only compiled module is _porty, built on the 3.12 Limited API."""
    assert _porty.LIMITED_API == 0x030C0000
    assert _porty.__file__.endswith(".abi3.so")
    assert "pyopennec._core" not in sys.modules
    assert pyopennec.porty.Structure is not None
