"""fill_reuse= on Structure: OpenNEC 2.4.2's fill reuse per structure.

OpenNEC reads OPENNEC_FILL_REUSE at every fill, and the environment is
process-wide, so a server with solves in flight cannot use it to let one
request say no. The kwarg sets the context's own choice, which wins over the
environment in both directions. The proof is bitwise: with the environment
pointing the other way, the kwarg's answer equals the answer the environment
alone gives for the same choice. Reuse matches pairs on quantised geometry,
so on and off differ in the last digits (about 1e-10 here), which is what
makes the comparison able to fail."""

import numpy as np
import pytest

import pyopennec
from pyopennec import porty

REUSE_BUILD = pyopennec.OPENMP and pyopennec.BACKEND != "original"
pytestmark = pytest.mark.skipif(not REUSE_BUILD, reason="fill reuse needs OpenMP with OpenBLAS or Accelerate")

WIRES = [(1, 401, (0.0, -0.5, 0.0), (0.0, 0.5, 0.0), 1e-3)]
FREQS = [140.0, 150.0]


def _y(monkeypatch, env, **kw):
    if env is None:
        monkeypatch.delenv("OPENNEC_FILL_REUSE", raising=False)
    else:
        monkeypatch.setenv("OPENNEC_FILL_REUSE", env)
    return np.asarray(porty.Structure(WIRES, ports=[(1, 201)], **kw).solve(FREQS).y)


def test_reuse_moves_the_last_digits_only(monkeypatch):
    on, off = _y(monkeypatch, "1"), _y(monkeypatch, "0")
    assert not np.array_equal(on, off), "reuse did not engage: the comparisons below would prove nothing"
    assert np.max(np.abs(on - off) / np.abs(off)) < 1e-8


def test_the_kwarg_wins_over_the_environment(monkeypatch):
    on, off = _y(monkeypatch, "1"), _y(monkeypatch, "0")
    assert np.array_equal(_y(monkeypatch, "1", fill_reuse=False), off)
    assert np.array_equal(_y(monkeypatch, "0", fill_reuse=True), on)


def test_none_follows_the_environment_and_then_the_build(monkeypatch):
    on, off = _y(monkeypatch, "1"), _y(monkeypatch, "0")
    assert np.array_equal(_y(monkeypatch, "0", fill_reuse=None), off)
    assert np.array_equal(_y(monkeypatch, None), on)  # the build default is on


@pytest.mark.parametrize("bad", [0, 1, "yes"])
def test_fill_reuse_takes_a_bool_or_none(bad):
    with pytest.raises(TypeError):
        porty.Structure(WIRES, ports=[(1, 201)], fill_reuse=bad)
