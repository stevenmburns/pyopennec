"""pyopennec: OpenNEC behind an N-port admittance interface.

The whole API is :mod:`pyopennec.porty`::

    from pyopennec import porty
    st = porty.Structure([(1, 21, (0, 0, -5.2), (0, 0, 5.2), 0.001)], ports=[(1, 11)])
    sol = st.solve([14.0, 14.2])
    z = 1 / sol.y[:, 0, 0]

The package itself only says which build it is.
"""

from __future__ import annotations

from . import porty
from ._porty import BACKEND, OPENMP, OPENNEC_VERSION

__version__ = "0.3.0"

__all__ = ["BACKEND", "OPENMP", "OPENNEC_VERSION", "porty"]

