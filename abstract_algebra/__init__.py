"""Vendored subset of alreich/abstract_algebra (MIT licensed).

Source: https://github.com/alreich/abstract_algebra (see LICENSE in this directory).

Only the modules required by MAD's group (`group-S`/`group-Z`/`group-A`) tasks are
included: finite_algebras, cayley_table, permutations, my_math, abstract_matrix.

Upstream ships these as flat top-level modules that import one another absolutely
(e.g. ``from cayley_table import CayleyTable``). To keep the vendored files
byte-for-byte identical to upstream, we add this package's directory to ``sys.path``
so those flat imports resolve, then re-export ``finite_algebras`` as
``abstract_algebra.finite_algebras`` to match the imports used across the codebase.
"""

import os as _os
import sys as _sys

_pkg_dir = _os.path.dirname(__file__)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from . import finite_algebras  # noqa: E402,F401
