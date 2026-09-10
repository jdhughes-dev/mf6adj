"""
Tests that a cell holding no water carries no adjoint state.

Nothing flows through a dry cell, so no sensitivity reaches through it. Its row
stays in the matrix, and under the Newton-Raphson formulation it still holds a
coefficient for every neighbour, whose residuals follow its head. What it loses
is its diagonal, which the storage term carrying the state back can exceed, and
the state is then multiplied once per time step.

Cases:
  - test_dry_below_precision : a saturation below the square root of machine
                               precision counts as dry, and one above it does not.
  - test_none_when_wet       : a model with no dry cell selects nothing.
  - test_shape_is_flat       : a grid-shaped saturation gives flat indices.
"""

import pathlib as pl
import sys

import numpy as np

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.utils.utils_conditioning import DPRECSQRT, dry_cells


def test_dry_below_precision():
    """A saturation below the square root of machine precision counts as dry."""
    assert np.isclose(DPRECSQRT, np.sqrt(np.finfo(float).eps))
    sat = np.array([1.0, 0.5, DPRECSQRT * 2.0, DPRECSQRT / 2.0, 3.19e-19, 0.0])
    assert np.array_equal(dry_cells(sat), np.array([3, 4, 5]))


def test_none_when_wet():
    """A model with no dry cell selects nothing."""
    assert dry_cells(np.array([1.0, 0.9, 0.5, 1.0e-3])).size == 0


def test_shape_is_flat():
    """A grid-shaped saturation gives indices into the flat node numbering."""
    sat = np.ones((1, 3, 4))
    sat[0, 2, 1] = 0.0
    assert np.array_equal(dry_cells(sat), np.array([9]))
