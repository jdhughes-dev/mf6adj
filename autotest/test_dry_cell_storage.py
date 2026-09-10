"""
Tests the storage term that carries the adjoint state back through a dry cell.

The previous head enters the specific-yield term through the saturation it
sets, so the derivative carries the slope of that saturation. MODFLOW 6 rounds
the saturation at both ends of a cell, and in that rounding the slope falls
away to nothing. A cell that has emptied does not hold a saturation of exactly
zero, it holds whatever the rounding gives it, so a term that asks only whether
the saturation is above zero hands back the whole coupling for a cell holding
no water.

Where that coupling is larger than the diagonal of the matrix, the adjoint
state is multiplied by the ratio once per time step, and the sensitivities grow
without bound backward in time.

Cases:
  - test_slope_matches_modflow   : the slope is MODFLOW's, over the thickness.
  - test_slope_at_the_ends       : an empty or a full cell passes nothing back.
  - test_slope_inside_rounding   : a cell just above its bottom passes back
                                   almost nothing, not the whole term.
  - test_without_newton          : a model that does not use the Newton-Raphson
                                   formulation has no rounding, and the slope
                                   is the plain one it was before.
"""

import pathlib as pl
import sys

import numpy as np
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.packages.storage import smoothed_saturation_slope

# MODFLOW sets this only under the Newton-Raphson formulation, and leaves it
# at zero otherwise
NEWTON_OMEGA = 1.0e-6

TOP, BOT = 10.0, 0.0
THICK = TOP - BOT


def modflow_slope(head, top=TOP, bot=BOT, eps=NEWTON_OMEGA):
    """sQuadraticSaturationDerivative of SmoothingFunctions.f90, over the thickness."""
    b = top - bot
    if head < bot:
        br = 0.0
    elif head > top:
        br = 1.0
    else:
        br = (head - bot) / b
    av = 1.0 / (1.0 - eps)
    if br < eps:
        return av * br / eps
    if br < 1.0 - eps:
        return av
    if br < 1.0:
        return av * (1.0 - br) / eps
    return 0.0


@pytest.mark.parametrize(
    "head",
    [
        -5.0,
        0.0,
        1.0e-9,
        1.0e-6,
        1.0e-4,
        0.5,
        2.0,
        5.0,
        9.0,
        THICK * (1.0 - NEWTON_OMEGA / 2.0),
        10.0,
        15.0,
    ],
)
def test_slope_matches_modflow(head):
    """The slope is the one MODFLOW forms, at every part of the cell."""
    got = float(
        smoothed_saturation_slope(
            np.array([head]), np.array([TOP]), np.array([BOT]), NEWTON_OMEGA
        )[0]
    )
    assert np.isclose(got, modflow_slope(head), rtol=1e-12, atol=1e-14)


def test_slope_at_the_ends():
    """A cell that is empty or full passes nothing back."""
    heads = np.array([BOT - 1.0, BOT, TOP, TOP + 1.0])
    slope = smoothed_saturation_slope(heads, np.full(4, TOP), np.full(4, BOT))
    assert np.allclose(slope, 0.0)


def test_slope_inside_rounding():
    """A cell just above its bottom passes back almost nothing.

    This is the case that grew without bound: a head a hundred-millionth of a
    metre above the bottom of the cell leaves a saturation that is positive but
    negligible, and the term it was given was the whole one.
    """
    head = np.array([BOT + 3.3e-8])
    slope = float(
        smoothed_saturation_slope(head, np.array([TOP]), np.array([BOT]), NEWTON_OMEGA)[
            0
        ]
    )
    # the fraction of the cell the head stands in, over the rounding width
    assert np.isclose(slope, (3.3e-8 / THICK) / NEWTON_OMEGA / (1.0 - NEWTON_OMEGA))
    assert slope < 1.0e-2, "a cell holding no water still carries the whole term"

    # and the slope rises to one only once the head is clear of the rounding
    clear = np.array([BOT + 2.0 * NEWTON_OMEGA * THICK])
    assert np.isclose(
        float(
            smoothed_saturation_slope(
                clear, np.array([TOP]), np.array([BOT]), NEWTON_OMEGA
            )[0]
        ),
        1.0 / (1.0 - NEWTON_OMEGA),
    )


def test_slope_is_bounded():
    """The slope never exceeds the linear one, wherever the head stands."""
    heads = np.linspace(BOT - 1.0, TOP + 1.0, 20001)
    slope = smoothed_saturation_slope(
        heads, np.full(heads.size, TOP), np.full(heads.size, BOT), NEWTON_OMEGA
    )
    assert slope.min() >= 0.0
    assert slope.max() <= 1.0 / (1.0 - NEWTON_OMEGA) + 1.0e-12


def test_without_newton():
    """No rounding without Newton-Raphson, so the slope is the plain one.

    MODFLOW leaves ``satomega`` at zero outside the Newton-Raphson formulation
    (gwf-sto.f90), where the saturation follows the head over the whole cell.
    The term is then what it was before this was changed, so a model solved
    that way is unaffected.
    """
    heads = np.array([BOT - 1.0, BOT, BOT + 3.3e-8, 5.0, TOP, TOP + 1.0])
    slope = smoothed_saturation_slope(
        heads, np.full(heads.size, TOP), np.full(heads.size, BOT), 0.0
    )
    assert np.array_equal(slope, np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
