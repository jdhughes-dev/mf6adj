"""Report an adjoint matrix that cannot carry a trustworthy sensitivity.

The adjoint is solved against the matrix MODFLOW 6 assembled. A row whose
diagonal has fallen away still solves, and the state it returns goes as one
over that diagonal, so a measure can come back with values far larger than the
flow model supports and nothing says why.

A cell the model barely connects to is such a row, and a steady-state step is
where it shows: a transient step carries a storage term on the diagonal, of the
order of the specific yield over the length of the step, which holds it up. A
cell going dry under the Newton-Raphson formulation is not such a row, measured
rather than assumed: at four tenths of a percent saturated its diagonal sat at
the median of the model, held there by that storage term, and its sensitivities
fell rather than grew.

The checks here are the ones affordable on a model of a few million nodes: they
read the diagonal, which is one pass, and the residual of the solve, which is
one matrix-vector product. A singular value decomposition tells more but cannot
be run at that size.
"""

from typing import Optional

import numpy as np

# a row whose diagonal is this far below the middle of the model is reported
SMALL_DIAGONAL = 1.0e-6

# a solve is reported when its residual is this large next to its right side
LOOSE_RESIDUAL = 1.0e-6

# how many of the worst rows to name
WORST = 5


def diagonal_report(amat, threshold: float = SMALL_DIAGONAL) -> Optional[dict]:
    """Return the rows of a matrix that cannot support a solution, or None.

    A row with no diagonal at all is singular. A row whose diagonal is far
    below the rest of the model is not, but the state it carries goes as one
    over that diagonal, so it is where an implausible sensitivity comes from.

    Parameters
    ----------
    amat : scipy.sparse.spmatrix
        Matrix the adjoint is solved against.
    threshold : float
        Fraction of the median diagonal below which a row is reported.

    Returns
    -------
    dict or None
        ``nzero``, ``nsmall``, the median diagonal, and the worst rows with
        their diagonals, or None when every row is sound.
    """
    diagonal = np.abs(np.asarray(amat.diagonal()))
    if diagonal.size == 0:
        return None

    nonzero = diagonal[diagonal > 0.0]
    if nonzero.size == 0:
        return {
            "nzero": int(diagonal.size),
            "nsmall": 0,
            "median": 0.0,
            "rows": np.arange(min(WORST, diagonal.size)),
            "diagonals": diagonal[: min(WORST, diagonal.size)],
        }

    median = float(np.median(nonzero))
    zero = diagonal <= 0.0
    small = (~zero) & (diagonal < threshold * median)
    if not zero.any() and not small.any():
        return None

    flagged = np.flatnonzero(zero | small)
    worst = flagged[np.argsort(diagonal[flagged])][:WORST]
    return {
        "nzero": int(zero.sum()),
        "nsmall": int(small.sum()),
        "median": median,
        "rows": worst,
        "diagonals": diagonal[worst],
    }


def solve_residual(amat, lamb, rhs) -> float:
    """Return the residual of a solved system next to its right-hand side.

    Parameters
    ----------
    amat : scipy.sparse.spmatrix
        Matrix that was solved.
    lamb : ndarray
        Solution returned by the solver.
    rhs : ndarray
        Right-hand side it was solved against.

    Returns
    -------
    float
        Largest residual over the largest right-hand side, or the largest
        residual alone where that side is zero.
    """
    residual = float(np.max(np.abs(amat.dot(lamb) - rhs)))
    scale = float(np.max(np.abs(rhs)))
    return residual / scale if scale > 0.0 else residual


def describe(report: dict, kper: int, kstp: int) -> str:
    """Return the message for a matrix whose rows cannot support a solution."""
    rows = ", ".join(
        f"{int(node)} ({value:.3e})"
        for node, value in zip(report["rows"], report["diagonals"])
    )
    what = []
    if report["nzero"]:
        what.append(f"{report['nzero']} with no diagonal")
    if report["nsmall"]:
        what.append(f"{report['nsmall']} with a diagonal far below the rest")
    return (
        f"the adjoint matrix for stress period {kper + 1}, time step "
        f"{kstp + 1} holds {' and '.join(what)}, against a median of "
        f"{report['median']:.3e}. The state at such a node goes as one over "
        f"its diagonal, so a sensitivity reported there may be far larger "
        f"than the flow model can support. Worst nodes, zero based: {rows}."
    )
