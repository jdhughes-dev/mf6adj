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
        ``nzero``, ``nsmall``, the median diagonal, the worst rows with their
        diagonals, and every flagged row, or None when every row is sound.
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
    order = flagged[np.argsort(diagonal[flagged])]
    worst = order[:WORST]
    return {
        "nzero": int(zero.sum()),
        "nsmall": int(small.sum()),
        "median": median,
        "rows": worst,
        "diagonals": diagonal[worst],
        "flagged": order,
        "flagged_diagonals": diagonal[order],
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


def cellid(node: int, nodeuser=None, grid_shape=None) -> str:
    """Return the cell of a row of the adjoint matrix, as the model names it.

    The matrix is assembled over the nodes left after the model drops the
    cells it does not solve, so a row of it is not a cell of the grid the user
    wrote. ``nodeuser`` carries a row back to the node it came from, and the
    shape of a structured grid carries that node to a layer, row and column.

    Parameters
    ----------
    node : int
        Zero-based row of the matrix.
    nodeuser : ndarray of int, optional
        Zero-based reduced node to user node map.
    grid_shape : tuple of int, optional
        ``(nlay, nrow, ncol)`` of a structured grid, or ``(nlay, ncpl)`` of a
        grid of vertices. A grid of neither is named by its node.

    Returns
    -------
    str
        One-based cell identifier.
    """
    node = int(node)
    user = int(nodeuser[node]) if nodeuser is not None else node
    if grid_shape is not None and len(grid_shape) == 3:
        k, i, j = np.unravel_index(user, grid_shape)
        return f"layer {int(k) + 1}, row {int(i) + 1}, column {int(j) + 1}"
    if grid_shape is not None and len(grid_shape) == 2:
        k, n = np.unravel_index(user, grid_shape)
        return f"layer {int(k) + 1}, cell {int(n) + 1}"
    return f"node {user + 1}"


def describe(report: dict, kper: int, kstp: int, nodeuser=None, grid_shape=None) -> str:
    """Return the message for a matrix whose rows cannot support a solution."""
    rows = "; ".join(
        f"{cellid(node, nodeuser, grid_shape)} ({value:.3e})"
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
        f"than the flow model can support. Worst cells: {rows}."
    )


# square root of machine precision, MODFLOW 6's DPRECSQRT
DPRECSQRT = float(np.sqrt(np.finfo(float).eps))


def dry_cells(saturation, tol: float = DPRECSQRT) -> np.ndarray:
    """Return the cells holding no water.

    Parameters
    ----------
    saturation : ndarray
        Cell saturation.
    tol : float
        Saturation below which a cell holds no water.

    Returns
    -------
    ndarray
        Indices of the cells holding none.
    """
    return np.flatnonzero(np.asarray(saturation).ravel() < tol)


def flagged_nodes(
    report: dict, kper: int, kstp: int, nodeuser=None, grid_shape=None
) -> str:
    """Return every cell of a report, one to a line, for the log file.

    The console names the worst few. A model of a few million nodes can flag
    more than a console holds, so the whole list is kept where it is read
    later rather than watched.
    """
    lines = [
        f"cells the adjoint matrix for stress period {kper + 1}, time step "
        + f"{kstp + 1} cannot carry a sensitivity through:"
    ]
    for node, diagonal in zip(report["flagged"], report["flagged_diagonals"]):
        where = cellid(node, nodeuser, grid_shape)
        lines.append(f"  {where}, diagonal {float(diagonal):.6e}")
    return "\n".join(lines)
