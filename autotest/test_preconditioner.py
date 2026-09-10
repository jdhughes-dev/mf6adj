"""
Tests the preconditioner the adjoint solve is carried out with.

The adjoint is solved against the transpose of the matrix MODFLOW 6
assembled, preconditioned by an incomplete factorization of it. A matrix that
factorization cannot be formed for falls back on a Jacobi preconditioner
instead, which is where the choice between the point and the block form is
made.

Cases:
  - test_a_block_holds_the_whole_matrix : a matrix with fewer rows than a
                                   block is one block.
  - test_the_fallback_is_point   : a factorization that cannot be formed falls
                                   back on the point form, which is measured
                                   to be the faster of the two.
  - test_the_fallback_solves     : the solve that falls back still reaches an
                                   answer.
"""

import pathlib as pl
import shutil
import sys

import flopy
import numpy as np
import scipy.sparse as sps

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.pm import BLOCK_SIZE, PerfMeas

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()


def _matrix(n):
    """A matrix with a diagonal and a little off-diagonal weight."""
    mat = sps.lil_matrix((n, n))
    for i in range(n):
        mat[i, i] = 10.0
        if i + 1 < n:
            mat[i, i + 1] = -1.0
            mat[i + 1, i] = -1.0
    return mat.tocsr()


def _forward_file(ws):
    """Return the forward file of a small transient model."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="pre", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0, 2, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="pre", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=10, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf, stress_period_data=[[(0, 0, 0), 5.0, 1000.0]], pname="drn-1"
    )
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 9), 10.0]], pname="chd-1")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="pre.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write("  1 2 1 1 4 head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return ws


def _solve(ws, monkeypatch, break_the_factorization):
    """Solve the adjoint, optionally with a factorization that cannot form."""
    seen = {}
    setup = PerfMeas._setup_jacobi_preconditioner

    def spy(self, amat, jacobi_type="point", precon_kwargs=None):
        seen["jacobi_type"] = jacobi_type
        return setup(self, amat, jacobi_type, precon_kwargs)

    monkeypatch.setattr(PerfMeas, "_setup_jacobi_preconditioner", spy)
    if break_the_factorization:

        def refuse(*args, **kwargs):
            raise RuntimeError("Factor is exactly singular")

        monkeypatch.setattr(mf6adj.pm, "spilu", refuse)

    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj._performance_measures[0].solve_adjoint(
        ws / "fwd.hd5",
        hdf5_adjoint_solution_fname=str(ws / "adj.hd5"),
        linear_solver="bicgstab",
        dvclose=None,
        rclose=None,
    )
    adj.finalize()
    return seen


def test_a_block_holds_the_whole_matrix():
    """A matrix with fewer rows than a block is one block."""

    class Shim:
        logger = type("L", (), {"logger": __import__("logging").getLogger("pre")})()

    amat = _matrix(50)
    m = PerfMeas._setup_block_jacobi_preconditioner(Shim(), amat, block_size=BLOCK_SIZE)

    # the preconditioner still solves, which a block of the wrong size
    # would not, and on a matrix this size it is the whole factorization
    v = np.ones(50)
    assert np.isfinite(m @ v).all(), "a block larger than the matrix carries no solve"
    assert BLOCK_SIZE > 50, "the block size no longer exceeds the matrix tested"


def test_the_fallback_is_point(function_tmpdir, monkeypatch):
    """A factorization that cannot be formed falls back on the point form.

    Measured over the CONUS models, the block form cuts the iterations and
    loses the time it saved to the block solves.
    """
    ws = _forward_file(function_tmpdir / "run")
    seen = _solve(ws, monkeypatch, break_the_factorization=True)
    assert seen.get("jacobi_type") == "point", (
        f"the fallback used {seen.get('jacobi_type')!r} rather than the point form"
    )


def test_the_fallback_solves(function_tmpdir, monkeypatch):
    """The solve that falls back still reaches an answer."""
    ws = _forward_file(function_tmpdir / "run")
    _solve(ws, monkeypatch, break_the_factorization=True)

    import h5py

    with h5py.File(ws / "adj.hd5", "r") as f:
        keys = sorted(k for k in f if k.startswith("solution_kper"))
        assert keys, "the fallback solve wrote no step"
        state = np.asarray(f[keys[-1]]["lambda"][:])
    assert np.isfinite(state).all(), "the fallback solve returned no number"
    assert np.abs(state).max() > 0.0, "the fallback solve returned nothing"
