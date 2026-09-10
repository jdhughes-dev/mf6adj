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
  - test_state_is_held_at_zero : the solve acts on the selection, on a forward
                               file with one cell put into the state that grew.

No model this size reaches that state on its own. MODFLOW gives a cell that
empties altogether a diagonal of its own, and the state there is already
nothing, so a test on such a model holds whether the solve acts or not. The
condition wants a cell the smoothing leaves at a saturation of 1e-19 with a
diagonal that has fallen below the term carrying the state back, which was
found on a model of a few million cells. The forward file of a small model is
edited into it instead: the adjoint is solved from that file and never from a
running model, so nothing else is needed.
"""

import pathlib as pl
import shutil
import sys

import flopy
import h5py
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


def _build(ws, nstp=4):
    """A strip aquifer with a stream at one end, a well, and a constant head."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    mf6_bin, _ = mf6adj.get_conda_mf6_paths()

    sim = flopy.mf6.MFSimulation(sim_name="dry", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0, nstp, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="dry", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=10, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, sy=0.2, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf, stress_period_data=[[(0, 0, 0), 5.0, 1000.0]], pname="drn-1"
    )
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 9), 10.0]], pname="chd-1")
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[(0, 0, 4), -20.0]], pname="wel-1")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="dry.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 {nstp} 1 1 1 drn-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return ws


def _empty_one_cell(src, dst, node, sat=1.0e-19, diagonal=3.0e-4, carry=2.0e-3):
    """Put one cell of a forward file into the state a dry cell reaches."""
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        ia = np.asarray(f["gwf_info"]["sln_ia"][:]).astype(int)
        ja = np.asarray(f["gwf_info"]["sln_ja"][:]).astype(int)
        if ia.min() == 1:
            ia, ja = ia - 1, ja - 1

        iconvert = f["gwf_info"]["iconvert"][:]
        iconvert.reshape(-1)[node] = 1
        f["gwf_info"]["iconvert"][...] = iconvert

        for key in [k for k in f if k.startswith("solution_kper")]:
            group = f[key]
            for name in ("sat", "sat_old"):
                values = np.asarray(group[name][:])
                values.reshape(-1)[node] = sat
                group[name][...] = values

            # the diagonal falls away while the term carrying the state back
            # does not, which multiplies the state once per time step
            amat = np.asarray(group["amat"][:])
            for position in range(ia[node], ia[node + 1]):
                amat[position] = -diagonal if ja[position] == node else 0.0
            group["amat"][...] = amat

            drhsdh = np.asarray(group["drhsdh"][:])
            drhsdh.reshape(-1)[node] = -carry
            group["drhsdh"][...] = drhsdh


def test_state_is_held_at_zero(function_tmpdir):
    """The solve holds the state at zero where the forward file says dry.

    Left to grow, the state at that cell reaches 1.6e+07 over four time steps,
    multiplied by the ratio of the coupling to the diagonal once per step.
    """
    node = 6
    ws = _build(function_tmpdir / "run")
    _, lib_name = mf6adj.get_conda_mf6_paths()

    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    _empty_one_cell(ws / "fwd.hd5", ws / "dried.hd5", node)
    adj._performance_measures[0].solve_adjoint(
        ws / "dried.hd5", hdf5_adjoint_solution_fname=str(ws / "dried_adj.hd5")
    )
    adj.finalize()

    with h5py.File(ws / "dried_adj.hd5", "r") as out:
        steps = sorted(k for k in out if k.startswith("solution_kper"))
        assert len(steps) > 1, "the state has to cross more than one step to grow"
        for key in steps:
            state = np.asarray(out[key]["lambda"][:]).ravel()
            assert state[node] == 0.0, f"{key} carries {state[node]:.4e} at a dry cell"
            assert np.abs(state).max() <= 1.0 + 1.0e-6, (
                f"{key} reports {np.abs(state).max():.4e}, above what a "
                "capture fraction can reach"
            )
