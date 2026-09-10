"""
Tests the finite-difference check the adjoint is verified against.

`perturbation_method` moves one boundary value at one time step and runs the
flow model again. The value is written into the memory MODFLOW solves from, and
MODFLOW rereads a boundary only where the stress period data gives it again, so
a perturbation that is not put back runs on to the end of the simulation. Each
step then answers for every step after it, and the sum over steps counts the
same response once per step.

Cases:
  - test_matches_the_adjoint    : the two agree over a range of time
                                  discretizations, and both agree with a
                                  difference taken outside mf6adj.
  - test_single_step_unchanged  : one period of one step, which was always
                                  right, stays right.
"""

import pathlib as pl
import shutil
import sys

import flopy
import numpy as np
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NCOL = 10
WELL = (0, 0, 4)
RATE = -20.0


def _build(ws, nper, nstp, rate=RATE):
    """A strip aquifer with a stream at one end, a well, and a constant head."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="pert", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=[(100.0, nstp, 1.0)] * nper)
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="pert", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=NCOL, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf, stress_period_data=[[(0, 0, 0), 5.0, 1000.0]], pname="drn-1"
    )
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, NCOL - 1), 10.0]], pname="chd-1"
    )
    # given for the first period only, so MODFLOW carries it forward and never
    # rereads it, which is what leaves a perturbation standing
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[WELL, rate]], pname="wel-1")
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="pert.hds",
        budget_filerecord="pert.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure capture\n")
        f.write(f"  {nper} {nstp} 1 1 1 drn-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return ws


def _leakage(ws, nper, nstp):
    cbc = flopy.utils.CellBudgetFile(str(pl.Path(ws) / "pert.cbc"))
    rec = cbc.get_data(kstpkper=(nstp - 1, nper - 1), text="DRN")[0]
    return float(np.sum([r[2] for r in rec]))


def _both(ws):
    """Return the adjoint sensitivity and the perturbation one."""
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    results = adj.solve_adjoint()
    perturbed = adj.perturbation_method(pert_mult=1.001)
    adj.finalize()

    adjoint = float(results["capture"]["wel6_q"].to_numpy().ravel()[WELL[2]])
    rows = perturbed.loc[perturbed.addr.str.contains("wel"), "capture"].to_numpy()
    assert rows.size == 1, f"expected one well, found {rows.size}"
    return adjoint, float(rows[0])


@pytest.mark.parametrize("nper, nstp", [(1, 2), (1, 4), (2, 1), (2, 3), (3, 4)])
def test_matches_the_adjoint(function_tmpdir, nper, nstp):
    """The perturbation reproduces the adjoint, and both a plain difference.

    Left standing, the perturbation made this too large by the number of time
    steps: twelve times over for three periods of four steps.
    """
    ws = _build(function_tmpdir / f"p{nper}x{nstp}", nper, nstp)

    epsilon = abs(RATE) * 1.0e-3
    up = _build(function_tmpdir / f"u{nper}x{nstp}", nper, nstp, rate=RATE + epsilon)
    down = _build(function_tmpdir / f"d{nper}x{nstp}", nper, nstp, rate=RATE - epsilon)
    difference = (_leakage(up, nper, nstp) - _leakage(down, nper, nstp)) / (
        2.0 * epsilon
    )

    adjoint, perturbed = _both(ws)
    assert np.isclose(perturbed, difference, rtol=1.0e-3), (
        f"the perturbation {perturbed:.6e} is not the difference "
        f"{difference:.6e}, a factor of {perturbed / difference:.3f}"
    )
    assert np.isclose(perturbed, adjoint, rtol=1.0e-3)


def test_single_step_unchanged(function_tmpdir):
    """One period of one step, which the sum over steps could not overcount."""
    ws = _build(function_tmpdir / "one", 1, 1)
    adjoint, perturbed = _both(ws)
    assert np.isclose(perturbed, adjoint, rtol=1.0e-6)
