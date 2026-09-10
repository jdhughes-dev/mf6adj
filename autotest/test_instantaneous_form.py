"""
Tests what an instantaneous measure reports, and what it depends on.

An instantaneous measure holds the earlier time steps fixed rather than letting
them feed back, so what it reports belongs to the model and to the way the run
is discretized in time together. The same model solved with more time steps
reports a smaller sensitivity. That is what the form is specified to do, and it
is pinned here so a change to it is a decision and not an accident.

A capture fraction wants the total derivative instead, which is a `direct`
measure at a single time step.

Cases:
  - test_direct_is_steady        : a direct measure at one time gives the same
                                   answer however the run is discretized.
  - test_instantaneous_moves     : the same measure in the instantaneous form
                                   does not, and falls as the steps are refined.
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
PERLEN = 3000.0


def _solve(ws, nstp, form):
    """Return the well sensitivity of a measure over the whole run."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="inst", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(PERLEN, nstp, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="inst", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=NCOL, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-2, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf, stress_period_data=[[(0, 0, 0), 5.0, 500.0]], pname="drn-1"
    )
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, NCOL - 1), 10.0]], pname="chd-1"
    )
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[WELL, -20.0]], pname="wel-1")
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord="inst.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure capture\n")
        if form == "instantaneous":
            # every step, which is what the form averages over
            for kstp in range(nstp):
                f.write(f"  1 {kstp + 1} 1 1 1 drn-1 instantaneous 1.0 -1.0e+30\n")
        else:
            f.write(f"  1 {nstp} 1 1 1 drn-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")

    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    results = adj.solve_adjoint()
    adj.finalize()
    return float(results["capture"]["wel6_q"].to_numpy().ravel()[WELL[2]])


@pytest.mark.parametrize("nstp", [4, 8, 16])
def test_direct_is_steady(function_tmpdir, nstp):
    """A direct measure at one time is the same however the run is stepped.

    It is the total derivative at that time, which belongs to the flow model
    and not to the discretization, so it is what a capture fraction wants.
    """
    coarse = _solve(function_tmpdir / "d4", 4, "direct")
    other = _solve(function_tmpdir / f"d{nstp}", nstp, "direct")
    assert np.isclose(other, coarse, rtol=1.0e-3), (
        f"{nstp} steps gives {other:.6e} where 4 gives {coarse:.6e}"
    )
    assert abs(coarse) <= 1.0 + 1.0e-6, "a capture fraction cannot exceed one"


def test_instantaneous_moves(function_tmpdir):
    """The instantaneous form falls as the time steps are refined.

    It holds the earlier steps fixed, so what a parameter does during one step
    is a smaller part of what it does over the run the shorter that step is.
    Two runs of one model that differ only in `nstp` therefore disagree: on
    this model four steps give -0.478 and sixteen give -0.345, a difference of
    28 percent. The form does not carry a quantity of the flow model alone.
    """
    coarse = _solve(function_tmpdir / "i4", 4, "instantaneous")
    fine = _solve(function_tmpdir / "i16", 16, "instantaneous")
    assert abs(fine) < abs(coarse), (
        f"refining the steps gave {fine:.6e} against {coarse:.6e}; the "
        "instantaneous form is expected to fall"
    )
    assert not np.isclose(fine, coarse, rtol=1.0e-2), (
        "the two agree, so this no longer depends on the discretization and "
        "the documented behaviour has changed"
    )
