"""
Tests a sensitivity on a model whose time steps are not all the same length.

The storage term that carries the adjoint state backward in time belongs to the
later step's equation and is formed over that step's length. Where every step
has the same length the two cannot be told apart, so the cases here vary it.

Capture is the derivative of the leakage to a stream with respect to a well
rate, and is compared with a central difference taken by re-running the flow
model, which uses none of the adjoint machinery.

Cases:
  - test_capture_with_tsmult   : a period whose steps grow geometrically.
  - test_capture_across_periods: periods whose lengths differ by 10,000.
  - test_constant_timestep     : the case that was already right stays right.
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
DRN_ELEV, DRN_COND = 5.0, 1000.0


def _build(ws, perioddata, rate=RATE):
    """A strip aquifer with a stream at one end, a well, and a constant head."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="vdt", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=len(perioddata), perioddata=perioddata)
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="vdt", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=NCOL, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, sy=0.2, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data=[[(0, 0, 0), DRN_ELEV, DRN_COND]],
        pname="drn-1",
    )
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, NCOL - 1), 10.0]], pname="chd-1"
    )
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[WELL, rate]], pname="wel-1")
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="vdt.hds",
        budget_filerecord="vdt.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    kper, kstp = len(perioddata) - 1, perioddata[-1][1] - 1
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure capture\n")
        f.write(f"  {kper + 1} {kstp + 1} 1 1 1 drn-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return ws


def _leakage(ws, perioddata):
    """The stream leakage at the step the measure is taken on."""
    kper, kstp = len(perioddata) - 1, perioddata[-1][1] - 1
    cbc = flopy.utils.CellBudgetFile(str(pl.Path(ws) / "vdt.cbc"))
    rec = cbc.get_data(kstpkper=(kstp, kper), text="DRN")[0]
    return float(np.sum([r[2] for r in rec]))


def _adjoint(ws):
    """The reported sensitivity of the measure to the well rate."""
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    df = adj.solve_adjoint()
    adj.finalize()
    return float(df["capture"]["wel6_q"].to_numpy().ravel()[WELL[2]])


def _finite_difference(tmpdir, perioddata):
    """Central difference of the measure in the well rate, without the adjoint."""
    eps = abs(RATE) * 1.0e-3
    up = _build(tmpdir / "up", perioddata, rate=RATE + eps)
    down = _build(tmpdir / "down", perioddata, rate=RATE - eps)
    return (_leakage(up, perioddata) - _leakage(down, perioddata)) / (2.0 * eps)


@pytest.mark.parametrize("tsmult", [1.5, 2.0, 5.0, 10.0])
def test_capture_with_tsmult(function_tmpdir, tsmult):
    """A period whose steps grow geometrically reproduces the difference.

    Formed over this step's length rather than the next one's, the reported
    sensitivity was out by 0.13 percent at a multiplier of 1.5 and 1.13
    percent at 10.
    """
    perioddata = [(100.0, 5, tsmult)] * 2
    ws = _build(function_tmpdir / f"ts{tsmult}", perioddata)
    assert np.isclose(
        _adjoint(ws), _finite_difference(function_tmpdir, perioddata), rtol=1.0e-4
    )


def test_capture_across_periods(function_tmpdir):
    """Periods of very different lengths reproduce the difference."""
    perioddata = [(10000.0, 2, 1.0), (1.0, 2, 1.0)]
    ws = _build(function_tmpdir / "periods", perioddata)
    assert np.isclose(
        _adjoint(ws), _finite_difference(function_tmpdir, perioddata), rtol=1.0e-4
    )


def test_constant_timestep(function_tmpdir):
    """Steps of one length, which the carry was already right for."""
    perioddata = [(100.0, 4, 1.0)] * 3
    ws = _build(function_tmpdir / "flat", perioddata)
    assert np.isclose(
        _adjoint(ws), _finite_difference(function_tmpdir, perioddata), rtol=1.0e-4
    )
