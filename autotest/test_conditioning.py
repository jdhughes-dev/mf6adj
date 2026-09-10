"""
Tests the report on an adjoint matrix that cannot carry a sensitivity.

The adjoint is solved against the matrix MODFLOW 6 assembled. A row whose
diagonal has fallen away still solves, and the state it returns goes as one
over that diagonal, so a measure can come back with values far larger than the
flow model supports and nothing says why.

The checks are the ones affordable at model scale: the diagonal, which is one
pass, and the residual of the solve, which is one matrix-vector product.

Cases:
  - test_sound_matrix_is_quiet   : a well formed matrix reports nothing.
  - test_row_with_no_diagonal    : a singular row is counted and named.
  - test_small_diagonal          : a row far below the rest is counted.
  - test_worst_rows_are_named    : the report names the smallest diagonals.
  - test_residual_is_relative    : the residual is scaled by the right side.
  - test_healthy_model_is_quiet  : a real solve warns about nothing.
  - test_isolated_cell_is_reported : a solve whose matrix holds a row far
                                   below the rest names the cell.
  - test_dry_cells_are_not_reported : cells that go dry under the
                                   Newton-Raphson formulation are not such
                                   rows, and are not reported.
  - test_a_matrix_with_no_solution_stops_the_run : a solve that returns
                                   values which are not numbers stops the run.
  - test_a_solve_that_stops_short_is_reported : a solve the accelerator did
                                   not finish says how far it got.
  - test_a_measure_that_stops_leaves_the_rest : a measure that reached such a
                                   step is named at the end of the run, and
                                   the measures beside it are still solved.
  - test_every_flagged_node_reaches_the_log : the console names the worst few
                                   rows and the log file holds them all.
  - test_a_row_is_named_as_the_model_names_it : a row of the matrix is
                                   reported as the cell it came from.
"""

import pathlib as pl
import shutil
import sys

import flopy
import h5py
import numpy as np
import pytest
import scipy.sparse as sps

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.pm import AdjointSolveError, PerfMeas
from mf6adj.utils.utils_conditioning import (
    cellid,
    describe,
    diagonal_report,
    flagged_nodes,
    solve_residual,
)

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()


def _matrix(diagonal):
    """A matrix with the given diagonal and a little off-diagonal weight."""
    n = len(diagonal)
    mat = sps.lil_matrix((n, n))
    for i, d in enumerate(diagonal):
        mat[i, i] = d
        if i + 1 < n:
            mat[i, i + 1] = -1.0
    return mat.tocsr()


def test_sound_matrix_is_quiet():
    """Every row supporting a solution reports nothing."""
    assert diagonal_report(_matrix([10.0, 11.0, 9.5, 10.5])) is None
    # a spread of two orders is still sound
    assert diagonal_report(_matrix([100.0, 10.0, 1.0])) is None


def test_row_with_no_diagonal():
    """A singular row is counted."""
    report = diagonal_report(_matrix([10.0, 0.0, 10.0]))
    assert report is not None
    assert report["nzero"] == 1
    assert report["nsmall"] == 0
    assert 1 in [int(r) for r in report["rows"]]


def test_small_diagonal():
    """A row far below the rest of the model is counted, and is not singular."""
    report = diagonal_report(_matrix([10.0, 10.0, 1.0e-9, 10.0]))
    assert report is not None
    assert report["nzero"] == 0
    assert report["nsmall"] == 1
    assert np.isclose(report["median"], 10.0)


def test_worst_rows_are_named():
    """The message names the smallest diagonals, smallest first."""
    report = diagonal_report(_matrix([10.0, 1.0e-12, 10.0, 1.0e-9, 10.0]))
    assert [int(r) for r in report["rows"]][:2] == [1, 3]
    message = describe(report, kper=2, kstp=3)
    assert "stress period 3" in message and "time step 4" in message
    assert "1" in message


def test_residual_is_relative():
    """The residual is scaled by the right-hand side it was solved against."""
    mat = _matrix([2.0, 2.0, 2.0])
    rhs = np.array([2.0, 2.0, 2.0])
    exact = sps.linalg.spsolve(mat.tocsc(), rhs)
    assert solve_residual(mat, exact, rhs) < 1.0e-12

    # a solution that is out by one unit against a right side of two
    assert np.isclose(solve_residual(mat, np.zeros(3), rhs), 1.0)


def test_healthy_model_is_quiet(function_tmpdir, caplog):
    """A model whose cells stay wet warns about neither rows nor residual."""
    ws = pl.Path(function_tmpdir) / "quiet"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="q", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=2, perioddata=[(10.0, 2, 1.0)] * 2)
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="q", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=8, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=9.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, transient={0: True})
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 9.0], [(0, 0, 7), 8.0]], pname="chd-1"
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="q.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write("  2 2 1 1 4 head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")

    with caplog.at_level("WARNING"):
        adj = mf6adj.Mf6Adj(
            "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
        )
        adj.solve_forward_model(hdf5_name="fwd.hd5")
        adj.solve_adjoint()
        adj.finalize()

    noise = [
        r.message
        for r in caplog.records
        if "diagonal" in r.message or "residual" in r.message
    ]
    assert not noise, f"a sound model reported {noise}"


def test_isolated_cell_is_reported(function_tmpdir, caplog):
    """A cell the model barely connects to is named, with its diagonal.

    Its conductances set the diagonal of its row, so a conductivity far below
    the rest of the model leaves a row whose state goes as one over almost
    nothing. The flow model solves without trouble, which is the point: there
    is nothing else to notice it by.
    """
    ws = pl.Path(function_tmpdir) / "isolated"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    k = np.full((1, 1, 10), 10.0)
    k[0, 0, 5] = 1.0e-11

    sim = flopy.mf6.MFSimulation(sim_name="iso", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple", outer_maximum=200)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="iso", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=10, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=9.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, steady_state={0: True})
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 9.0], [(0, 0, 9), 8.0]], pname="chd-1"
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="iso.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "the flow model is expected to solve\n" + "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write("  1 1 1 1 3 head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")

    with caplog.at_level("WARNING"):
        adj = mf6adj.Mf6Adj(
            "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
        )
        adj.solve_forward_model(hdf5_name="fwd.hd5")
        adj.solve_adjoint()
        adj.finalize()

    reported = [r.message for r in caplog.records if "diagonal" in r.message]
    assert reported, "the row far below the rest was not reported"
    assert "5" in reported[0], reported[0]


def test_dry_cells_are_not_reported(function_tmpdir, caplog):
    """A cell that goes dry is not a row that has fallen away.

    Drying was the reason first given for this report and it is the wrong one.
    A drain moved from the upper layer into the lower one empties the upper
    layer, and MODFLOW 6 gives a dry cell a diagonal of its own rather than
    letting it fall to nothing. The sensitivities stay small, so nothing here
    is reported, and a large sensitivity on such a model has another cause.
    """
    ws = pl.Path(function_tmpdir) / "dry"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    ncol, top, mid, bot, chd = 8, 20.0, 10.0, 0.0, 18.0
    # the drain starts high in the upper layer and finishes low in the lower one
    schedule = [
        (0, 18.0),
        (0, 15.0),
        (0, 12.0),
        (0, 10.5),
        (1, 9.0),
        (1, 5.0),
        (1, 1.0),
        (1, 0.05),
    ]

    sim = flopy.mf6.MFSimulation(sim_name="dry", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(
        sim, nper=len(schedule), perioddata=[(5000.0, 2, 1.0)] * len(schedule)
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        outer_maximum=1000,
        inner_maximum=500,
        linear_acceleration="BICGSTAB",
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname="dry",
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=2,
        nrow=1,
        ncol=ncol,
        delr=100.0,
        delc=100.0,
        top=top,
        botm=[mid, bot],
    )
    flopy.mf6.ModflowGwfic(gwf, strt=chd)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0, k33=1.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1.0e-5, sy=0.2, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data={
            p: [[(lay, 0, 0), elev, 100.0]] for p, (lay, elev) in enumerate(schedule)
        },
        pname="drn-1",
    )
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(1, 0, ncol - 1), chd]], pname="chd-1"
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="dry.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    heads = flopy.utils.HeadFile(str(ws / "dry.hds")).get_alldata()
    upper = np.clip((heads[-1, 0, 0, :] - mid) / (top - mid), 0.0, 1.0)
    assert (upper == 0.0).any(), "the upper layer is expected to go dry"

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  {len(schedule)} 2 2 1 4 head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")

    with caplog.at_level("WARNING"):
        adj = mf6adj.Mf6Adj(
            "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
        )
        adj.solve_forward_model(hdf5_name="fwd.hd5")
        results = adj.solve_adjoint()
        adj.finalize()

    reported = [
        r.message
        for r in caplog.records
        if "diagonal" in r.message or "residual" in r.message
    ]
    assert not reported, f"a model with dry cells reported {reported}"
    assert np.nanmax(np.abs(results["obs"]["k11"].to_numpy())) < 1.0


def _forward_file(ws):
    """Return the forward file of a small transient model."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="sing", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0, 2, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="sing", save_flows=True)
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
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord="sing.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write("  1 2 1 1 4 head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return ws


def test_a_matrix_with_no_solution_stops_the_run(function_tmpdir):
    """A solve that returns values which are not numbers stops the run.

    No model this size leaves a matrix with no solution, so the forward file of
    one is edited into it: a row is emptied, which is what a cell the model has
    taken out of the solution would leave. The solve then returns values that
    are not numbers, and every comparison against those is false, so a check on
    the size of the residual passes over them without tripping. The state is
    carried into the right side of every earlier step, so nothing later in the
    recursion returns a number either and the run is stopped.
    """
    ws = _forward_file(function_tmpdir / "run")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")

    shutil.copy(ws / "fwd.hd5", ws / "sing.hd5")
    with h5py.File(ws / "sing.hd5", "r+") as f:
        ia = np.asarray(f["gwf_info"]["sln_ia"][:]).astype(int)
        if ia.min() == 1:
            ia = ia - 1
        for key in [k for k in f if k.startswith("solution_kper")]:
            amat = np.asarray(f[key]["amat"][:])
            amat[ia[3] : ia[4]] = 0.0
            f[key]["amat"][...] = amat

    with pytest.raises(Exception, match="values that are not numbers"):
        adj._performance_measures[0].solve_adjoint(
            ws / "sing.hd5", hdf5_adjoint_solution_fname=str(ws / "sing_adj.hd5")
        )
    adj.finalize()

    # the step that stopped the run is written, so the cells holding no number
    # can be read from it, and the report of it survives in the log file
    with h5py.File(ws / "sing_adj.hd5", "r") as f:
        keys = sorted(k for k in f if k.startswith("solution_kper"))
        assert keys, "the step that stopped the run was not written"
        state = np.asarray(f[keys[-1]]["lambda"][:])
    assert np.isnan(state).any(), "no cell of the state written is without a number"
    assert "values that are not numbers" in (ws / "pm.log").read_text()


def test_a_solve_that_stops_short_is_reported(function_tmpdir, caplog):
    """A solve the accelerator did not finish says how far it got.

    The direct solver solves exactly, so the residual reaches the threshold
    only where an iterative solver stops short. A preconditioner finishes this
    model in one iteration, so the solve is run without one.
    """
    ws = _forward_file(function_tmpdir / "run")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")

    with caplog.at_level("WARNING"):
        adj._performance_measures[0].solve_adjoint(
            ws / "fwd.hd5",
            hdf5_adjoint_solution_fname=str(ws / "short_adj.hd5"),
            linear_solver="bicgstab",
            linear_solver_kwargs={"maxiter": 1},
            use_precon=False,
            dvclose=None,
            rclose=None,
        )
    adj.finalize()

    reported = [r.message for r in caplog.records if "left a residual" in r.message]
    assert reported, "a solve that stopped short was not reported"


def test_a_measure_that_stops_leaves_the_rest(function_tmpdir, caplog, monkeypatch):
    """A measure that stopped is named at the end, and the rest are solved.

    The measures of a run share one matrix, so a step with no solution ends
    every one of them. One is stopped at the call instead, which is where the
    solve reports it, leaving the others to be solved as they would be.
    """
    ws = _forward_file(function_tmpdir / "run")
    with open(ws / "pm.dat", "w") as f:
        for name in ("obs", "gone"):
            f.write(f"begin performance_measure {name}\n")
            f.write("  1 2 1 1 4 head direct 1.0 -1.0e+30\n")
            f.write("end performance_measure\n")

    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")

    solve_adjoint = PerfMeas.solve_adjoint

    def stop_one(self, *args, **kwargs):
        if self.name == "gone":
            raise AdjointSolveError("values that are not numbers", 1, 2)
        return solve_adjoint(self, *args, **kwargs)

    monkeypatch.setattr(PerfMeas, "solve_adjoint", stop_one)

    with caplog.at_level("ERROR"):
        dfs = adj.solve_adjoint()
    adj.finalize()

    assert "obs" in dfs, "a measure beside the one that stopped was not solved"
    assert "gone" not in dfs, "a measure that stopped came back with a summary"

    reported = [r.message for r in caplog.records if "were not solved" in r.message]
    assert reported, "a measure that stopped was not named at the end of the run"
    assert "gone (stress period 1, time step 2)" in reported[0]


def test_every_flagged_node_reaches_the_log(function_tmpdir, caplog):
    """The console names the worst few rows, and the log file holds them all.

    A model of a few million nodes can flag more rows than a console holds, so
    the whole list is written where it is read later rather than watched.
    """
    diagonal = [10.0] * 20
    for node in (3, 7, 11, 13, 15, 17, 19):
        diagonal[node] = 0.0
    report = diagonal_report(_matrix(diagonal))
    assert report is not None
    assert report["nzero"] == 7

    # the console names the worst five of them
    named = describe(report, 0, 0)
    assert len(report["rows"]) == 5

    # the log file holds every one, and names the cell the model does
    listed = flagged_nodes(report, 0, 0, grid_shape=(1, 4, 5))
    for node in (3, 7, 11, 13, 15, 17, 19):
        row, column = divmod(node, 5)
        assert f"layer 1, row {row + 1}, column {column + 1}," in listed
    assert named.count(",") < listed.count(",")


def test_the_flagged_nodes_are_not_written_to_the_console(function_tmpdir, capsys):
    """The list of flagged nodes goes to the log file and not the console."""
    ws = _forward_file(function_tmpdir / "run")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")

    shutil.copy(ws / "fwd.hd5", ws / "sing.hd5")
    with h5py.File(ws / "sing.hd5", "r+") as f:
        ia = np.asarray(f["gwf_info"]["sln_ia"][:]).astype(int)
        if ia.min() == 1:
            ia = ia - 1
        for key in [k for k in f if k.startswith("solution_kper")]:
            amat = np.asarray(f[key]["amat"][:])
            amat[ia[3] : ia[4]] = 0.0
            f[key]["amat"][...] = amat

    with pytest.raises(Exception, match="values that are not numbers"):
        adj._performance_measures[0].solve_adjoint(
            ws / "sing.hd5", hdf5_adjoint_solution_fname=str(ws / "sing_adj.hd5")
        )
    adj.finalize()

    written = (ws / "pm.log").read_text()
    console = capsys.readouterr().err
    assert "row 1, column 4" in written, "the log file holds no list of cells"
    assert "diagonal 0.000000e+00" not in console
    # the console still carries the report naming the worst few, which is
    # what makes the check above a check
    assert "Worst cells" in console, "the console carried no report at all"


def test_a_row_is_named_as_the_model_names_it():
    """A row of the matrix is reported as the cell the model names.

    The matrix is assembled over the nodes left after the model drops the
    cells it does not solve, so a row of it is not a cell of the grid the user
    wrote.
    """
    # a structured grid is a layer, a row and a column
    assert cellid(3, None, (1, 1, 10)) == "layer 1, row 1, column 4"
    assert cellid(13, None, (2, 3, 5)) == "layer 1, row 3, column 4"

    # a grid of vertices is a layer and the cell within it
    assert cellid(13, None, (2, 10)) == "layer 2, cell 4"

    # a grid of neither is named by its node
    assert cellid(13, None, None) == "node 14"

    # a row of a reduced matrix is carried back to the cell it came from
    nodeuser = np.array([0, 5, 13])
    assert cellid(2, nodeuser, (2, 10)) == "layer 2, cell 4"
    assert cellid(2, nodeuser, None) == "node 14"
