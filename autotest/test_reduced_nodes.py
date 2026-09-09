"""
Tests for mapping a performance measure onto a reduced node numbering.

MODFLOW 6 omits inactive cells from its internal node numbering, so a measure
given in grid indices has to be mapped onto the nodes the model actually holds.
The map is built once for a file; searching NODEUSER for each entry is one pass
over the model per entry, which is minutes on a model of a few million cells.

Cases:
  - test_map_matches_the_scan   : the map gives what searching NODEUSER gives.
  - test_no_reduction           : a model that omits nothing maps to itself.
  - test_node_outside_the_map   : a node the model does not hold is reported.
  - test_solves_on_a_reduced_model : a measure on a model with inactive cells
                                  reproduces a perturbation of the same model.
  - test_measure_on_an_inactive_cell : a measure on an omitted cell is refused.
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

from mf6adj.utils.utils_pm_read import (
    build_reduced_map,
    map_reduced_node,
    map_reduced_nodes,
)

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NAME = "reduced"
NLAY, NROW, NCOL = 1, 5, 5
INACTIVE = [(0, 1, 1), (0, 2, 2)]
OBS = (0, 3, 3)


def scan(inode, nuser):
    """The mapping as it was done before, one pass over the model per entry."""
    return int(np.where(nuser == inode)[0][0])


def test_map_matches_the_scan():
    """The map gives, for every node, what searching NODEUSER gives."""
    rng = np.random.default_rng(0)
    nuser = np.sort(rng.choice(5_000, size=4_000, replace=False))

    reduced = build_reduced_map(nuser)
    assert reduced is not None

    for inode in nuser[:: len(nuser) // 50]:
        assert map_reduced_node(int(inode), reduced) == scan(int(inode), nuser)

    assert np.array_equal(map_reduced_nodes(nuser, reduced), np.arange(nuser.size))
    picks = rng.choice(nuser, size=200, replace=False)
    assert np.array_equal(
        map_reduced_nodes(picks, reduced), [scan(int(n), nuser) for n in picks]
    )


def test_no_reduction():
    """A model that omits nothing has no map, and a node is already its own."""
    assert build_reduced_map(np.array([0])) is None
    assert map_reduced_node(17, None) == 17
    assert np.array_equal(map_reduced_nodes(np.arange(5), None), np.arange(5))


def test_node_outside_the_map():
    """A node the model does not hold is reported rather than mapped."""
    nuser = np.array([0, 1, 3, 4])
    reduced = build_reduced_map(nuser)

    assert map_reduced_node(3, reduced) == 2
    with pytest.raises(Exception, match="not in reduced node num"):
        map_reduced_node(2, reduced)
    with pytest.raises(Exception, match="not in reduced node num"):
        map_reduced_node(99, reduced)
    with pytest.raises(Exception, match="not in reduced node num"):
        map_reduced_nodes(np.array([0, 2]), reduced)
    with pytest.raises(Exception, match="not in reduced node num"):
        map_reduced_nodes(np.array([0, 99]), reduced)


def _build(ws, cell):
    """A confined model with two inactive cells, measured at one cell."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    idomain = np.ones((NLAY, NROW, NCOL), dtype=int)
    for k, i, j in INACTIVE:
        idomain[k, i, j] = 0

    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=10.0,
        delc=10.0,
        top=10.0,
        botm=0.0,
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=[[(0, i, 0), 6.0] for i in range(NROW)]
        + [[(0, i, NCOL - 1), 4.0] for i in range(NROW)],
        pname="chd-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    k, i, j = cell
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    return ws


def test_solves_on_a_reduced_model(function_tmpdir):
    """A measure on a model with inactive cells matches a perturbation of it.

    The perturbation reruns the model for each parameter and does not use the
    map, so agreeing with it is what shows the entry was mapped onto the node
    the model actually solved.
    """
    ws = _build(function_tmpdir / "solve", OBS)

    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    df = adj.solve_adjoint()
    pert = adj.perturbation_method(pert_mult=1.0001)
    adj.finalize()

    # the results are in the model's own node space, which omits the two cells
    nactive = NLAY * NROW * NCOL - len(INACTIVE)
    assert len(df["obs"]) == nactive
    assert df["obs"]["k11"].notna().all()

    omitted = {(i, j) for _, i, j in INACTIVE}
    assert not omitted & set(zip(pert["i"], pert["j"]))

    adjoint = df["obs"]["k11"].to_numpy()
    finite = pert.loc[pert.addr == "k11_reduced_npf", "obs"].to_numpy()
    assert finite.size == nactive

    # a forward difference carries its own truncation error, so the comparison
    # is loose where the sensitivity is not near zero
    resolved = np.abs(adjoint) > 1.0e-8
    assert resolved.sum() > nactive // 2
    assert np.allclose(adjoint[resolved], finite[resolved], rtol=0.01)
    assert np.allclose(adjoint[~resolved], finite[~resolved], atol=1.0e-6)


def test_measure_on_an_inactive_cell(function_tmpdir):
    """A measure on a cell the model omits is refused rather than mapped."""
    ws = _build(function_tmpdir / "inactive", INACTIVE[0])
    with pytest.raises(Exception, match="not in reduced node num"):
        mf6adj.Mf6Adj(
            "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
        )
