"""
Tests for writing and reading a performance measure.

A measure is a set of locations crossed with a set of times, and callers build
it from a dict, a numpy recarray, or a dataframe. The columns are written as
ascii, which is what MODFLOW 6 input is, or as hdf5, which is smaller.

Cases:
  - test_matches_hand_written  : the file matches what the tests write by hand.
  - test_container             : dict, recarray and dataframe give one file.
  - test_times_are_crossed     : `times` crosses with the locations, time
                                 slowest, and `all_times` covers periods that
                                 hold different numbers of time steps.
  - test_per_entry_times       : `kper` and `kstp` are one value per entry.
  - test_value_columns         : a value is a scalar, per location, or per entry.
  - test_discretization        : dis, disv and disu round trip.
  - test_round_trip            : ascii -> hdf5 -> ascii is unchanged.
  - test_hdf5_is_smaller       : the hdf5 file is the smaller of the two,
                                 which a column of names held as variable-
                                 length strings would not be.
  - test_reads_hand_written    : comments, blank lines and mixed case read.
  - test_committed_files       : the measures kept in the repository read.
  - test_refusals              : the input a measure cannot be built from.
  - test_solves_the_same       : a model solved from a written file and from a
                                 hand-written one gives the same sensitivities.
  - test_solves_from_hdf5      : the same measure written as hdf5 is read and
                                 solved, and gives the same sensitivities.
"""

import pathlib as pl
import shutil
import sys

import flopy
import numpy as np
import pandas as pd
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj import all_times, read_performance_measures, write_performance_measures

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

CELLS = [(0, 1, 2), (0, 1, 3), (1, 4, 5)]


def entries(path):
    """Return the entry lines of a written file."""
    return [
        ln.strip()
        for ln in pl.Path(path).read_text().splitlines()
        if ln.strip() and not ln.strip().lower().startswith(("begin", "end"))
    ]


def test_matches_hand_written(function_tmpdir):
    """The written file is what the tests write by hand, one-based."""
    path = write_performance_measures(
        function_tmpdir / "t.adj", {"pm": {"cellid": CELLS, "pm_type": "lak-1"}}
    )
    hand = [f"1 1 {k + 1} {i + 1} {j + 1} lak-1 direct 1.0 -1e+30" for k, i, j in CELLS]
    assert entries(path) == hand


def test_container(function_tmpdir):
    """A dict, a recarray, and a dataframe give the same file."""
    layer, row, column = (np.array(c) for c in zip(*CELLS))
    columns = {
        "layer": layer,
        "row": row,
        "column": column,
        "pm_type": np.array(["ghb-1"] * 3),
        "weight": np.array([1.0, 2.0, 3.0]),
    }
    rec = np.rec.fromarrays(list(columns.values()), names=list(columns))
    frame = pd.DataFrame(columns)

    written = [
        write_performance_measures(
            function_tmpdir / f"{tag}.adj", {"pm": measure}
        ).read_text()
        for tag, measure in (("d", columns), ("r", rec), ("f", frame))
    ]
    assert written[0] == written[1] == written[2]


def test_times_are_crossed(function_tmpdir):
    """`times` crosses with the locations, the time varying slowest."""
    path = write_performance_measures(
        function_tmpdir / "t.adj",
        {"pm": {"cellid": CELLS, "times": range(3), "pm_type": "head"}},
    )
    lines = entries(path)
    assert len(lines) == 9
    assert [ln.split()[0] for ln in lines] == ["1"] * 3 + ["2"] * 3 + ["3"] * 3
    assert [ln.split()[4] for ln in lines] == ["3", "4", "6"] * 3

    # a period holding three time steps, which no cross of two axes gives
    path = write_performance_measures(
        function_tmpdir / "s.adj",
        {"pm": {"cellid": CELLS[:1], "times": all_times([1, 3]), "pm_type": "head"}},
    )
    assert [ln.split()[:2] for ln in entries(path)] == [
        ["1", "1"],
        ["2", "1"],
        ["2", "2"],
        ["2", "3"],
    ]


def test_per_entry_times(function_tmpdir):
    """`kper` and `kstp` are one value per entry rather than an axis."""
    path = write_performance_measures(
        function_tmpdir / "t.adj",
        {
            "pm": {
                "cellid": CELLS,
                "kper": [0, 1, 2],
                "kstp": [0, 0, 1],
                "pm_type": "head",
            }
        },
    )
    lines = entries(path)
    assert len(lines) == 3
    assert [ln.split()[:2] for ln in lines] == [["1", "1"], ["2", "1"], ["3", "2"]]


def test_value_columns(function_tmpdir):
    """A value is a scalar, one per location, or one per entry."""
    measure = {
        "cellid": CELLS,
        "times": range(2),
        "pm_type": "head",
        "weight": [1.0, 2.0, 3.0],  # per location, repeated each time
        "obsval": np.arange(6, dtype=float),  # per entry
    }
    path = write_performance_measures(function_tmpdir / "t.adj", {"pm": measure})
    lines = entries(path)
    assert [ln.split()[-2] for ln in lines] == ["1.0", "2.0", "3.0"] * 2
    assert [ln.split()[-1] for ln in lines] == [f"{v}.0" for v in range(6)]


@pytest.mark.parametrize(
    "cellid, ntoken",
    [([(0, 1, 2), (1, 3, 4)], 9), ([(0, 5), (1, 6)], 8), ([(7,), (8,)], 7)],
)
def test_discretization(function_tmpdir, cellid, ntoken):
    """A cellid of three, two, or one index is dis, disv, or disu."""
    path = write_performance_measures(
        function_tmpdir / "t.adj", {"pm": {"cellid": cellid, "pm_type": "head"}}
    )
    assert all(len(ln.split()) == ntoken for ln in entries(path))

    measures, _ = read_performance_measures(path)
    back = write_performance_measures(function_tmpdir / "b.adj", measures)
    assert back.read_text() == path.read_text()


def test_round_trip(function_tmpdir):
    """A measure written as ascii, read, and written as hdf5 comes back whole."""
    options = {"hdf5_name": "out.h5"}
    measure = {
        "cellid": CELLS,
        "times": range(2),
        "pm_type": "head",
        "pm_form": "residual",
        "weight": [1.0, 2.0, 3.0],
        "obsval": np.arange(6, dtype=float),
    }
    ascii_path = write_performance_measures(
        function_tmpdir / "t.adj", {"pm": measure}, options=options
    )

    first, first_options = read_performance_measures(ascii_path)
    h5 = write_performance_measures(
        function_tmpdir / "t.h5", first, options=first_options, format="hdf5"
    )
    assert mf6adj.utils.utils_pm_write.is_hdf5(h5)

    second, second_options = read_performance_measures(h5)
    again = write_performance_measures(
        function_tmpdir / "again.adj", second, options=second_options
    )
    assert again.read_text() == ascii_path.read_text()
    assert dict(second_options) == options
    for name in first:
        for column in first[name]:
            assert np.array_equal(first[name][column], second[name][column]), column


HAND_WRITTEN = """\
# a measure written by hand, as the tests write them

begin options
  hdf5_name out.h5
end options

BEGIN PERFORMANCE_MEASURE Head
  1 1 3 5 2 head direct 1.0 -1.0e+30
  2 1 3 5 2 head residual 2.5 34.125

end performance_measure

begin performance_measure sfr
  # a comment inside a block
  2 1 2 4 1 sfr_1 direct 1.0 -1.0e+30
end performance_measure
"""


def test_hdf5_is_smaller(function_tmpdir):
    """hdf5 is the smaller format, which is the reason to offer it.

    A column of package names held as variable-length strings carries a pointer
    and a heap entry for every entry and does not compress, which made the hdf5
    file larger than the ascii one it replaces.
    """
    nentry = 20_000
    rng = np.random.default_rng(0)
    measure = {
        "layer": rng.integers(0, 4, nentry),
        "row": rng.integers(0, 800, nentry),
        "column": rng.integers(0, 1000, nentry),
        "pm_type": np.where(rng.random(nentry) < 0.5, "drn-1", "ghb-1"),
    }
    ascii_path = write_performance_measures(function_tmpdir / "m.adj", {"m": measure})
    h5_path = write_performance_measures(
        function_tmpdir / "m.h5", {"m": measure}, format="hdf5"
    )

    ascii_size = ascii_path.stat().st_size
    h5_size = h5_path.stat().st_size
    assert h5_size < ascii_size / 3, f"hdf5 {h5_size} against ascii {ascii_size}"


def test_reads_hand_written(function_tmpdir):
    """A file with comments, blank lines, and mixed case reads as written."""
    path = function_tmpdir / "hand.adj"
    path.write_text(HAND_WRITTEN)

    measures, options = read_performance_measures(path)
    assert options == {"hdf5_name": "out.h5"}
    assert sorted(measures) == ["head", "sfr"]

    head = measures["head"]
    assert np.array_equal(head["kper"], [0, 1])
    assert np.array_equal(head["layer"], [2, 2])
    assert np.array_equal(head["row"], [4, 4])
    assert np.array_equal(head["column"], [1, 1])
    assert list(head["pm_form"]) == ["direct", "residual"]
    assert np.array_equal(head["weight"], [1.0, 2.5])
    assert np.array_equal(head["obsval"], [-1.0e30, 34.125])
    assert list(measures["sfr"]["pm_type"]) == ["sfr_1"]

    # and it writes back to the same entries it was read from
    again = write_performance_measures(
        function_tmpdir / "again.adj", measures, options=options
    )
    back, back_options = read_performance_measures(again)
    assert back_options == options
    for name in measures:
        for column in measures[name]:
            assert np.array_equal(measures[name][column], back[name][column]), column


def test_committed_files(function_tmpdir):
    """The measures kept in the repository read and write back unchanged."""
    root = pl.Path(__file__).parent.parent
    paths = [
        root / "autotest/ie_1sp/ie_perfmeas.dat",
        root / "examples/xd_box_chd_ana/test.adj",
    ]

    for path in paths:
        if not path.is_file():
            continue
        measures, options = read_performance_measures(path)
        assert measures, path
        again = write_performance_measures(
            function_tmpdir / "rt.adj", measures, options=options
        )
        back, _ = read_performance_measures(again)
        assert measures.keys() == back.keys(), path
        for name in measures:
            for column in measures[name]:
                assert np.array_equal(measures[name][column], back[name][column]), (
                    f"{path} {name} {column}"
                )


def test_refusals(function_tmpdir):
    """Input a measure cannot be built from is refused where it is given."""
    path = function_tmpdir / "t.adj"

    with pytest.raises(Exception, match="pm_type"):
        write_performance_measures(path, {"pm": {"cellid": CELLS}})

    with pytest.raises(Exception, match="locations"):
        write_performance_measures(path, {"pm": {"row": [1], "pm_type": "head"}})

    with pytest.raises(Exception, match="not as both"):
        write_performance_measures(
            path,
            {"pm": {"cellid": CELLS, "times": range(2), "kper": 1, "pm_type": "head"}},
        )

    with pytest.raises(Exception, match="do not broadcast"):
        write_performance_measures(
            path,
            {"pm": {"cellid": CELLS, "times": (range(3), range(2)), "pm_type": "head"}},
        )

    with pytest.raises(Exception, match="weight"):
        write_performance_measures(
            path, {"pm": {"cellid": CELLS, "pm_type": "head", "weight": [1.0, 2.0]}}
        )

    with pytest.raises(Exception, match="unrecognized"):
        write_performance_measures(
            path, {"pm": {"cellid": CELLS, "pm_type": "head", "nonsense": 1}}
        )

    with pytest.raises(Exception, match="ascii"):
        write_performance_measures(
            path, {"pm": {"cellid": CELLS, "pm_type": "head"}}, format="parquet"
        )

    with pytest.raises(Exception, match="no performance measures"):
        write_performance_measures(path, {})


def _build(ws):
    """A small confined model with a general-head boundary along one edge."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name="pmw", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=2, perioddata=[(1.0, 1, 1.0)] * 2)
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="pmw", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=2, nrow=5, ncol=5, delr=10.0, delc=10.0, top=10.0, botm=[5.0, 0.0]
    )
    flopy.mf6.ModflowGwfic(gwf, strt=8.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0, k33=1.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, steady_state={0: True})
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), 9.0, 100.0] for i in range(5)],
        pname="ghb-1",
    )
    flopy.mf6.ModflowGwfchd(
        gwf, stress_period_data=[[(0, i, 4), 6.0] for i in range(5)], pname="chd-1"
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="pmw.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])
    return ws


def _solve(ws):
    """Return the sensitivities of the measure in a workspace."""
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    df = adj.solve_adjoint()
    adj.finalize()
    return df


def test_solves_the_same(function_tmpdir):
    """A written measure gives the sensitivities a hand-written one gives."""
    cells = [(0, i, 0) for i in range(5)]

    hand_ws = _build(function_tmpdir / "hand")
    with open(hand_ws / "pm.dat", "w") as f:
        f.write("begin performance_measure ghb\n")
        for kper in range(2):
            for k, i, j in cells:
                f.write(
                    f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} ghb-1 direct 1.0 -1.0e+30\n"
                )
        f.write("end performance_measure\n\n")

    written_ws = _build(function_tmpdir / "written")
    write_performance_measures(
        written_ws / "pm.dat",
        {"ghb": {"cellid": cells, "times": range(2), "pm_type": "ghb-1"}},
    )

    hand = _solve(hand_ws)
    written = _solve(written_ws)

    assert hand.keys() == written.keys()
    for name in hand:
        pd.testing.assert_frame_equal(hand[name], written[name])


def test_solves_from_hdf5(function_tmpdir):
    """A measure written as hdf5 is read and solved like the ascii one."""
    cells = [(0, i, 0) for i in range(5)]
    measures = {"ghb": {"cellid": cells, "times": range(2), "pm_type": "ghb-1"}}

    ascii_ws = _build(function_tmpdir / "ascii")
    write_performance_measures(ascii_ws / "pm.dat", measures)

    hdf5_ws = _build(function_tmpdir / "hdf5")
    write_performance_measures(hdf5_ws / "pm.dat", measures, format="hdf5")

    from_ascii = _solve(ascii_ws)
    from_hdf5 = _solve(hdf5_ws)

    assert from_ascii.keys() == from_hdf5.keys()
    for name in from_ascii:
        pd.testing.assert_frame_equal(from_ascii[name], from_hdf5[name])
