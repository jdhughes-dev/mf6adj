"""Write and read the performance measures of an adjoint input file.

A measure is a set of locations crossed with a set of times, which is the shape
every caller builds by hand. Columns are given as a dict, a numpy recarray, or a
dataframe, and are held as one-dimensional arrays, which is the fastest form for
both the ascii and the hdf5 target.

Indices are zero-based here, as they are elsewhere in mf6adj and in flopy. The
file is one-based, and the conversion happens on the way out and back.
"""

import pathlib as pl
from typing import Any, Iterable, Optional

import numpy as np

PathLike = str | pl.Path

# the location columns of each discretization, in the order the file writes them
LOCATION_COLUMNS = {
    "dis": ("layer", "row", "column"),
    "disv": ("layer", "cell2d"),
    "disu": ("node",),
}

# everything a measure carries that is not a location or a time
VALUE_COLUMNS = ("pm_type", "pm_form", "weight", "obsval")

TIME_COLUMNS = ("kper", "kstp")

_DEFAULTS = {"pm_form": "direct", "weight": 1.0, "obsval": -1.0e30}


def all_times(nstp: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return every ``(kper, kstp)`` pair for a model's time steps per period.

    Periods do not have to hold the same number of time steps, so the pairs
    cannot be formed by crossing one with the other.

    Parameters
    ----------
    nstp : iterable of int
        Number of time steps in each stress period.

    Returns
    -------
    tuple[ndarray, ndarray]
        Zero-based stress periods and time steps, of equal length.
    """
    nstp = np.asarray(list(nstp), dtype=np.int64)
    kper = np.repeat(np.arange(nstp.size, dtype=np.int64), nstp)
    kstp = np.concatenate(
        [np.arange(n, dtype=np.int64) for n in nstp]
        if nstp.size
        else [np.zeros(0, dtype=np.int64)]
    )
    return kper, kstp


def as_columns(measure: Any) -> dict[str, Any]:
    """Return a measure given as a dict, recarray, or dataframe as a dict.

    Parameters
    ----------
    measure : dict, ndarray, or DataFrame
        Measure columns in any of the accepted containers.

    Returns
    -------
    dict
        Column name to value. A recarray field is a view, not a copy.
    """
    dtype = getattr(measure, "dtype", None)
    if dtype is not None and dtype.names:
        return {name: measure[name] for name in dtype.names}
    columns = getattr(measure, "columns", None)
    if columns is not None:
        return {str(name): measure[name].to_numpy() for name in columns}
    return dict(measure)


def split_cellid(cellid) -> tuple[str, dict[str, np.ndarray]]:
    """Return the discretization and location columns of a flopy cellid.

    Parameters
    ----------
    cellid : sequence
        Zero-based cell identifiers, each a tuple whose length gives the
        discretization: three for `dis`, two for `disv`, and one for `disu`.

    Returns
    -------
    tuple[str, dict[str, ndarray]]
        Discretization name and its location columns.
    """
    values = np.asarray(cellid)
    if values.dtype == object or values.ndim == 1:
        values = np.array([tuple(np.atleast_1d(c)) for c in values], dtype=np.int64)
    values = np.atleast_2d(values)

    ndim = values.shape[1]
    grid = {3: "dis", 2: "disv", 1: "disu"}.get(ndim)
    if grid is None:
        raise Exception(
            f"a cellid holds one, two, or three indices, not {ndim}. A cellid "
            "of three is dis, two is disv, and one is disu."
        )
    return grid, {
        name: values[:, idx].astype(np.int64)
        for idx, name in enumerate(LOCATION_COLUMNS[grid])
    }


def _locations(columns: dict[str, Any]) -> tuple[str, dict[str, np.ndarray]]:
    """Return the discretization and location columns of one measure."""
    if "cellid" in columns:
        return split_cellid(columns.pop("cellid"))

    for grid, names in LOCATION_COLUMNS.items():
        if all(name in columns for name in names):
            return grid, {
                name: np.asarray(columns.pop(name), dtype=np.int64).ravel()
                for name in names
            }

    raise Exception(
        "a measure gives its locations as `cellid`, or as "
        + ", or as ".join(
            "`" + "`, `".join(names) + "`" for names in LOCATION_COLUMNS.values()
        )
        + f". Found {sorted(columns)}."
    )


def _times(columns, nloc):
    """Return the ``(kper, kstp)`` of every entry, and how many entries there are.

    ``times`` is an axis and is crossed with the locations. ``kper`` and
    ``kstp`` are not: they are a scalar, or one value per entry.
    """
    if "times" in columns:
        if "kper" in columns or "kstp" in columns:
            raise Exception(
                "a measure gives its times as `times`, which is crossed with "
                "the locations, or as `kper` and `kstp`, which are one value "
                "per entry, but not as both"
            )
        times = columns.pop("times")
        if isinstance(times, tuple) and len(times) == 2:
            kper, kstp = times
        else:
            values = np.asarray(times, dtype=np.int64)
            if values.ndim == 2 and values.shape[1] == 2:
                kper, kstp = values[:, 0], values[:, 1]
            else:
                kper, kstp = values.ravel(), 0
        kper = np.atleast_1d(np.asarray(kper, dtype=np.int64)).ravel()
        kstp = np.atleast_1d(np.asarray(kstp, dtype=np.int64)).ravel()
        try:
            kper, kstp = np.broadcast_arrays(kper, kstp)
        except ValueError:
            raise Exception(
                f"`times` holds {kper.size} periods and {kstp.size} time steps, "
                "which do not broadcast. `all_times` builds the pairs for a "
                "model whose periods hold different numbers of time steps."
            ) from None
        ntime = kper.size
        at_time = np.repeat(np.arange(ntime), nloc)
        at_loc = np.tile(np.arange(nloc), ntime)
        return kper[at_time], kstp[at_time], ntime * nloc, at_loc

    # one value per entry, so the locations already carry every entry
    kper = _expand("kper", columns.pop("kper", 0), nloc, nloc, np.arange(nloc))
    kstp = _expand("kstp", columns.pop("kstp", 0), nloc, nloc, np.arange(nloc))
    return kper, kstp, nloc, np.arange(nloc)


def normalize(measure: Any) -> tuple[str, dict[str, np.ndarray]]:
    """Return one measure as equal-length columns, one row per entry.

    Times are given one of two ways. ``times`` is an axis, crossed with the
    locations, the time varying slowest; it is a sequence of periods, or a
    ``(kper, kstp)`` pair, which `all_times` builds for a model whose periods
    hold different numbers of time steps. ``kper`` and ``kstp`` are instead one
    value per entry, or a scalar, and are what `read_performance_measures`
    returns, so a measure read from a file writes back unchanged.

    A value column is a scalar, one value per location, or one value per entry.

    Parameters
    ----------
    measure : dict, ndarray, or DataFrame
        Location, time, and value columns of a measure.

    Returns
    -------
    tuple[str, dict[str, ndarray]]
        Discretization name and its columns, each of length ``nentry``.
    """
    columns = as_columns(measure)
    grid, location = _locations(columns)

    nloc = len(next(iter(location.values())))
    for name, values in location.items():
        if len(values) != nloc:
            raise Exception(
                f"location column `{name}` has {len(values)} values, and "
                f"the measure has {nloc} locations"
            )

    if "pm_type" not in columns:
        raise Exception("a measure needs a `pm_type`, either `head` or a package name")

    kper, kstp, nentry, at_loc = _times(columns, nloc)

    out = {name: values[at_loc] for name, values in location.items()}
    out["kper"] = kper
    out["kstp"] = kstp

    for name in VALUE_COLUMNS:
        value = columns.pop(name, _DEFAULTS.get(name))
        out[name] = _expand(name, value, nloc, nentry, at_loc)

    if columns:
        raise Exception(f"unrecognized columns in a measure: {sorted(columns)}")

    return grid, out


def _expand(name, value, nloc, nentry, at_loc) -> np.ndarray:
    """Return one value column as one value per entry."""
    values = np.asarray(value)
    if values.ndim == 0:
        return np.repeat(values, nentry)
    values = values.ravel()
    if values.size == nentry:
        return values
    # a per-location value is repeated for every time the measure is taken;
    # with one time the two lengths are the same and either reading holds
    if values.size == nloc:
        return values[at_loc]
    raise Exception(
        f"column `{name}` has {values.size} values, and the measure has "
        f"{nloc} locations and {nentry} entries"
    )


def _ascii_lines(grid: str, columns: dict[str, np.ndarray]):
    """Yield the entry lines of one measure, one-based as the file is."""
    names = (*TIME_COLUMNS, *LOCATION_COLUMNS[grid])
    template = "  " + " ".join(["%d"] * len(names)) + " %s %s %r %r\n"

    # tolist gives python scalars, so %r writes a float that reads back
    # exactly and a string without the numpy wrapper
    values = [(columns[name] + 1).tolist() for name in names]
    values += [columns[name].tolist() for name in VALUE_COLUMNS]
    for row in zip(*values):
        yield template % row


def _write_ascii(path, measures, options) -> None:
    """Write measures as an adjoint input file."""
    with open(path, "w") as f:
        if options:
            f.write("begin options\n")
            for key, value in options.items():
                f.write(f"  {key} {value}\n")
            f.write("end options\n\n")
        for name, (grid, columns) in measures.items():
            f.write(f"begin performance_measure {name}\n")
            f.writelines(_ascii_lines(grid, columns))
            f.write("end performance_measure\n\n")


def _write_hdf5(path, measures, options) -> None:
    """Write measures as an hdf5 file, one dataset per column."""
    import h5py

    with h5py.File(path, "w") as hf:
        hf.attrs["mf6adj_pm_version"] = 1
        for key, value in (options or {}).items():
            hf.attrs[key] = str(value)
        for name, (grid, columns) in measures.items():
            group = hf.create_group(f"performance_measures/{name}")
            group.attrs["discretization"] = grid
            for column, values in columns.items():
                # a column of names is held as fixed-length bytes, which
                # compress; the variable-length form carries a pointer and a
                # heap entry per value and is larger than the ascii file
                if values.dtype.kind in "US":
                    values = values.astype("S")
                group.create_dataset(
                    column, data=values, compression="gzip", compression_opts=4
                )


def write_performance_measures(
    path: PathLike,
    measures: dict[str, Any],
    options: Optional[dict[str, Any]] = None,
    format: str = "ascii",
) -> pl.Path:
    """Write performance measures as an adjoint input file.

    Parameters
    ----------
    path : PathLike
        File to write.
    measures : dict
        Measure name to its columns, given as a dict, a numpy recarray, or a
        dataframe. Locations are `cellid`, or the index columns of the
        discretization; times are `kper` and `kstp`; values are `pm_type`,
        `pm_form`, `weight`, and `obsval`. Indices are zero-based.
    options : dict, optional
        Options block, such as ``{"hdf5_name": "out.h5"}``.
    format : str
        ``ascii``, which is the format MODFLOW 6 input is written in and the
        default, or ``hdf5``, which is about nine times smaller.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    if format not in ("ascii", "hdf5"):
        raise Exception(f"format is `ascii` or `hdf5`, not `{format}`")
    if not measures:
        raise Exception("no performance measures to write")

    prepared = {name: normalize(measure) for name, measure in measures.items()}

    grids = {grid for grid, _ in prepared.values()}
    if len(grids) > 1:
        raise Exception(
            f"the measures are not all on the same discretization: {sorted(grids)}"
        )

    path = pl.Path(path)
    if format == "ascii":
        _write_ascii(path, prepared, options)
    else:
        _write_hdf5(path, prepared, options)
    return path


def is_hdf5(path: PathLike) -> bool:
    """Return whether a file is hdf5, by its signature rather than its name."""
    path = pl.Path(path)
    if not path.is_file():
        return False
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def _read_ascii(path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Read an adjoint input file into columns, without a model to resolve nodes."""
    measures: dict[str, dict[str, np.ndarray]] = {}
    options: dict[str, Any] = {}

    block = None
    rows: list[list[str]] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lowered = stripped.lower()

            if lowered.startswith("begin options"):
                block = "options"
            elif lowered.startswith("begin performance_measure"):
                block, rows = lowered.split()[2], []
            elif lowered.startswith("end options"):
                block = None
            elif lowered.startswith("end performance_measure"):
                measures[block] = _columns_from_rows(rows)
                block, rows = None, []
            elif block == "options":
                key, value = lowered.split(maxsplit=1)
                options[key] = value
            elif block is None:
                raise Exception(f"line outside a block: '{stripped}'")
            else:
                rows.append(lowered.split())

    if block is not None:
        raise EOFError(f"end of file while reading block '{block}'")
    return measures, options


def _columns_from_rows(rows) -> dict[str, np.ndarray]:
    """Return the entry lines of one measure as zero-based columns."""
    if not rows:
        raise Exception("a performance measure holds no entries")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise Exception(f"the entries do not all have the same width: {sorted(widths)}")

    width = widths.pop()
    grid = {9: "dis", 8: "disv", 7: "disu"}.get(width)
    if grid is None:
        raise Exception(f"an entry holds 7, 8, or 9 items, not {width}")

    names = (*TIME_COLUMNS, *LOCATION_COLUMNS[grid])
    values = list(zip(*rows))
    columns = {
        name: np.array(values[idx], dtype=np.int64) - 1
        for idx, name in enumerate(names)
    }
    columns["pm_type"] = np.array(values[-4])
    columns["pm_form"] = np.array(values[-3])
    columns["weight"] = np.array(values[-2], dtype=float)
    columns["obsval"] = np.array(values[-1], dtype=float)
    return columns


def _read_hdf5(path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Read an hdf5 performance-measure file into columns."""
    import h5py

    measures: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as hf:
        options = {
            key: value for key, value in hf.attrs.items() if key != "mf6adj_pm_version"
        }
        for name, group in hf.get("performance_measures", {}).items():
            columns = {}
            for column, dataset in group.items():
                values = dataset[:]
                if values.dtype.kind == "O":
                    values = np.array([v.decode() for v in values])
                elif values.dtype.kind == "S":
                    values = np.char.decode(values)
                columns[column] = values
            measures[name] = columns
    return measures, options


def read_performance_measures(
    path: PathLike,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Read performance measures from an adjoint input file, ascii or hdf5.

    The format is taken from the file itself rather than its name.

    Parameters
    ----------
    path : PathLike
        File to read.

    Returns
    -------
    tuple[dict, dict]
        Measure name to its zero-based columns, and the options block. The
        measures are in the form `write_performance_measures` accepts, so a
        file read here can be written back in either format.
    """
    path = pl.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"performance measure file not found: {path}")
    return _read_hdf5(path) if is_hdf5(path) else _read_ascii(path)
