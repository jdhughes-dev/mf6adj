import os
import pathlib as pl
import platform
import shutil
import sys
from datetime import datetime

import flopy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyemu
from matplotlib.backends.backend_pdf import PdfPages

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

env_path = pl.Path(os.environ.get("CONDA_PREFIX", None))
assert env_path is not None, (
    "autotest script must be run from the mf6adj Conda environment"
)

bin_path = "bin"
exe_ext = ""
if "linux" in platform.platform().lower():
    lib_ext = ".so"
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    lib_ext = ".dylib"
else:
    bin_path = "Scripts"
    lib_ext = ".dll"
    exe_ext = ".exe"
lib_name = env_path / f"{bin_path}/libmf6{lib_ext}"
mf6_bin = env_path / f"{bin_path}/mf6{exe_ext}"


def test_sagehen():
    prep = True

    org_d = "ex-gwf-sagehen-external"
    new_d = "sagehen_test"

    adj_file = os.path.join(new_d, "test.adj")
    if prep:
        if os.path.exists(new_d):
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)
        pyemu.os_utils.run("mf6", cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d, load_only=["dis", "sfr"])
    gwf = sim.get_model()

    with open(adj_file, "w") as f:
        sfr_data = pd.DataFrame.from_records(gwf.sfr.packagedata.array)
        f.write("begin performance_measure swgw\n")
        for kper in range(sim.tdis.nper.data):
            for kij in sfr_data.cellid.values:
                f.write(
                    f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} "
                    + "sfr-1 direct 1.0 -1.0e+30\n"
                )
        f.write("end performance_measure\n\n")

        # now a direct head pm at the terminal sfr reach (the last kij covered)
        f.write("begin performance_measure terminalhead\n")
        for kper in range(sim.tdis.nper.data):
            f.write(
                f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} "
                + "head direct 1.0 -1.0e+30\n"
            )
        f.write("end performance_measure\n")

    start = datetime.now()
    os.chdir(new_d)

    print("calculating adjoint...")
    adj = mf6adj.Mf6Adj(os.path.split(adj_file)[1], lib_name, logging_level="INFO")

    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)

    result_hdf = [
        f
        for f in os.listdir(new_d)
        if f.endswith("hd5") and f.startswith("adjoint_solution_swgw")
    ]
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]

    hdf = h5py.File(os.path.join(new_d, result_hdf), "r")
    keys = list(hdf.keys())
    keys.sort()

    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data

    idomain = gwf.dis.idomain.array
    thresh = 0.0001
    with PdfPages(os.path.join(new_d, "results.pdf")) as pdf:
        for key in keys:
            if key != "composite":
                continue
            grp = hdf[key]

            plot_keys = [i for i in grp.keys() if grp[i].shape == (nlay, nrow, ncol)]

            for pkey in plot_keys:
                arr = grp[pkey][:].reshape((nlay, nrow, ncol))
                for k, karr in enumerate(arr):
                    karr[idomain[k, :, :] < 1] = np.nan
                    ib = idomain[k, :, :].copy().astype(float)
                    ib[ib > 0] = np.nan
                    # karr[np.abs(karr)>1e20] = np.nan
                    karr[np.abs(karr) < thresh] = np.nan
                    # karr = np.log10(karr)
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    ax.imshow(ib, cmap="Greys_r")
                    cb = ax.imshow(karr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(
                        key
                        + ", "
                        + pkey
                        + f", layer:{k + 1}, masked where abs < {thresh}",
                        loc="left",
                    )
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...", key, pkey, k + 1)
