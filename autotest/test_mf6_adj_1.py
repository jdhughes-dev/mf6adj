import os
import sys
import platform
from datetime import datetime
import shutil
import string
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0,".")
import modflowapi
import flopy
import flopy.utils.cvfdutil
import pyemu

if "linux" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "linux", "libmf6.so")
    mf6_bin = os.path.join("..", "bin", "linux", "mf6")
    local_lib_name = "./libmf6.so"
    local_mf6_bin = "./mf6"
    gg_bin = "gridgen"
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() and "arm" not in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
    local_lib_name = "./libmf6.dylib"
    local_mf6_bin = "./mf6"
    gg_bin = "gridgen"
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() and "arm" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6_arm.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
    local_lib_name = "./libmf6_arm.dylib"
    local_mf6_bin = "./mf6"
    gg_bin = "gridgen"
else:
    lib_name = os.path.join("..", "bin", "win", "libmf6.dll")
    mf6_bin = os.path.join("..", "bin", "win", "mf6.exe")
    local_lib_name = "libmf6.dll"
    local_mf6_bin = "mf6.exe"
    gg_bin = "gridgen.exe"


#some plotting functions
def plot_colorbar_sensitivity(x, y, Sper,Sadj, contour_intervals, fname, nodenumber = None, rowcol = None):
    from matplotlib.patches import Polygon

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    # first subplot
    ax = axes[0]
    ax.set_title("Perturbation", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(Sper)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")

    contours = modelmap.contour_array(
        Sper,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%2.2f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)
    pa.set_clim(vmin=np.min(contour_intervals), vmax=np.max(contour_intervals))
    # if nodenumber is not None:
    #     verts = gwf.modelgrid.get_cell_vertices(nodenumber)
    # else:
    #     verts = gwf.modelgrid.get_cell_vertices(rowcol[0],rowcol[1])
    # p = Polygon(verts, facecolor='k')
    # ax.add_patch(p)

    # second subplot
    ax = axes[1]
    ax.set_title("MF6-ADJ", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    pa = modelmap.plot_array(Sadj)
    contours = modelmap.contour_array(
        Sadj,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%2.2f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)
    pa.set_clim(vmin=np.min(contour_intervals), vmax=np.max(contour_intervals))

    plt.savefig('{0}'.format(fname))

def basic_freyberg():
    #import mf6adj source
    if os.path.exists('mf6adj'):
        shutil.rmtree('mf6adj')
    shutil.copytree(os.path.join('..','mf6adj'),os.path.join('mf6adj'))
    import mf6adj
    
    org_d = "freyberg_mf6_pestppv5_test"#os.path.join("models","freyberg")
    new_d = "freyberg"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)
    shutil.copy2(lib_name,os.path.join(new_d,os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin,os.path.join(new_d,os.path.split(mf6_bin)[1]))

    os.chdir(new_d)
    os.system("mf6")
    os.chdir("..")

    kijs = []
    with open(os.path.join(new_d,"head.obs"),'r') as f:
        f.readline()
        for line in f:
            if line.strip().lower().startswith("end"):
                break
            raw = line.strip().split()
            kijs.append(" ".join(raw[2:]))

    np.random.seed(11111)
    rvals = np.random.random(len(kijs)) + 36
    with open(os.path.join(new_d,"test.adj"),'w') as f:
        f.write("\nbegin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type direct\n")
        for kij in kijs:
            for kper in range(25):
                f.write("{0} 1 {1} 1.0 \n".format(kper+1,kij))
        f.write("end performance_measure\n\n")

        f.write("begin performance_measure pm2 type residual\n")
        for rval,kij in zip(rvals,kijs):
            for kper in range(25):
                f.write("{0} 1 {1} 1.0  {2}\n".format(kper+1,kij,rval))
        f.write("end performance_measure\n\n")

    os.chdir(new_d)
    adj = mf6adj.Mf6Adj("test.adj",os.path.split(local_lib_name)[1])
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    
def twod_ss_nested_hetero_head_at_point():
    new_d = 'twod_ss_nested_hetero_head_at_point'
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.mkdir(new_d)
    shutil.copytree(os.path.join('..', 'mf6adj'), os.path.join(new_d, 'mf6adj'))
    shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))


    shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))
    os.chdir(new_d)
    import mf6adj

    name = "snglhdtest"
    tf = 1.0
    h1 = 100
    N = 7
    L = 700.0
    H = 1.0
    k = 1.0
    q = -300.0
    T = k * H
    L1 = L2 = L
    D = L1 * L2
    Nlay = 1
    Nrow = Ncol = N
    delrow = delcol = L / (N - 1)

    # first set up model for analytical solution comparison
    # ### Create the FloPy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=local_mf6_bin,
        version="mf6",
        sim_ws=".",
        memory_print_option="ALL"
    )

    # ### Create the Flopy `TDIS` object
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(tf, 1, 1.0)]
    )

    # ### Create the Flopy `IMS` Package object
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        linear_acceleration="BICGSTAB",
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{name}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )

    # outer grid
    nlay = 1
    nrow = ncol = 7
    delr = 100.0 * np.ones(ncol)
    delc = 100.0 * np.ones(nrow)
    tp = np.zeros((nrow, ncol))
    bt = -100.0 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol))
    idomain[:, 2:5, 2:5] = 0
    sg1 = flopy.discretization.StructuredGrid(
        delr=delr, delc=delc, top=tp, botm=bt, idomain=idomain
    )
    # inner grid
    nlay = 1
    nrow = ncol = 9
    delr = 100.0 / 3.0 * np.ones(ncol)
    delc = 100.0 / 3.0 * np.ones(nrow)
    tp = np.zeros((nrow, ncol))
    bt = -100 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol))
    sg2 = flopy.discretization.StructuredGrid(
        delr=delr,
        delc=delc,
        top=tp,
        botm=bt,
        xoff=200.0,
        yoff=200,
        idomain=idomain,
    )
    gridprops = flopy.utils.cvfdutil.gridlist_to_disv_gridprops([sg1, sg2])

    flopy.mf6.ModflowGwfdisv(
        gwf,
        length_units='meters',
        nlay=nlay,
        top=0.,
        botm=-100.,
        **gridprops,
    )

    # ### Create the initial conditions (`IC`) Package
    # start = h1 * np.zeros((nlay, N, N))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=0.)

    # ### Create the storage (`STO`) Package
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=0,
        steady_state = True,
    )

    chd_spd = []
    chd_spd += [[0, i, 0.0] for i in [0, 7, 14, 18, 22, 26, 33]]
    chd_spd = {0: chd_spd}
    chdl = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
        filename="{}.left.chd".format(gwf.name),
    )

    chd_spd = []
    chd_spd += [[0, i, 100.0] for i in [6, 13, 17, 21, 25, 32, 39]]
    chd_spd = {0: chd_spd}
    chdr = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
        filename="{}.right.chd".format(gwf.name),
    )
    # # ### Create the well (`WEL`) Package
    wel_rec = [(0, 80, q)]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_rec,
    )

    # ### Create the output control (`OC`) Package
    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "ALL")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )
    list_cond = [8.5113804,
                 7.0794578,
                 9.1201084,
                 8.7096359,
                 12.0226443,
                 10.4712855,
                 7.7624712,
                 8.7096359,
                 12.8824955,
                 13.1825674,
                 9.7723722,
                 8.5113804,
                 8.1283052,
                 11.4815362,
                 9.5499259,
                 13.1825674,
                 11.4815362,
                 8.9125094,
                 10.9647820,
                 11.4815362,
                 11.2201845,
                 12.3026877,
                 8.7096359,
                 9.5499259,
                 10.7151931,
                 11.4815362,
                 10.4712855,
                 9.7723722,
                 9.5499259,
                 7.0794578,
                 10.0000000,
                 7.0794578,
                 9.7723722,
                 11.2201845,
                 12.0226443,
                 12.3026877,
                 8.3176377,
                 8.3176377,
                 7.0794578,
                 13.8038426,
                 12.3026877,
                 12.5892541,
                 8.5113804,
                 10.7151931,
                 9.5499259,
                 10.9647820,
                 10.2329299,
                 12.8824955,
                 8.1283052,
                 11.7489755,
                 10.7151931,
                 14.1253754,
                 10.0000000,
                 7.2443596,
                 12.0226443,
                 8.3176377,
                 10.2329299,
                 8.5113804,
                 8.7096359,
                 9.1201084,
                 12.8824955,
                 12.8824955,
                 12.5892541,
                 9.1201084,
                 7.0794578,
                 13.4896288,
                 10.0000000,
                 9.3325430,
                 13.4896288,
                 10.0000000,
                 12.0226443,
                 10.7151931,
                 10.9647820,
                 7.7624712,
                 9.3325430,
                 9.3325430,
                 13.1825674,
                 7.4131024,
                 13.8038426,
                 7.4131024,
                 7.7624712,
                 9.5499259,
                 8.5113804,
                 7.0794578,
                 10.4712855,
                 12.5892541,
                 12.0226443,
                 8.5113804,
                 13.1825674,
                 9.3325430,
                 7.5857758,
                 10.4712855,
                 8.7096359,
                 11.7489755,
                 7.2443596,
                 10.2329299,
                 8.7096359,
                 13.8038426,
                 12.0226443,
                 7.0794578,
                 14.1253754,
                 7.0794578,
                 7.0794578,
                 12.3026877,
                 10.7151931,
                 12.0226443,
                 9.3325430,
                 10.9647820,
                 9.1201084,
                 8.3176377,
                 13.1825674,
                 9.5499259,
                 8.1283052,
                 11.4815362,
                 11.4815362,
                 9.1201084,
                 14.1253754,
                 13.1825674,
                 13.4896288,
                 8.9125094,
                 10.2329299
                 ]

    array_cond = np.array(list_cond)
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=array_cond,
    )

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    pyemu.os_utils.run(local_mf6_bin,cwd=".'")

    # now run with API
    print('test run to completion with API')
    mf6api = modflowapi.ModflowApi(local_lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
    CELLAREA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name)))
    CL1 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("CL1", "%s/CON" % name)))
    CL2 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("CL2", "%s/CON" % name)))
    HWVA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("HWVA", "%s/CON" % name)))
    CELLTOP = np.array(mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name)))
    CELLBOT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name)))
    JA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name)))
    IA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name)))
    JA_p = np.subtract(JA, 1)
    IA_p = np.subtract(IA, 1)
    IAC = np.array([IA[i + 1] - IA[i] for i in range(len(IA) - 1)])
    SAT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name)))
    K11 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name)))
    K22 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name)))
    K33 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name)))
    NODES = np.array(mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name)))[0]

    head = []
    amat = []
    time = []
    deltat = []

    h = []
    chd = []
    MAT = []
    iii = -1
    # Solving Forward MF6 Model
    while current_time < end_time:
        time.append(current_time)
        dt = mf6api.get_time_step()
        mf6api.prepare_time_step(dt)
        kiter = 0
        mf6api.prepare_solve(1)
        while kiter < max_iter:
            has_converged = mf6api.solve(1)
            kiter += 1
            if has_converged:
                iii += 1
                if iii == 0:
                    print('****************************************************************')
                    hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                    h.append([hi[item] for item in range(len(hi))])
                break
        mf6api.finalize_solve(1)
        mf6api.finalize_time_step()
        current_time = mf6api.get_current_time()
        dt1 = mf6api.get_time_step()
        deltat.append(dt1)
        CHD0 = mf6api.get_value_ptr(mf6api.get_var_address("NODELIST", "%s/CHD_0" % name))
        CHD1 = mf6api.get_value_ptr(mf6api.get_var_address("NODELIST", "%s/CHD_1" % name))
        chd = np.append(CHD0,CHD1) - 1
        amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
        MAT.append([amat[item] for item in range(len(amat))])
        rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
        head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
        h.append([head[item] for item in range(len(head))])
        head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
        if not has_converged:
            print("model did not converge")
            break
    try:
        mf6api.finalize()
        success = True
    except:
        raise RuntimeError

    # then calculate perturbation for head at point
    print('now calculating perturbation sensitivity')
    count = 0
    for index_sens in range(gwf.modelgrid.nnodes):
        count += 1
        if index_sens in chd:
            pass
        else:
            k = npf.k.array
            # k[0][index_sens] = k[0][index_sens] + epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=0,
                k=k,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            # DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            # DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            # DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            # DELC = np.array([DELC_[item] for item in range(len(DELC_))])
            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = gwf.output.head().get_alldata()[-1]
            J_constant = h2[0,0,79]
            break

    # now set epsilon
    f_sens = open("sens_per_nested.dat", "w")
    epsilon = 0.01
    list_S_per = []
    count = 0
    for index_sens in range(gwf.modelgrid.nnodes):
        k = array_cond.copy()
        if index_sens in chd:
            print(f_sens.write('{:2.4E}\n'.format(0.0)))
            list_S_per.append(0.)
        else:
            k[index_sens] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=0,
                k=k,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = gwf.output.head().get_alldata()[-1]
            J = h2[0,0,79]
            sens = (J - J_constant) / epsilon
            list_S_per.append(sens)
            print(f_sens.write('{:2.4E}\n'.format(sens)))
            count += 1
    f_sens.close()

    # # then calculate mfadj for head at point
    print('now calculating mfadj sensitivity from jeremy script')

    with open("test.adj",'w') as f:
        f.write("\nbegin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type direct\n")
        f.write("1 1 80 1.0 \n")
        f.write("end performance_measure\n\n")

    adj = mf6adj.Mf6Adj("test.adj", local_lib_name, False)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()

    #now plot up all three results
    array_S_per = np.array(list_S_per)
    array_S_adj = pd.read_csv('k123.dat')
    list_S_adj = array_S_adj.values.reshape(-1)

    x = np.linspace(0, L1, Ncol)
    y = np.linspace(0, L2, Nrow)
    y = y[::-1]
    minval = min(array_S_per)
    maxval = max(array_S_per)
    contour_intervals = np.linspace(minval, maxval, 5)
    plot_colorbar_sensitivity(x, y, array_S_per,array_S_adj.value, contour_intervals, 'snglhdtest_nested_hetero.png', nodenumber=79)

    f = open("sensitivity_nested_hetero.dat", "w")
    print(f.write('Perturbation  MF6-ADJ\n'))
    print(f.write('-----------------------\n'))
    for i in range(len(list_S_per)):
        print(f.write('{:2.4E} '.format(list_S_per[i])))
        print(f.write('{:2.4E} \n'.format(list_S_adj[i])))
    f.close()
    os.chdir('..')

def twod_ss_hetero_coarsegrid_test():
    new_d = 'twod_ss_hetero_coarsegrid'
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.mkdir(new_d)
    shutil.copytree(os.path.join('..', 'mf6adj'), os.path.join(new_d,'mf6adj'))
    shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))

    shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))
    os.chdir(new_d)
    import mf6adj

    name = "2dsshetrcgr"
    epsilon = 0 #start with zero for perturbation method, then run with 1% rule
    tf = 1.0
    h1 = 100
    N = 11
    L = 100.0
    H = 1.0
    k = 10.0
    q = -300.0
    T = k * H
    L1 = L2 = L
    D = L1 * L2
    Nlay = 1
    Nrow = Ncol = N
    delrow = delcol = L / (N - 1)
    bot = np.linspace(-H / Nlay, -H, Nlay)  # botm

    # first set up model for analytical solution comparison
    # ### Create the FloPy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=local_mf6_bin,
        version="mf6",
        sim_ws=".",
        memory_print_option="ALL"
    )

    # ### Create the Flopy `TDIS` object
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(tf, 1, 1.0)]
    )

    # ### Create the Flopy `IMS` Package object
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        linear_acceleration="BICGSTAB",
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{name}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )

    # ### Create the discretization (`DIS`) Package
    idm = np.ones((Nlay, N, N))
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=Nlay,
        nrow=N,
        ncol=N,
        delr=delrow,
        delc=delcol,
        top=0.0,
        botm=bot,
        idomain=idm
    )

    # ### Create the initial conditions (`IC`) Package
    start = h1 * np.ones((Nlay, N, N))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

    # ### Create the storage (`STO`) Package
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
    )

    # ### Create the constant head (`CHD`) Package
    chd_rec = []
    layer = 0
    for row_col in range(0, N):
        chd_rec.append(((layer, row_col, 0), h1))
        chd_rec.append(((layer, row_col, N - 1), h1))
        if row_col != 0 and row_col != N - 1:
            chd_rec.append(((layer, 0, row_col), h1))
            chd_rec.append(((layer, N - 1, row_col), h1))
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_rec,
    )

    iper = 0
    ra = chd.stress_period_data.get_data(key=iper)
    list_ch = [ra[item][0] for item in range(len(ra))]

    # # ### Create the well (`WEL`) Package
    wel_rec = [(Nlay - 1, int(N / 2), int(N / 2), q)]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_rec,
    )

    # ### Create the output control (`OC`) Package
    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "ALL")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )

    # ### Create the node property flow (`NPF`) Package
    list_cond = [8.5113804,
                 7.0794578,
                 9.1201084,
                 8.7096359,
                 12.0226443,
                 10.4712855,
                 7.7624712,
                 8.7096359,
                 12.8824955,
                 13.1825674,
                 9.7723722,
                 8.5113804,
                 8.1283052,
                 11.4815362,
                 9.5499259,
                 13.1825674,
                 11.4815362,
                 8.9125094,
                 10.9647820,
                 11.4815362,
                 11.2201845,
                 12.3026877,
                 8.7096359,
                 9.5499259,
                 10.7151931,
                 11.4815362,
                 10.4712855,
                 9.7723722,
                 9.5499259,
                 7.0794578,
                 10.0000000,
                 7.0794578,
                 9.7723722,
                 11.2201845,
                 12.0226443,
                 12.3026877,
                 8.3176377,
                 8.3176377,
                 7.0794578,
                 13.8038426,
                 12.3026877,
                 12.5892541,
                 8.5113804,
                 10.7151931,
                 9.5499259,
                 10.9647820,
                 10.2329299,
                 12.8824955,
                 8.1283052,
                 11.7489755,
                 10.7151931,
                 14.1253754,
                 10.0000000,
                 7.2443596,
                 12.0226443,
                 8.3176377,
                 10.2329299,
                 8.5113804,
                 8.7096359,
                 9.1201084,
                 12.8824955,
                 12.8824955,
                 12.5892541,
                 9.1201084,
                 7.0794578,
                 13.4896288,
                 10.0000000,
                 9.3325430,
                 13.4896288,
                 10.0000000,
                 12.0226443,
                 10.7151931,
                 10.9647820,
                 7.7624712,
                 9.3325430,
                 9.3325430,
                 13.1825674,
                 7.4131024,
                 13.8038426,
                 7.4131024,
                 7.7624712,
                 9.5499259,
                 8.5113804,
                 7.0794578,
                 10.4712855,
                 12.5892541,
                 12.0226443,
                 8.5113804,
                 13.1825674,
                 9.3325430,
                 7.5857758,
                 10.4712855,
                 8.7096359,
                 11.7489755,
                 7.2443596,
                 10.2329299,
                 8.7096359,
                 13.8038426,
                 12.0226443,
                 7.0794578,
                 14.1253754,
                 7.0794578,
                 7.0794578,
                 12.3026877,
                 10.7151931,
                 12.0226443,
                 9.3325430,
                 10.9647820,
                 9.1201084,
                 8.3176377,
                 13.1825674,
                 9.5499259,
                 8.1283052,
                 11.4815362,
                 11.4815362,
                 9.1201084,
                 14.1253754,
                 13.1825674,
                 13.4896288,
                 8.9125094,
                 10.2329299
                 ]

    array_cond = np.array(list_cond)
    array_cond_3D = np.reshape(array_cond, (Nlay, Nrow, Ncol))
    for kk in range(Nlay):
        for jj in range(Nrow):
            for ii in range(Ncol):
                if (kk, jj, ii) in list_ch:
                    array_cond_3D[kk][jj][ii] = k

    k = np.array(array_cond_3D)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=k,
    )

    list_triplets = []
    for nn in range(Nlay * Nrow * Ncol):
        list_triplets.append(gwf.modelgrid.get_lrc(nn))

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    pyemu.os_utils.run(local_mf6_bin)

    # now run with API
    print('test run to completion with API')
    mf6api = modflowapi.ModflowApi(local_lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
    CELLAREA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name)))
    DELR = np.array(mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name)))
    DELC = np.array(mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name)))
    CELLTOP = np.array(mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name)))
    CELLBOT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name)))
    JA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name)))
    IA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name)))
    JA_p = np.subtract(JA, 1)
    IA_p = np.subtract(IA, 1)
    IAC = np.array([IA[i + 1] - IA[i] for i in range(len(IA) - 1)])
    SAT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name)))
    K11 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name)))
    K22 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name)))
    K33 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name)))
    NODES = np.array(mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name)))[0]

    head = []
    amat = []
    time = []
    deltat = []

    h = []
    MAT = []
    iii = -1
    # Solving Forward MF6 Model
    while current_time < end_time:
        time.append(current_time)
        dt = mf6api.get_time_step()
        mf6api.prepare_time_step(dt)
        kiter = 0
        mf6api.prepare_solve(1)
        while kiter < max_iter:
            has_converged = mf6api.solve(1)
            kiter += 1
            if has_converged:
                iii += 1
                if iii == 0:
                    print('****************************************************************')
                    hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                    h.append([hi[item] for item in range(len(hi))])
                break
        mf6api.finalize_solve(1)
        mf6api.finalize_time_step()
        current_time = mf6api.get_current_time()
        dt1 = mf6api.get_time_step()
        deltat.append(dt1)
        amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
        MAT.append([amat[item] for item in range(len(amat))])
        rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
        head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
        h.append([head[item] for item in range(len(head))])
        head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
        if not has_converged:
            print("model did not converge")
            break
    try:
        mf6api.finalize()
        success = True
    except:
        raise RuntimeError


    # then calculate analytical solution for head integral
    print('now calculating perturbation sensitivity')
    count = 0
    for index_sens in list_triplets[0:105]:
        count += 1
        k = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            # print(f_sens.write('{:2.4E}\n'.format(0.0)))
            pass
        else:
            k[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=0,
                k=k,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            DELC = np.array([DELC_[item] for item in range(len(DELC_))])

            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = gwf.output.head().get_alldata()[-1]
            hh = np.reshape(h2[0], (Nlay * Nrow * Ncol))
            sum = 0.0
            for ii in range(len(IA) - 1):
                sum += hh[ii]# * CELLAREA[ii]
            J_constant = sum
            print(J_constant)
            break
    #now set epsilon
    f_sens = open("sens_per.dat", "w")
    list_S_per = []
    epsilon = 1.0e-5
    count = 0
    for index_sens in list_triplets:
        count += 1
        k = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            print(f_sens.write('{:2.4E}\n'.format(0.0)))
            list_S_per.append(0.)
        else:
            k[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=0,
                k=k,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            DELC = np.array([DELC_[item] for item in range(len(DELC_))])

            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = gwf.output.head().get_alldata()[-1]
            hh = np.reshape(h2[0], (Nlay * Nrow * Ncol))
            sum = 0.0
            for ii in range(len(IA) - 1):
                sum += hh[ii]# * CELLAREA[ii]
            J = sum

            sens = (J - J_constant) / epsilon
            list_S_per.append(sens)
            print(f_sens.write('{:2.4E}\n'.format(sens)))
    f_sens.close()

    with open("test.adj", 'w') as f:
        f.write("begin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type direct\n")
        for k in range(Nlay):
            for i in range(Nrow):
                for j in range(Ncol):
                    f.write("1 1 {0} {1} {2} 1.0\n".format(k+1,i+1,j+1))
        f.write("end performance_measure pm1\n")

    adj = mf6adj.Mf6Adj("test.adj",local_lib_name,True)
    adj.solve_gwf()
    adj.solve_adjoint()

    s_adj = np.loadtxt("k123_layer001.dat").reshape(-1)
    os.chdir('..')
    s_per = np.array(list_S_per)

    diff = np.abs(s_adj - s_per)
    print(diff)
    assert diff.max() < 1e-5


def twod_ss_hetero_head_at_point():
    new_d = 'twod_ss_hetero_head_at_point'
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.mkdir(new_d)
    shutil.copytree(os.path.join('..', 'mf6adj'), os.path.join(new_d,'mf6adj'))
    shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))
    if "linux" in platform.platform().lower():
        local_lib_name =  "libmf6.so"
        local_mf6_bin =  "mf6"
    elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
        local_lib_name = "libmf6.dylib"
        local_mf6_bin = "mf6"
    else:
        local_lib_name =  "libmf6.dll"
        local_mf6_bin = "mf6.exe"
    shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))
    
    org_mh_dir = "mh_org_codes"
    for f in os.listdir(org_mh_dir):
        shutil.copy2(os.path.join(org_mh_dir,f),os.path.join(new_d,f))


    os.chdir(new_d)
    
    
    sys.path.append(os.path.join("..",".."))
    import mf6adj

    name = "freyberg6"
    epsilon = 1.0e-05
    tf = 1.0
    h1 = 100
    N = 11 # dont change this...
    L = 100.0
    H = 1.0
    k = 10.0
    q = -300.0
    T = k * H
    L1 = L2 = L
    D = L1 * L2
    Nlay = 1
    Nrow = Ncol = N
    delrow = delcol = L / (N - 1)
    bot = np.linspace(-H / Nlay, -H, Nlay)  # botm

    # first set up model for analytical solution comparison
    # ### Create the FloPy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=local_mf6_bin,
        version="mf6",
        sim_ws=".",
        memory_print_option="ALL"
    )

    # ### Create the Flopy `TDIS` object
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(tf, 1, 1.0)]
    )

    # ### Create the Flopy `IMS` Package object
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        linear_acceleration="BICGSTAB",
        inner_dvclose=0.000000001,
        outer_dvclose=0.000000001
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{name}.nam"
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )

    # ### Create the discretization (`DIS`) Package
    idm = np.ones((Nlay, N, N))
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=Nlay,
        nrow=N,
        ncol=N,
        delr=delrow,
        delc=delcol,
        top=0.0,
        botm=bot,
        idomain=idm
    )

    # ### Create the initial conditions (`IC`) Package
    start = h1 * np.ones((Nlay, N, N))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

    # ### Create the storage (`STO`) Package
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
    )

    # ### Create the constant head (`CHD`) Package
    chd_rec = []
    layer = 0
    for row in range(Nrow):
        chd_rec.append(((layer, row, 0), 0))
        chd_rec.append(((layer, row, Ncol - 1), 100))

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_rec,
    )


    iper = 0
    ra = chd.stress_period_data.get_data(key=iper)
    list_ch = [ra[item][0] for item in range(len(ra))]

    # # ### Create the well (`WEL`) Package
    wel_rec = [(Nlay - 1, int(N / 2), int(N / 2)-1, q)]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_rec,
    )

    # ### Create the output control (`OC`) Package
    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "ALL")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )

    # ### Create the node property flow (`NPF`) Package
    list_cond = [8.5113804,
                 7.0794578,
                 9.1201084,
                 8.7096359,
                 12.0226443,
                 10.4712855,
                 7.7624712,
                 8.7096359,
                 12.8824955,
                 13.1825674,
                 9.7723722,
                 8.5113804,
                 8.1283052,
                 11.4815362,
                 9.5499259,
                 13.1825674,
                 11.4815362,
                 8.9125094,
                 10.9647820,
                 11.4815362,
                 11.2201845,
                 12.3026877,
                 8.7096359,
                 9.5499259,
                 10.7151931,
                 11.4815362,
                 10.4712855,
                 9.7723722,
                 9.5499259,
                 7.0794578,
                 10.0000000,
                 7.0794578,
                 9.7723722,
                 11.2201845,
                 12.0226443,
                 12.3026877,
                 8.3176377,
                 8.3176377,
                 7.0794578,
                 13.8038426,
                 12.3026877,
                 12.5892541,
                 8.5113804,
                 10.7151931,
                 9.5499259,
                 10.9647820,
                 10.2329299,
                 12.8824955,
                 8.1283052,
                 11.7489755,
                 10.7151931,
                 14.1253754,
                 10.0000000,
                 7.2443596,
                 12.0226443,
                 8.3176377,
                 10.2329299,
                 8.5113804,
                 8.7096359,
                 9.1201084,
                 12.8824955,
                 12.8824955,
                 12.5892541,
                 9.1201084,
                 7.0794578,
                 13.4896288,
                 10.0000000,
                 9.3325430,
                 13.4896288,
                 10.0000000,
                 12.0226443,
                 10.7151931,
                 10.9647820,
                 7.7624712,
                 9.3325430,
                 9.3325430,
                 13.1825674,
                 7.4131024,
                 13.8038426,
                 7.4131024,
                 7.7624712,
                 9.5499259,
                 8.5113804,
                 7.0794578,
                 10.4712855,
                 12.5892541,
                 12.0226443,
                 8.5113804,
                 13.1825674,
                 9.3325430,
                 7.5857758,
                 10.4712855,
                 8.7096359,
                 11.7489755,
                 7.2443596,
                 10.2329299,
                 8.7096359,
                 13.8038426,
                 12.0226443,
                 7.0794578,
                 14.1253754,
                 7.0794578,
                 7.0794578,
                 12.3026877,
                 10.7151931,
                 12.0226443,
                 9.3325430,
                 10.9647820,
                 9.1201084,
                 8.3176377,
                 13.1825674,
                 9.5499259,
                 8.1283052,
                 11.4815362,
                 11.4815362,
                 9.1201084,
                 14.1253754,
                 13.1825674,
                 13.4896288,
                 8.9125094,
                 10.2329299
                 ]

    array_cond = np.array(list_cond)
    array_cond_3D = np.reshape(array_cond, (Nlay, Nrow, Ncol))
    for kk in range(Nlay):
        for jj in range(Nrow):
            for ii in range(Ncol):
                if (kk, jj, ii) in list_ch:
                    array_cond_3D[kk][jj][ii] = k

    k = np.array(array_cond_3D)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=k,
    )

    list_triplets = []
    for nn in range(Nlay * Nrow * Ncol):
        list_triplets.append(gwf.modelgrid.get_lrc(nn))

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    pyemu.os_utils.run(local_mf6_bin)

    # now run with API
    print('test run to completion with API')
    mf6api = modflowapi.ModflowApi(local_lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
    CELLAREA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name)))
    DELR = np.array(mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name)))
    DELC = np.array(mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name)))
    CELLTOP = np.array(mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name)))
    CELLBOT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name)))
    JA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name)))
    IA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name)))
    JA_p = np.subtract(JA, 1)
    IA_p = np.subtract(IA, 1)
    IAC = np.array([IA[i + 1] - IA[i] for i in range(len(IA) - 1)])
    SAT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name)))
    K11 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name)))
    K22 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name)))
    K33 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name)))
    NODES = np.array(mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name)))[0]

    head = []
    amat = []
    time = []
    deltat = []

    h = []
    MAT = []
    iii = -1
    # Solving Forward MF6 Model
    while current_time < end_time:
        time.append(current_time)
        dt = mf6api.get_time_step()
        mf6api.prepare_time_step(dt)
        kiter = 0
        mf6api.prepare_solve(1)
        while kiter < max_iter:
            has_converged = mf6api.solve(1)
            kiter += 1
            if has_converged:
                iii += 1
                if iii == 0:
                    print('****************************************************************')
                    hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                    h.append([hi[item] for item in range(len(hi))])
                break
        mf6api.finalize_solve(1)
        mf6api.finalize_time_step()
        current_time = mf6api.get_current_time()
        dt1 = mf6api.get_time_step()
        deltat.append(dt1)
        amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
        MAT.append([amat[item] for item in range(len(amat))])
        rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
        head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
        h.append([head[item] for item in range(len(head))])
        head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
        if not has_converged:
            print("model did not converge")
            break
    try:
        mf6api.finalize()
        success = True
    except:
        raise RuntimeError

    # then calculate perturbation for head at point
    print('now calculating perturbation sensitivity')
    epsilon = 0
    count = 0
    for index_sens in list_triplets:#[0:105]:
        count += 1
        k = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            # print(f_sens.write('{:2.4E}\n'.format(0.0)))
            pass
        else:
            k[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=0,
                k=k,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            DELC = np.array([DELC_[item] for item in range(len(DELC_))])
            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = gwf.output.head().get_alldata()[-1]
            hh = np.reshape(h2[0], (Nlay * Nrow * Ncol))
            J_constant = h2[0,4,4]
            print(J_constant)
            break
    # now for the rest
    f_sens = open("sens_per_single_head.dat", "w")
    epsilon = 0.0001
    list_S_per = []
    count = 0
    for index_sens in list_triplets:
        count += 1
        k = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            print(f_sens.write('{:2.4E}\n'.format(0.0)))
            list_S_per.append(0.)
        else:
            k[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=0,
                k=k,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            DELC = np.array([DELC_[item] for item in range(len(DELC_))])
            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = gwf.output.head().get_alldata()[-1]
            J = h2[0,4,4]
            sens = (J - J_constant) / epsilon
            list_S_per.append(sens)
            print(f_sens.write('{:2.4E}\n'.format(sens)))
    f_sens.close()

    # then calculate mfadj for head at point
    print('now calculating mf6adj sensitivity')

    with open("test.adj",'w') as f:
        f.write("\nbegin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type direct\n")
        f.write("1 1 1 {0} {1} 1.0 \n".format(5,5))
        f.write("end performance_measure\n\n")

    adj = mf6adj.Mf6Adj("test.adj", local_lib_name, True)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()

    #now plot up both results
    array_S_per = np.array(list_S_per)

    S_per = np.reshape(array_S_per, (Nlay, Nrow, Ncol))
    np.savetxt("s_per.txt",S_per[0,:,:],fmt="%15.6e")
    S_adj = np.loadtxt('pm-pm1_comp_sens_k_k000.dat')

    # now plot up both results

    diff = np.abs(S_per - S_adj)
    print(diff)
    print(diff.max())
    assert diff.max() < 1.0e-5

    os.chdir('..')

def _skip_for_now_freyberg():
    org_d = "freyberg"
    test_d = "freyberg_test1"
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    shutil.copytree(org_d,test_d)
    shutil.copy2(mf6_bin,os.path.join(test_d,os.path.split(mf6_bin)[1]))
    shutil.copy2(lib_name,os.path.join(test_d,os.path.split(lib_name)[1]))

    sim = flopy.mf6.MFSimulation.load(sim_ws = test_d, exe_name=os.path.split(mf6_bin)[1])

    sim.tdis.nper = 1
    sim.tdis.perioddata = [(sim.tdis.perioddata.get_data()[0][0],sim.tdis.perioddata.get_data()[0][1],sim.tdis.perioddata.get_data()[0][2])]

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    pyemu.os_utils.run(local_mf6_bin)

    pert_results = os.path.join(test_d,"pert_results.csv")
    if not os.path.exists(pert_results):
        df = update_freyberg_pert(sim,test_d)
        df.to_csv(pert_results)
    else:
        df = pd.read_csv(pert_results)


    # then calculate mfadj for head at point

    print('now calculating mfadj sensitivity from jeremy script')

    with open("test.adj",'w') as f:
        f.write("\nbegin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type direct\n")
        f.write("1 1 1 20 10 1.0 \n")
        f.write("end performance_measure\n\n")

    if os.path.exists('mf6adj'):
        shutil.rmtree('mf6adj')
    shutil.copytree(os.path.join('..','mf6adj'),os.path.join('mf6adj'))
    local_lib_name = os.path.split(lib_name)[1]
    import mf6adj
    bd = os.getcwd()
    os.chdir(test_d)
    adj = mf6adj.Mf6Adj("test.adj", local_lib_name, True)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()

    os.chdir('..')
    arr = np.loadtxt(os.path.join('freyberg','k123_layer001.dat'))
    print(arr)
    df.loc[:,"mf6adj"] = df.apply(lambda x: arr[int(x.i),int(x.j)],axis=1)
    print(df)
    # S_jdub = np.loadtxt(os.path.join('freyberg','k123_layer001.dat'))
    # list_S_mf6adj = S_jdub.reshape(-1)
    #
    # x = np.linspace(0, L1, Ncol)
    # y = np.linspace(0, L2, Nrow)
    # y = y[::-1]
    # minval = min(array_S_per)
    # maxval = max(array_S_per)
    # contour_intervals = np.linspace(minval, maxval, 5)
    # plot_colorbar_sensitivity(x, y, array_S_per, array_S_per,list_S_mf6adj, contour_intervals, 'freyberg_test.png', rowcol=[19,9])
    #
    # f = open("sensitivity_freyberg.dat", "w")
    # print(f.write('Perturbation  MF6-ADJ\n'))
    # print(f.write('-----------------------\n'))
    # for i in range(len(array_S_per)):
    #     print(f.write('{:2.4E} '.format(list_S_per[i])))
    #     print(f.write('{:2.4E} \n'.format(list_S_mf6adj[i])))
    # f.close()

def update_freyberg_pert(sim,model_ws):
    # then calculate perturbation for head at point
    print('now calculating perturbation sensitivity')
    local_lib_name = os.path.split(lib_name)[1]
    m = sim.get_model()
    global gwf
    gwf = m
    name = m.name
    idomain = m.dis.idomain.array
    org_k = m.npf.k.array.copy()
    global Nlay
    Nlay = m.dis.nlay.data
    Nrow, Ncol = m.dis.nrow.data, m.dis.ncol.data
    delr, delc = m.dis.delr.data, m.dis.delc.data
    L1 = Ncol * delc
    L2 = Nrow * delr
    bd = os.getcwd()
    os.chdir(model_ws)
    count = 0
    for index_sens in range(m.modelgrid.nnodes):
        (k, i, j) = m.modelgrid.get_lrc(index_sens)[0]

        kh = org_k
        if idomain[k, i, j] == 0:
            pass
        else:
            kh[k, i, j] += 0
            m.npf.k = kh

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            DELC = np.array([DELC_[item] for item in range(len(DELC_))])
            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except Exception as e:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = m.output.head().get_alldata()[-1]
            J_constant = h2[0, 19, 9]
            count += 1
            break
    # now set epsilon
    f_sens = open("sens_per_single_head.dat", "w")
    list_S_per,inode,kk,ii,jj = [],[],[],[],[]
    epsilon = .001
    for index_sens in range(m.modelgrid.nnodes):
        (k, i, j) = m.modelgrid.get_lrc(index_sens)[0]
        inode.append(index_sens)
        kk.append(k)
        ii.append(i)
        jj.append(j)

        kh = org_k
        if idomain[k, i, j] == 0:
            print(f_sens.write('{:2.4E}\n'.format(0.0)))
            list_S_per.append(0.)

        else:
            kh[k, i, j] += epsilon
            m.npf.k = kh

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            AREA = mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name))
            DELR_ = mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name))
            DELC_ = mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name))
            DELR = np.array([DELR_[item] for item in range(len(DELR_))])
            DELC = np.array([DELC_[item] for item in range(len(DELC_))])
            TOP = mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name))
            BOT = mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name))
            JA_ = mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name))
            IA_ = mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name))
            SAT_ = mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name))
            SAT = np.array([SAT_[item] for item in range(len(SAT_))])
            K11_ = mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name))
            K11 = np.array([K11_[item] for item in range(len(K11_))])
            K22_ = mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name))
            K22 = np.array([K22_[item] for item in range(len(K22_))])
            K33_ = mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name))
            K33 = np.array([K33_[item] for item in range(len(K33_))])
            SAT_TH = np.array([SAT[item] * (TOP[item] - BOT[item]) for item in range(len(SAT))])  # Saturated thickness
            NODES = mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name))

            JA = np.array([JA_[item] for item in range(len(JA_))])
            IA = np.array([IA_[item] for item in range(len(IA_))])
            IAC = []
            for i in range(len(IA) - 1):
                IAC.append(IA[i + 1] - IA[i])

            IAC = np.array([IAC[item] for item in range(len(IAC))])

            CELLAREA = [AREA[item] for item in range(len(AREA))]
            CELLTOP = [TOP[item] for item in range(len(TOP))]
            CELLBOT = [BOT[item] for item in range(len(BOT))]

            head = []
            amat = []
            time = []
            deltat = []
            JA_p = np.array([number - 1 for number in JA])
            IA_p = np.array([number - 1 for number in IA])

            h = []
            MAT = []
            iii = -1

            while current_time < end_time:
                time.append(current_time)
                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        iii += 1
                        if iii == 0:
                            print('****************************************************************')
                            hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                            h.append([hi[item] for item in range(len(hi))])
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                dt1 = mf6api.get_time_step()
                deltat.append(dt1)
                amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
                MAT.append([amat[item] for item in range(len(amat))])
                rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
                head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
                h.append([head[item] for item in range(len(head))])
                head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError

            # -----------------------------------------------------------------------------------------------------------------------
            time.append(end_time)
            h2 = m.output.head().get_alldata()[-1]
            J = h2[0, 19, 9]
            sens = (J - J_constant) / epsilon
            list_S_per.append(sens)

            print(f_sens.write('{:2.4E}\n'.format(sens)))
    f_sens.close()
    df = pd.DataFrame(data={"node":inode,"k":kk,"i":ii,"j":jj,"pert":list_S_per})
    os.chdir(bd)
    return df


def test_freyberg_mh():
    """compare the Mf6Adj results to MH's adj code

    TODO: get an MH variant that does direct heads instead of obj function
    
    """
    org_d2 = "freyberg_mf6_pestppv5_mh"
    org_d1 = "freyberg_mh_adj"
    org_ds = [org_d1]
    for org_d in org_ds:
        test_d = org_d + "_test"
        if os.path.exists(test_d):
            shutil.rmtree(test_d)
        shutil.copytree(org_d,test_d)

        sim = flopy.mf6.MFSimulation.load(sim_ws=org_d)
        gwf = sim.get_model()
        id = gwf.dis.idomain.array[0,:,:]    
        mf6adj_d = os.path.join(test_d,'mf6adj')
        if os.path.exists(mf6adj_d):
            shutil.rmtree(mf6adj_d)
        shutil.copytree(os.path.join('..','mf6adj'),mf6adj_d)
        shutil.copy2(mf6_bin,os.path.join(test_d,os.path.split(mf6_bin)[1]))
        shutil.copy2(lib_name,os.path.join(test_d,os.path.split(lib_name)[1]))

        for d in ["bmipy","xmipy","modflowapi"]:
            dest = os.path.join(test_d,d)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(d,dest)

        local_lib_name = os.path.split(lib_name)[1]
        bd = os.getcwd()
        sys.path.append(os.path.join(".."))
        import mf6adj
        print(mf6adj.__file__)
        os.chdir(test_d)
        #try:     
        adj = mf6adj.Mf6Adj("test.adj", local_lib_name, True,verbose_level=3)
        # pd = mf6adj.Mf6Adj.get_package_names_from_gwfname(os.path.join("freyberg6.nam"))
        # print(pd)
        # exit()
        
        adj.solve_gwf()
        adj.solve_adjoint()
        adj.finalize()
        #except Exception as e:
        #    os.chdir(bd)
        #    raise Exception(e)
        os.chdir(bd)
        base_d = org_d + "_base"

        if True:
            
            # run MH's adj code
            
            if os.path.exists(base_d):
                shutil.rmtree(base_d)
            shutil.copytree(org_d,base_d)

            org_code_d = "mh_org_codes"
            for pyfile in [f for f in os.listdir(org_code_d) if f.endswith(".py")]:
                assert not os.path.exists(os.path.join(base_d,pyfile)),"pyfile exists "+pyfile
                shutil.copy2(os.path.join(org_code_d,pyfile),os.path.join(base_d,pyfile))
                          

            bd = os.getcwd()
            os.chdir(base_d)
            try:
                ret_val = os.system("python Adjoint.py")
            except Exception as e:
                os.chdir(bd)
                raise Exception(e)
            if "window" in platform.platform().lower():
                if ret_val != 0:
                    os.chdir(bd)
                    raise Exception("run() returned non-zero: {0}".format(ret_val))
            else:
                estat = os.WEXITSTATUS(ret_val)
                if estat != 0:
                    os.chdir(bd)
                    raise Exception("run() returned non-zero: {0}".format(estat))
            os.chdir(bd)

        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages("h_k.pdf") as pdf:
            for kper in range(sim.tdis.nper.data):
                h1file = os.path.join(test_d,"pm-pm1_head_kper{0:05d}_k000.dat".format(kper))
                k1file = os.path.join(test_d,"pm-pm1_sens_k11_kper{0:05d}_k000.dat".format(kper))
                k2file = os.path.join(base_d,"sens_k_kper{0:05d}_k000.dat".format(kper))
                h1 = np.loadtxt(h1file)
                k1 = np.loadtxt(k1file)
                k2 = np.loadtxt(k2file)
                h1[id==0] = np.nan
                k1[id==0] = np.nan
                k2[id==0] = np.nan
                fig,axes= plt.subplots(1,2,figsize=(10,5))
                ax = axes[0]
                cb = ax.imshow(k1)
                plt.colorbar(cb,ax=ax)
                con = ax.contour(h1,colors='k')
                ax.clabel(con)
                ax.set_title("kper:{0}".format(kper))
                ax = axes[1]
                cb = ax.imshow(k2)
                plt.colorbar(cb,ax=ax)
                con = ax.contour(h1,colors='k')
                ax.clabel(con)
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)


        a1 = np.loadtxt(os.path.join(test_d,"pm-pm1_dadk11_kper00000.dat"))
        a2 = np.loadtxt(os.path.join(base_d,"dadk11.dat"))
        d = np.abs(a1-a2)
        print(d.max())
       # assert d.max() < 1.0e-6

        files = [f for f in os.listdir(test_d) if f.startswith("pm-pm1_dadk11_kper") and "_k0" not in f]
        assert len(files) == 25
        for f in files:
            a1 = np.loadtxt(os.path.join(test_d,f))
            d = np.abs(a1-a2)
            #assert d.max() < 1.0e-6

        for kper in range(sim.tdis.nper.data):
            a1 = np.loadtxt(os.path.join(test_d,"pm-pm1_adjstates_kper{0:05d}_k000.dat".format(kper)))
            a2 = np.loadtxt(os.path.join(base_d,"adjstates_kper{0:05d}_k000.dat".format(kper)))
            a1[id==0] = 0
            a2[id==0] = 0
            d = np.abs(a1-a2)
            print(d.max())
            assert d.max() < 1e-6
            
        tags = ["comp_sens_k11_k000.dat","comp_sens_k33_k000.dat","comp_sens_ss_k000.dat"]
        
        obs_arr = np.zeros((gwf.dis.nrow.data,gwf.dis.ncol.data))
        print(obs_arr.shape)
        for pm in adj._performance_measures[0]._entries:
            print(pm._i,pm._j)
            obs_arr[pm._i,pm._j] = 1
        obs_arr[obs_arr==0] = np.nan
        h1 = np.loadtxt(os.path.join(test_d,"pm-pm1_head_kper00024_k000.dat"))
        h1[id==0] = np.nan
        
        with PdfPages("compare.pdf") as pdf:
            for tag in tags:
                a1 = np.loadtxt(os.path.join(test_d,"pm-pm1_"+tag))
                a1[id==0] = 0
                a2 = np.loadtxt(os.path.join(base_d,tag))
                a2[id==0] = 0
                vmn = min(a1.min(),a2.min())
                vmx = max(a1.max(),a2.max())
                d = (a1 - a2)
                a1[id==0] = np.nan
                a2[id==0] = np.nan
                fig,axes = plt.subplots(1,3,figsize=(11,7))
                cb = axes[0].imshow(a1,vmin=vmn,vmax=vmx)
                plt.colorbar(cb,ax=axes[0])
                cb = axes[1].imshow(a2,vmin=vmn,vmax=vmx)
                plt.colorbar(cb,ax=axes[1])
                cb = axes[2].imshow(d)
                plt.colorbar(cb,ax=axes[2])
                axes[0].set_title(tag)
                axes[2].set_title("max:{0:4.1f}, min:{1:4.1}".format(d.max(),d.min()))
                for ax in axes.flatten():
                    ax.imshow(obs_arr,cmap="jet_r",alpha=0.85)
                    ax.contour(h1)
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                print(d)
                print(tag,d.max())
                assert np.abs(d).max() < 1e-6

def test_3d_freyberg():
    """test the 3D freyberg model from the pestpp v5 report.  
    
    TODO: work out what the basis of comparison is - pertubations?  
    TODO: variants using the different vcond options - need to add the vcond option checking to Mf6Adj

    """
    org_d = "freyberg_mf6_pestppv5"
    test_d = "freyberg_mf6_pestppv5_test"
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    shutil.copytree(org_d,test_d)

    #write the adj file using the truth values
    import pyemu
    pst = pyemu.Pst(os.path.join(test_d,"truth.pst"))
    obs = pst.observation_data
    res = pst.res
    obs.loc[:,"obsval"] = res.loc[obs.obsnme,"modelled"].values
    tobs = obs.loc[obs.obsnme.str.startswith("trgw"),:].copy()
    tobs.loc[:,"k"] = tobs.obsnme.apply(lambda x: int(x.split("_")[1]))
    tobs.loc[:,"i"] = tobs.obsnme.apply(lambda x: int(x.split("_")[2]))
    tobs.loc[:,"j"] = tobs.obsnme.apply(lambda x: int(x.split("_")[3]))
    tobs.loc[:,"datetime"] = tobs.obsnme.apply(lambda x: pd.to_datetime(x.split("_")[-1],format="%Y%m%d"))
    udatetimes = list(tobs.datetime.unique())
    udatetimes.sort()
    with open(os.path.join(test_d,"test.adj"),'w') as f:
        f.write("begin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type residual\n")
        for kper,udt in enumerate(udatetimes):
            uobs = tobs.loc[tobs.datetime==udt,:].copy()
            uobs.sort_index(inplace=True)
            for k,i,j,weight,obsval,obsnme in zip(uobs.k,uobs.i,uobs.j,uobs.weight,uobs.obsval,uobs.obsnme):
                f.write("{0} 1 {1} {2} {3} {4} {5} #{6}\n".format(kper+1,k+1,i+1,j+1,weight,obsval,obsnme))
        f.write("end performance_measure pm1\n\n")
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_d)
    gwf = sim.get_model()
    id = gwf.dis.idomain.array[0,:,:]    
    mf6adj_d = os.path.join(test_d,'mf6adj')
    if os.path.exists(mf6adj_d):
        shutil.rmtree(mf6adj_d)
    shutil.copytree(os.path.join('..','mf6adj'),mf6adj_d)
    shutil.copy2(mf6_bin,os.path.join(test_d,os.path.split(mf6_bin)[1]))
    shutil.copy2(lib_name,os.path.join(test_d,os.path.split(lib_name)[1]))

    for d in ["bmipy","xmipy","modflowapi"]:
        dest = os.path.join(test_d,d)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(d,dest)

    local_lib_name = os.path.split(lib_name)[1]
    bd = os.getcwd()
    sys.path.append(os.path.join(".."))
    import mf6adj
    print(mf6adj.__file__)
    os.chdir(test_d)
    adj = mf6adj.Mf6Adj("test.adj",local_lib_name,verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir(bd)



def test_freyberg_unstruct():
    """test a quadtree refined grid.  
    TODO: work out what the base condition for comparison is?  Do we run pertubations with pestpp-glm? 
    TODO: variants for the vcond options.

    """
    org_d = "freyberg_quadtree"
    test_d = "freyberg_quadtree_test"
    if os.path.exists(test_d):
       shutil.rmtree(test_d)
    shutil.copytree(org_d,test_d)
  
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_d)
    gwf = sim.get_model()
    mf6adj_d = os.path.join(test_d,'mf6adj')
    if os.path.exists(mf6adj_d):
        shutil.rmtree(mf6adj_d)
    shutil.copytree(os.path.join('..','mf6adj'),mf6adj_d)
    shutil.copy2(mf6_bin,os.path.join(test_d,os.path.split(mf6_bin)[1]))
    shutil.copy2(lib_name,os.path.join(test_d,os.path.split(lib_name)[1]))

    for d in ["bmipy","xmipy","modflowapi"]:
        dest = os.path.join(test_d,d)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(d,dest)

    local_lib_name = os.path.split(lib_name)[1]
    bd = os.getcwd()
    sys.path.append(os.path.join(".."))
    import mf6adj
    print(mf6adj.__file__)
    os.chdir(test_d)
    adj = mf6adj.Mf6Adj("test.adj",local_lib_name,verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir(bd)

    # todo: some kinda of testing/plotting here!


def test_freyberg_unstructured_match():
    """unstructured test  using the same discret, etc as the MH freyberg test but uses disv, 
    which will allow us to directly compare the results with MH's adj code
    
    TODO: use gridgen to setup the basic files; 
    TODO: isolate MH's adj code calls above so that we can use them here
    TODO: intersect/workout gw obs locations in the node numbers for the adj file

    """
    pass

def test_zaidel():
    """test the upstream weighting conductance scheme and newton solution process,stolen from mf6 examples!

    TODO: variants for both direct and residual performance measures

    """
    org_d = "ex-gwf-zaidel"
    test_d = org_d+"_test"
    if os.path.exists(test_d):
       shutil.rmtree(test_d)
    shutil.copytree(org_d,test_d)
  
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_d)
    gwf = sim.get_model()
    botm = gwf.dis.botm.array
    print(botm)
    
    mf6adj_d = os.path.join(test_d,'mf6adj')
    if os.path.exists(mf6adj_d):
        shutil.rmtree(mf6adj_d)
    shutil.copytree(os.path.join('..','mf6adj'),mf6adj_d)
    shutil.copy2(mf6_bin,os.path.join(test_d,os.path.split(mf6_bin)[1]))
    shutil.copy2(lib_name,os.path.join(test_d,os.path.split(lib_name)[1]))

    for d in ["bmipy","xmipy","modflowapi"]:
        dest = os.path.join(test_d,d)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(d,dest)

    local_lib_name = os.path.split(lib_name)[1]
    bd = os.getcwd()
    sys.path.append(os.path.join(".."))
    import mf6adj
    print(mf6adj.__file__)
    os.chdir(test_d)
    adj = mf6adj.Mf6Adj("test.adj",local_lib_name,verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir(bd)

def plot_freyberg_verbose_structured_output(test_d):
    from matplotlib.backends.backend_pdf import PdfPages
    files = os.listdir(test_d)
    sim = flopy.mf6.MFSimulation.load(sim_ws=test_d)
    id = sim.get_model().dis.idomain.array[0,:,:]
    pm_files = [f for f in files if f.lower().endswith(".dat") and f.lower().startswith("pm-") and "k22" not in f]
    pm_vals = [f.replace("pm-","").split("_")[0] for f in pm_files]
    pm_vals = list(set(pm_vals))
    for pm_val in pm_vals:
        ppm_files = [f for f in pm_files if f.lower().startswith("pm-{0}".format(pm_val)) and "_k0" in f.lower()]
        ppm_files.sort()
        with PdfPages(os.path.join(test_d,pm_val+".pdf")) as pdf:
            for ppm_file in ppm_files:
                arr = np.loadtxt(os.path.join(test_d,ppm_file))
                arr[id==0] = np.nan
                fig,ax = plt.subplots(1,1,figsize=(4,7))
                cb = ax.imshow(arr)
                plt.colorbar(cb)
                ax.set_title(ppm_file)
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                print(ppm_file)


def setup_xd_box_model(new_d,sp_len=1.0,nper=1,hk=1.0,k33=1.0,q=-0.1,ss=1.0e-5,
                     nlay=1,nrow=10,ncol=10,delrowcol=1.0,icelltype=0,iconvert=0,newton=False,
                     top=1,botm=None,include_sto=True,include_id0=True,name = "freyberg6",
                       full_sat_ghb=True):

    tdis_pd = [(sp_len, 1, 1.0) for _ in range(nper)]
    if botm is None:
        botm = np.linspace(0, -nlay + 1, nlay)
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    os.mkdir(new_d)
    shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))

    #shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    #shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    #shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))

    org_mh_dir = "mh_org_codes"
    for f in os.listdir(org_mh_dir):
        shutil.copy2(os.path.join(org_mh_dir, f), os.path.join(new_d, f))

    #org_aux_dir = "xd_box_test_aux_files"
    #for f in os.listdir(org_aux_dir):
    #    shutil.copy2(os.path.join(org_aux_dir, f), os.path.join(new_d, f))



    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name=local_mf6_bin, version="mf6", sim_ws=new_d,
                                 memory_print_option="ALL",continue_=True)

    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", nper=len(tdis_pd), perioddata=tdis_pd)

    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="SIMPLE", linear_acceleration="BICGSTAB",
                               inner_dvclose=1e-6, outer_dvclose=1e-6, outer_maximum=1000, inner_maximum=1000)

    model_nam_file = f"{name}.nam"
    newtonoptions = []
    if newton:
        newtonoptions = ["NEWTON"]
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file, save_flows=True,newtonoptions=newtonoptions)

    idm = np.ones((nlay, nrow, ncol))
    if ncol > 1 and nrow > 1 and include_id0:
        idm[0, 0, 1] = 0  # just to have one in active cell...
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delrowcol, delc=delrowcol, top=top, botm=botm,
                                  idomain=idm)
    id = dis.idomain.array.copy()
    # ### Create the initial conditions (`IC`) Package
    start = top * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

    # ### Create the storage (`STO`) Package
    if include_sto:
        #if len(tdis_pd) > 1:
        #    raise Exception("not implemented")
        sy = []
        geo = [top]
        geo.extend(botm)
        for k in range(nlay):
            t,b = geo[k],geo[k+1]
            sy.append(ss*(t-b))
        steady_state = [False]
        if len(tdis_pd) > 1:
            steady_state = [False for _ in range(len(tdis_pd))]
            steady_state[0] = True
        sto = flopy.mf6.ModflowGwfsto(gwf, iconvert=iconvert, steady_state=steady_state,ss=ss,sy=sy)

    chd_rec = []
    if ncol > 1:
        chd_stage = top
        if not full_sat_ghb:
            chd_stage = (top-botm[0])/4.0

        for k in [nlay-1]:
            for i in range(nrow):
                chd_rec.append(((k, i, 0), chd_stage,1000.0))

        #chd = flopy.mf6.ModflowGwfghb(gwf, stress_period_data=chd_rec)

    ghb_rec = chd_rec
    ghb_stage = top+1
    if not full_sat_ghb:
        ghb_stage = 3.*top/4.
    for k in [0]:
        for i in range(nrow):
            ghb_rec.append(((k, i, ncol - 1), ghb_stage, 1000.0))

    ghb = flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_rec)

    if nrow > 1 and ncol > 1:
        wel_rec = [(nlay - 1, int(nrow / 2), int(ncol / 2), q)]
        wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_rec)

    flopy.mf6.ModflowGwfrcha(gwf, recharge=0.0001)

    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = {kper:[("HEAD", "ALL"), ("BUDGET", "ALL")] for kper in range(nper)}
    printrecord = {kper:[("HEAD", "ALL")] for kper in range(nper)}
    oc = flopy.mf6.ModflowGwfoc(gwf, saverecord=saverecord, head_filerecord=head_filerecord,
                                budget_filerecord=budget_filerecord, printrecord=printrecord)



    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=hk, k33=k33)


    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()

    # hack in tvk so we can use pert within api later
    with open(os.path.join(new_d, "blank.tvk"), 'w') as f:
        f.write("\n")

    # npf_file = os.path.join(new_d,name+".npf")
    # lines = open(npf_file,'r').readlines()
    # with open(npf_file,'w') as f:
    #     for line in lines:
    #         f.write(line)
    #         if "begin options" in line.lower():
    #             f.write("tvk6_filename blank.tvk\n")

    pyemu.os_utils.run(local_mf6_bin,cwd=new_d)
    return sim


def run_xd_box_pert(new_d,p_kijs,plot_pert_results=True,weight=1.0,pert_mult=1.01,
                    name = "freyberg6",obsval=1.0,pm_locs=None):
    import modflowapi
    # # now run with API
    bd = os.getcwd()
    os.chdir(new_d)
    sys.path.append(os.path.join("..", ".."))
    import mf6adj
    print(os.listdir("."))
    print('test run to completion with API')
    mf6api = modflowapi.ModflowApi(local_lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))

    sim = flopy.mf6.MFSimulation.load(sim_ws=".")
    gwf = sim.get_model()
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    if pm_locs is None:
        pm_locs = [(nlay-1,nrow-1,ncol-2)]
    include_sto = True
    if gwf.sto is None:
        include_sto = False
    head_base = []
    swr_head_base = []
    while current_time < end_time:
        dt = mf6api.get_time_step()
        mf6api.prepare_time_step(dt)
        kiter = 0
        mf6api.prepare_solve(1)
        while kiter < max_iter:
            has_converged = mf6api.solve(1)
            kiter += 1
            if has_converged:
                break
        mf6api.finalize_solve(1)
        mf6api.finalize_time_step()
        current_time = mf6api.get_current_time()
        dt1 = mf6api.get_time_step()
        head_base.append(mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name)).copy())
        swr_head_base.append(
            ((mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name)).copy() - obsval) * weight) ** 2)
        if not has_converged:
            print("model did not converge")
            break

    np.savetxt("dbase.dat", head_base[0], fmt="%15.6E")
    np.savetxt("swrbase.dat", swr_head_base[0], fmt="%15.6E")

    addr = ["NODEUSER", "FREYBERG6", "DIS"]
    wbaddr = mf6api.get_var_address(*addr)
    nuser = mf6api.get_value(wbaddr).copy() - 1
    if nuser.shape[0] == 1:
        nuser = np.arange(nlay * nrow * ncol, dtype=int)

    kijs = gwf.modelgrid.get_lrc(list(nuser))
    try:
        mf6api.finalize()
        success = True
    except:
        raise RuntimeError()

    props = [gwf.npf.k.array.copy()]
    flopy_objects = [gwf.npf]
    flopy_prop_names = ["k"]
    tags = ["k11"]

    if nlay > 1:
        props.append(gwf.npf.k33.array.copy())
        flopy_objects.append(gwf.npf)
        flopy_prop_names.append("k33")
        tags.append("k33")

    if include_sto:
        props.append(gwf.sto.ss.array.copy())
        flopy_objects.append(gwf.sto)
        flopy_prop_names.append("ss")
        tags.append("ss")

    for prop, fobj, fname, tag in zip(props, flopy_objects, flopy_prop_names, tags):

        arr = prop.copy()
        epsilon = {}
        head_pert = {}
        swr_pert = {}
        count = 0
        for k, i, j in p_kijs:
            count += 1
            kk_arr = arr.copy()
            kij = (k, i, j)
            dk = kk_arr[k, i, j] * pert_mult
            epsilon[kij] = dk - kk_arr[k, i, j]
            kk_arr[k, i, j] = dk

            if kij not in head_pert:
                head_pert[kij] = []
                swr_pert[kij] = []
            fobj.__setattr__(fname, kk_arr)
            sim.write_simulation()
            np.savetxt("pert_arr_{0}_pk{1}_pi{2}_pj{3}.dat".format(tag,k,i,j), kk_arr.flatten(), fmt="%15.6E")
            # sim.run_simulation()
            mf6api = modflowapi.ModflowApi(local_lib_name)
            mf6api.initialize()
            current_time = mf6api.get_current_time()
            end_time = mf6api.get_end_time()
            max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
            #condsat = mf6api.get_value_ptr(mf6api.get_var_address("CONDSAT", name, "NPF"))
            kper = 0
            while current_time < end_time:

                dt = mf6api.get_time_step()
                mf6api.prepare_time_step(dt)
                kiter = 0
                mf6api.prepare_solve(1)
                while kiter < max_iter:
                    has_converged = mf6api.solve(1)
                    kiter += 1
                    if has_converged:
                        break
                mf6api.finalize_solve(1)
                mf6api.finalize_time_step()
                current_time = mf6api.get_current_time()
                head = mf6api.get_value(mf6api.get_var_address("X", "%s" % name))
                head_pert[(k, i, j)].append(head.copy())
                swr_pert[(k, i, j)].append(((head.copy() - obsval) * weight) ** 2)
                #if tag == "ss":
                np.savetxt("dpert_{3}_kper{4}_pk{0}_pi{1}_pj{2}.dat".format(k, i, j,tag,kper), head_pert[(k, i, j)][-1], fmt="%15.6E")
                np.savetxt("swrpert_{3}_kper{4}_pk{0}_pi{1}_pj{2}.dat".format(k, i, j,tag,kper), swr_pert[(k, i, j)][-1], fmt="%15.6E")

                if not has_converged:
                    print("model did not converge")
                    break
            try:
                mf6api.finalize()
                success = True
            except:
                raise RuntimeError()
            kper += 1

        pert_direct_sens = {}
        pert_swr_sens = {}
        for kij in p_kijs:

            delt = []
            swr_delt = []
            for kper in range(sim.tdis.nper.data):
                delt.append((head_pert[kij][kper] - head_base[kper]) / epsilon[kij])
                swr_delt.append((swr_pert[kij][kper] - swr_head_base[kper]) / epsilon[kij])
            pert_direct_sens[kij] = delt
            pert_swr_sens[kij] = swr_delt

        if plot_pert_results:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf = PdfPages("pert_sens_{0}.pdf".format(tag))

        #for p_i_node,p_kij in enumerate(p_kijs):
        for p_kij in pm_locs:
            pk, pi, pj = p_kij
            #if p_kij != (0, 0, 2):
            #    continue
            if p_kij not in kijs:
                print("pm kij missing",p_kij)
                continue
            p_i_node = kijs.index(p_kij)
            for kper in range(sim.tdis.nper.data):
                head_plot = np.zeros((nlay, nrow, ncol))
                for kij, h, p in zip(kijs, head_base[kper], head_pert[p_kij][kper]):
                    head_plot[kij] = h
                head_plot = head_plot.reshape((nlay, nrow, ncol))
                head_plot[id == 0] = np.nan
                dsens_plot = np.zeros((nlay, nrow, ncol))
                ssens_plot = np.zeros((nlay, nrow, ncol))
                for inode,kij in enumerate(kijs):
                    dsens = pert_direct_sens[kij][kper][p_i_node]
                    swrsens = pert_swr_sens[kij][kper][p_i_node]
                    dsens_plot[kij] = dsens
                    ssens_plot[kij] = swrsens
                dsens_plot = dsens_plot.reshape((nlay, nrow, ncol))
                dsens_plot[id == 0] = np.nan
                ssens_plot = ssens_plot.reshape((nlay, nrow, ncol))
                ssens_plot[id == 0] = np.nan

                if plot_pert_results:
                    if nlay > 1:
                        fig, axes = plt.subplots(nlay, 3, figsize=(nlay * 10,15))
                    else:
                        fig, axes = plt.subplots(nlay, 3, figsize=(15,5))
                        axes = np.atleast_2d(axes)
                for k in range(nlay):
                    if plot_pert_results:
                        cb = axes[k, 0].imshow(head_plot[k, :, :])
                        plt.colorbar(cb, ax=axes[k,0])
                        cb = axes[k, 1].imshow(dsens_plot[k, :, :])
                        plt.colorbar(cb, ax=axes[k,1])
                        cb = axes[k, 2].imshow(ssens_plot[k, :, :])
                        plt.colorbar(cb, ax=axes[k,2])
                        axes[k, 0].set_title("heads k:{0},kper:{1}".format(k, kper), loc="left")
                        axes[k, 1].set_title("dsens kij:{2} k:{0},kper:{1}".format(k, kper, p_kij), loc="left")
                        axes[k, 2].set_title("swrsens kij:{2} k:{0},kper:{1}".format(k, kper, p_kij), loc="left")
                    np.savetxt("pert-direct_kper{0:03d}_pk{1:03d}_pi{2:03d}_pj{3:03d}_comp_sens_{4}_k{5:03d}.dat".format(kper,pk, pi, pj, tag,k),
                               dsens_plot[k, :, :], fmt="%15.6E")
                    np.savetxt("pert-phi_kper{0:03d}_pk{1:03d}_pi{2:03d}_pj{3:03d}_comp_sens_{4}_k{5:03d}.dat".format(kper,pk,pi,pj,tag,k),ssens_plot[k, :, :], fmt="%15.6E")

                if plot_pert_results:
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)


        fobj.__setattr__(fname, arr)
        sim.write_simulation()
        #sim.run_simulation()
        if plot_pert_results:
            pdf.close()
    os.chdir(bd)


def xd_box_compare(new_d,plot_compare=False,plt_zero_thres=1e-6):
    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data

    pm_files = [os.path.join(new_d,f) for f in os.listdir(new_d) if f.startswith("pm-") and f.endswith(".dat") and "comp_sens" in f]
    if nlay == 1:
        pm_files = [f for f in pm_files if "k33" not in f]
    assert len(pm_files) > 0

    # temp filter
    pm_files = [f for f in pm_files if "ghb" not in f and "wel" not in f]
    pm_files.sort()
    pert_files = [f.replace("pm-","pert-") for f in pm_files]
    if plot_compare:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(os.path.join(new_d,"compare.pdf"))

    for pm_file,pert_file in zip(pm_files,pert_files):
        k = int(pm_file.split(".")[0].split("_")[-1][1:])
        pm_arr = np.atleast_2d(np.loadtxt(pm_file))
        pm_arr[id[k,:,:]==0] = 0
        pert_arr = np.atleast_2d(np.loadtxt(pert_file))
        pm_arr[id[k, :, :] == 0] = 0
        d = pm_arr - pert_arr
        demon = pert_arr.copy()
        demon[demon==0] = 1e-10
        p = 100 * np.abs(d) / np.nanmax(np.abs(pert_arr))
        # todo checks for closeness...

        print(pert_file,np.nanmax(np.abs(d)),np.nanmax(np.abs(p)))
        if plot_compare:
            fig,axes = plt.subplots(1,4,figsize=(35,4))
            absd = np.abs(d)
            absp = np.abs(p)

            absd[absd<plt_zero_thres] = 0
            #pert_arr[absd==0] = np.nan
            #pm_arr[absd==0] = np.nan
            d[absd==0] = np.nan
            p[absd==0] = np.nan
            mx = max(np.nanmax(pert_arr),np.nanmax(pm_arr))
            mn = min(np.nanmin(pert_arr), np.nanmin(pm_arr))
            cb = axes[0].imshow(pert_arr,vmax=mx,vmin=mn)
            plt.colorbar(cb,ax=axes[0])
            axes[0].set_title(pert_file,loc="left")


            cb = axes[1].imshow(pm_arr, vmax=mx, vmin=mn)
            plt.colorbar(cb, ax=axes[1])
            axes[1].set_title(pm_file,loc="left")
            mx = np.nanmax(absd)
            cb = axes[2].imshow(d,vmin=-mx,vmax=mx,cmap="coolwarm")
            plt.colorbar(cb,ax=axes[2])
            axes[2].set_title("pert - pm, not showing abs(diff) <= {0}".format(plt_zero_thres),loc="left")
            mx = np.nanmax(absp)
            cb = axes[3].imshow(p)
            plt.colorbar(cb, ax=axes[3])
            axes[3].set_title("percent diff pert - pm",loc="left")
            if np.any(id==0):
                idp = id[k,:,:].copy().astype(float)
                idp[idp!=0] = np.nan
                axes[0].imshow(idp,cmap="magma")
                axes[1].imshow(idp, cmap="magma")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
    if plot_compare:
        pdf.close()


def test_xd_box_1():

    """
    permutations:
     - w and w/o sto
     - icelltype (including spatially varying)
     - inconvert (including spatially varying)
     - 2d/3d
     - nrow == 1 or not
     - ncol == 1 or not
     - has/has not id 0
     - spatially varying props
     - multiple kper (including mix of ss and tr, stresses turning on and off, different lengths, with and without multiple timesteps)
     - newton/nonnewton
     - newton with dry cells/upstream weighting - possibly with turning off pumping well
     - unit/nonunit delrowcol
     - spatially varying top and/or botm
     - unstructured (pass thru gridgen with no refinement)
     - multiple instances of the same package type

    Todo:
     - add bcs to pert
     - setup assertions in compare
     - add trap for CHDs and warn...

    """
    # workflow flags
    include_id0 = True  # include an idomain = cell
    include_sto = True

    clean = True # run the pertbuation process
    run_pert = True # the pertubations
    plot_pert_results = True #plot the pertubation results

    run_adj = True
    plot_adj_results = False # plot adj result

    plot_compare = True

    new_d = 'xd_box_1_test'

    if clean:
       sim = setup_xd_box_model(new_d,nper=3,include_sto=include_sto,include_id0=include_id0,nrow=5,ncol=5,nlay=2,
                                q=-0.1,icelltype=1,iconvert=1,newton=True,delrowcol=1.0,full_sat_ghb=False)
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay,nrow,ncol = gwf.dis.nlay.data,gwf.dis.nrow.data,gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    p_kijs = []
    # pm_locs = [(nlay-1,int(nrow/2),ncol-2),(0,int(nrow/2),ncol-2)]

    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                p_kijs.append((k, i, j))

    pm_locs = []
    for k in [0, int(nlay / 2), nlay-1]:
        for i in [0, int(nrow / 2), nrow-1]:
            for j in [0, int(ncol / 2), ncol-1]:
                    pm_locs.append((k, i, j))
            break
    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0
    sys.path.insert(0,os.path.join(".."))
    import mf6adj
    if run_adj:
        bd = os.getcwd()
        os.chdir(new_d)
        sys.path.append(os.path.join("..",".."))

        print('calculating mf6adj sensitivity')
        # with open("test.adj",'w') as f:
        #     f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
        #     for kper in range(sim.tdis.nper.data):
        #         for p_kij in pm_locs:
        #             k,i,j = p_kij
        #             if id[k,i,j] <= 0:
        #                 continue
        #             # just looking at one pm location for now...
        #             #if p_kij != (0,0,2):
        #             #    continue
        #             pm_name = "direct_kper{0:03d}_pk{1:03d}_pi{2:03d}_pj{3:03d}".format(kper,k,i,j)
        #             f.write("begin performance_measure {0} type direct\n".format(pm_name))
        #             f.write("{0} 1 {1} {2} {3} {4} \n".format(kper+1,k+1,i+1,j+1,weight))
        #             f.write("end performance_measure\n\n")
        #
        #             pm_name = "phi_kper{0:03d}_pk{1:03d}_pi{2:03d}_pj{3:03d}".format(kper,k,i,j)
        #             f.write("begin performance_measure {0} type residual\n".format(pm_name))
        #             # just use top as the obs val....
        #             f.write("{0} 1 {1} {2} {3} {4} {5}\n".format(kper+1,k+1,i+1,j+1,obsval,weight))
        #             f.write("end performance_measure\n\n")

        with open("test.adj",'w') as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = "direct_pk{1:03d}_pi{2:03d}_pj{3:03d}".format("0", k, i, j)
                f.write("begin performance_measure {0} type direct\n".format(pm_name))
                for kper in range(sim.tdis.nper.data):
                    f.write("{0} 1 {1} {2} {3} {4} \n".format(kper+1,k+1,i+1,j+1,weight))
                f.write("end performance_measure\n\n")


                pm_name = "phi_pk{1:03d}_pi{2:03d}_pj{3:03d}".format("0",k,i,j)

                f.write("begin performance_measure {0} type residual\n".format(pm_name))
                for kper in range(sim.tdis.nper.data):
                    f.write("{0} 1 {1} {2} {3} {4} {5}\n".format(kper+1,k+1,i+1,j+1,obsval,weight))
                f.write("end performance_measure\n\n")

        adj = mf6adj.Mf6Adj("test.adj", local_lib_name,verbose_level=1)
        adj.solve_gwf()
        adj.solve_adjoint()
        adj._perturbation_test()
        adj.finalize()


        if plot_adj_results:
            afiles_to_plot = [f for f in os.listdir(".") if (f.startswith("pm-direct") or f.startswith("pm-phi")) and f.endswith(".dat")]
            afiles_to_plot.sort()
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages('adj.pdf') as pdf:
                for i,afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(afile))
                    fig,ax = plt.subplots(1,1,figsize=(10,10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb,ax=ax)
                    ax.set_title(afile,loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile,i,len(afiles_to_plot))


        os.chdir(bd)

    if run_pert:
        run_xd_box_pert(new_d,p_kijs,plot_pert_results,weight,pert_mult,obsval=obsval,pm_locs=pm_locs)

    xd_box_compare(new_d,plot_compare)


def test_xd_box_unstruct_1():


    # workflow flags
    include_id0 = True  # include an idomain = cell - has to be false for unstructured...
    include_sto = True

    clean = True # run the pertbuation process
    run_pert = True # the pertubations
    plot_pert_results = True #plot the pertubation results

    run_adj = True
    plot_adj_results = False # plot adj result

    plot_compare = True

    new_d = 'xd_box_1_unstruct'
    nrow = ncol = 15
    nlay = 2
    if clean:
       sim = setup_xd_box_model(new_d,include_sto=include_sto,include_id0=include_id0,nrow=nrow, ncol=ncol,nlay=nlay,
                                q=-0.1,icelltype=1,iconvert=0,newton=True,delrowcol=1.0,full_sat_ghb=False)
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0
    xcc, ycc = np.atleast_2d(gwf.modelgrid.xcellcenters),np.atleast_2d(gwf.modelgrid.ycellcenters)

    from flopy.utils.gridgen import Gridgen
    g = Gridgen(gwf.dis, model_ws=new_d, exe_name=gg_bin)
    g.build(verbose=True)
    gridprops = g.get_gridprops_disv()
    idomain = gwf.dis.idomain.array.copy()

    disv_idomain = []
    for k in range(gwf.dis.nlay.data):
        disv_idomain.append(idomain[k].flatten())
    gwf.remove_package("dis")
    disv = flopy.mf6.ModflowGwfdisv(gwf,idomain=disv_idomain,**gridprops)

    disv.write()
    ghb = gwf.get_package("ghb")

    f_ghb = open(os.path.join(new_d,"freyberg6_disv.ghb"),'w')
    f_ghb.write("begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n")

    f_ghb.write("begin dimensions\nmaxbound {0}\nend dimensions\n\n".format(ghb.stress_period_data.data[0].shape[0]))

    if ghb is not None:
        ghb_spd = {}
        for kper in range(sim.tdis.nper.data):
            rarray = ghb.stress_period_data.data[kper]
            #print(rarray)
            xs = [xcc[cid[1],cid[2]] for cid in rarray.cellid]
            ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
            ilay = [cid[0] for cid in rarray.cellid]
            xys = [(x,y) for x,y in zip(xs,ys)]
            # use zero for the layer so that we get the cell2d value back
            inodes = [g.intersect([xy],"point",0)[0][0] for xy,il in zip(xys,ilay)]
            data = [[(il,inode),bhead,cond] for il,inode,bhead,cond in zip(ilay,inodes,rarray.bhead,rarray.cond)]
            ghb_spd[kper] = data
            f_ghb.write("begin period {0}\n".format(kper+1))
            [f_ghb.write("{0:9d} {1:9d} {2:15.6E} {3:15.6E}\n".format(il+1,inode+1,bhead,cond)) for il, inode, bhead, cond in
                    zip(ilay, inodes, rarray.bhead, rarray.cond)]
            f_ghb.write("end period {0}\n\n".format(kper+1))
    f_ghb.close()

    # now hack the nam file
    nam_file = os.path.join(new_d,"freyberg6.nam")
    lines = open(nam_file,'r').readlines()


    with open(nam_file,'w') as f:
        for line in lines:
            if "dis" in line.lower():
                line = "DISV6 freyberg6.disv disv\n"
            elif "ghb" in line.lower():
                line = "GHB6 freyberg6_disv.ghb ghb\n"
            f.write(line)

    pyemu.os_utils.run("mf6",cwd=new_d)

    p_kinodes = []
    for k in range(nlay):
        for inode in range(gwf.disv.ncpl.data):
            p_kinodes.append((k, inode))

    pm_locs = []
    for k in [0, int(nlay / 2), nlay-1]:
        for inode in range(0,gwf.disv.ncpl.data,int(nrow/2)):
            pm_locs.append((k, inode))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    #if run_pert:
    #    run_xd_box_pert(new_d,p_kinodes,plot_pert_results,weight,pert_mult,obsval=obsval,pm_locs=pm_locs)

    sys.path.insert(0,os.path.join(".."))
    import mf6adj

    if run_adj:
        bd = os.getcwd()
        os.chdir(new_d)


        with open("test.adj",'w') as f:
            f.write("\nbegin options\n\nend options\n\n")
            for kper in range(sim.tdis.nper.data):
                for p_kinode in pm_locs:
                    k,inode = p_kinode
                    pm_name = "direct_kper{0:03d}_pk{1:03d}_pinode{2:03d}".format(kper,k,inode)
                    f.write("begin performance_measure {0} type direct\n".format(pm_name))
                    f.write("{0} 1 {1} {2} {3} \n".format(kper+1,k+1,inode+1,weight))
                    f.write("end performance_measure\n\n")

                    pm_name = "phi_kper{0:03d}_pk{1:03d}_pinode{2:03d}".format(kper, k, inode)
                    f.write("begin performance_measure {0} type residual\n".format(pm_name))
                    f.write("{0} 1 {1} {2} {3} {4}\n".format(kper+1,k+1,inode+1,obsval,weight))
                    f.write("end performance_measure\n\n")

        adj = mf6adj.Mf6Adj("test.adj", local_lib_name, verbose_level=1)
        adj.solve_gwf()
        adj.solve_adjoint()
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [f for f in os.listdir(".") if (f.startswith("pm-direct") or f.startswith("pm-phi")) and f.endswith(".dat")]
            afiles_to_plot.sort()
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages('adj.pdf') as pdf:
                for i,afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(afile))
                    fig,ax = plt.subplots(1,1,figsize=(10,10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb,ax=ax)
                    ax.set_title(afile,loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile,i,len(afiles_to_plot))


        os.chdir(bd)



def freyberg_structured_demo():
    
    if os.path.exists('mf6adj'):
        shutil.rmtree('mf6adj')
    #shutil.copytree(os.path.join('..','mf6adj'),os.path.join('mf6adj'))
    sys.path.insert(0,os.path.join(".."))
    import mf6adj

    org_d = "freyberg_structured"
    new_d = "freyberg_structured_test"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)
    shutil.copy2(lib_name,os.path.join(new_d,os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin,os.path.join(new_d,os.path.split(mf6_bin)[1]))
    shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))
    #shutil.copytree(os.path.join('mf6adj'), os.path.join(new_d, 'mf6adj'))

    os.chdir(new_d)
    os.system("mf6")
    os.chdir("..")

    lrcs = []
    k_dict = {}
    with open(os.path.join(new_d,"head.obs"),'r') as f:
        f.readline()
        for line in f:
            if line.strip().lower().startswith("end"):
                break
            raw = line.strip().split()
            lrcs.append(" ".join(raw[2:]))
            k = int(raw[2]) - 1
            i = int(raw[3]) - 1
            j = int(raw[4]) - 1
            if k not in k_dict:
                k_dict[k] = []
            k_dict[k].append([i,j])

    np.random.seed(11111)
    rvals = np.random.random(len(lrcs)) + 36
    with open(os.path.join(new_d,"test.adj"),'w') as f:

        f.write("begin performance_measure pm1 type residual\n")
        for rval,lrc in zip(rvals,lrcs):
            for kper in range(25):
                f.write("{0} 1 {1} 1.0  {2}\n".format(kper+1,lrc,rval))
        f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)
    adj = mf6adj.Mf6Adj("test.adj",os.path.split(local_lib_name)[1],verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:",duration)

    result_hdf = [f for f in os.listdir(new_d) if f.endswith("hd5") and f.startswith("adj_pm")]
    print(result_hdf)
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]
    import h5py
    hdf = h5py.File(os.path.join(new_d,result_hdf),'r')
    keys = list(hdf.keys())
    keys.sort()
    print(keys)
    from matplotlib.backends.backend_pdf import PdfPages
    idomain = np.loadtxt(os.path.join(new_d,"freyberg6.dis_idomain_layer1.txt"))
    with PdfPages(os.path.join(new_d,"results.pdf")) as pdf:
        for key in keys:
            if key != "composite":
                continue

            grp = hdf[key]
            plot_keys = [i for i in grp.keys() if len(grp[i].shape) == 3]
            for pkey in plot_keys:

                arr = grp[pkey][:]
                for k,karr in enumerate(arr):
                    karr[idomain < 1] = np.nan
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    cb = ax.imshow(karr)
                    plt.colorbar(cb,ax=ax)
                    ax.set_title(key+", "+pkey+", layer:{0}".format(k+1),loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...",key, pkey,k+1)


def freyberg_quadtree_demo():
    if os.path.exists('mf6adj'):
        shutil.rmtree('mf6adj')
    #shutil.copytree(os.path.join('..', 'mf6adj'), os.path.join('mf6adj'))
    sys.path.insert(0,os.path.join(".."))
    import mf6adj

    org_d = "freyberg_quadtree"
    new_d = "freyberg_quadtree_test"
    prep_run = False
    run_adj = True

    if prep_run:
        if os.path.exists(new_d):
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)
        shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
        shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))
        shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
        shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
        shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
        shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))

        os.chdir(new_d)
        os.system("mf6")
        os.chdir("..")


    if run_adj:
        df = pd.read_csv(os.path.join(new_d,"freyberg6.obs_continuous_heads.csv.txt"),header=None,names=["site","otype","layer","node"])
        df.loc[:,"layer"] = df.layer.astype(int)
        df.loc[:,"node"] = df.node.astype(int)

        np.random.seed(11111)
        rvals = np.random.random(df.shape[0]) + 36
        with open(os.path.join(new_d, "test.adj"), 'w') as f:

            f.write("begin performance_measure pm1 type residual\n")
            for rval, lay,node in zip(rvals, df.layer,df.node):
                for kper in range(25):
                    f.write("{0} 1 {1} {2} 1.0  {3}\n".format(kper + 1, lay, node, rval))
            f.write("end performance_measure\n\n")

        start = datetime.now()
        os.chdir(new_d)
        adj = mf6adj.Mf6Adj("test.adj", os.path.split(local_lib_name)[1], verbose_level=2)
        adj.solve_gwf()
        adj.solve_adjoint()
        adj.finalize()
        os.chdir("..")
        duration = (datetime.now() - start).total_seconds()
        print("took:", duration)

    result_hdf = [f for f in os.listdir(new_d) if f.endswith("hd5") and f.startswith("adj_pm")]

    result_hdf.sort()
    print(result_hdf)
    result_hdf = result_hdf[-1]
    print("using hdf",result_hdf)
    import h5py
    hdf = h5py.File(os.path.join(new_d, result_hdf), 'r')
    keys = list(hdf.keys())
    keys.sort()
    print(keys)
    import flopy
    sim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join(new_d))
    m = sim.get_model()
    #m.dis.top.plot()
    #plt.show()
    #return
    nplc = m.dis.top.array.shape[0]
    head = hdf["solution_kper:00000_kstp:00000"]["head"][:]
    nlay = int(head.shape[0] / nplc)
    head = head.reshape((nlay,nplc))
    idomain = m.dis.idomain.array.copy()
    idomain = idomain.reshape((nlay,nplc))
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(os.path.join(new_d, "results.pdf")) as pdf:
        for key in keys:
            print(key)
            if key != "composite":
                continue
            grp = hdf[key]
            plot_keys = [i for i in grp.keys()]

            for pkey in plot_keys:
                print(pkey)
                #if "k11" not in pkey:
                #    continue

                arr = grp[pkey][:]
                nlay = int(arr.shape[0] / nplc)
                arr = arr.reshape((nlay,nplc))#.transpose()
                for k, karr in enumerate(arr):
                    print(karr)
                    karr[idomain[k]==0] = np.nan
                    print(np.nanmin(karr),np.nanmax(karr))
                    #fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    m.dis.top = karr#np.log10(np.abs(karr))
                    ax = m.dis.top.plot()
                    h = head[k].copy()
                    h[idomain[k] == 0] = np.nan
                    m.dis.top = h
                    print(np.nanmin(h),np.nanmax(h))
                    m.dis.top.plot()
                    print(h)
                    plt.show()
                    return
                #break

if __name__ == "__main__":
    test_xd_box_unstruct_1()
    #test_xd_box_1()
    #freyberg_structured_demo()
    #freyberg_quadtree_demo()
    #basic_freyberg()
    #twod_ss_hetero_head_at_point()
    #twod_ss_nested_hetero_head_at_point()
    #_skip_for_now_freyberg()
    #test_freyberg_mh()

    #test_3d_freyberg()
    #test_freyberg_unstruct()
    #test_zaidel()
    #plot_freyberg_verbose_structured_output("freyberg_mf6_pestppv5_test")
    #plot_freyberg_verbose_structured_output("freyberg_mh_adj_test")

