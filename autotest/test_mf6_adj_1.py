import os
import sys
import platform
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

if "linux" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "linux", "libmf6.so")
    mf6_bin = os.path.join("..", "bin", "linux", "mf6")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
else:
    lib_name = os.path.join("..", "bin", "win", "libmf6.dll")
    mf6_bin = os.path.join("..", "bin", "win", "mf6.exe")

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
    
    org_d = os.path.join("models","freyberg")
    new_d = "freyberg"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)
    shutil.copy2(local_lib_name,os.path.join(new_d,os.path.split(local_lib_name)[1]))
    shutil.copy2(local_mf6_bin,os.path.join(new_d,os.path.split(local_mf6_bin)[1]))

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
    sim.run_simulation()

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
    if "linux" in platform.platform().lower():
        local_lib_name = "libmf6.so"
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
    sim.run_simulation()

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
    os.chdir(new_d)
    import mf6adj

    name = "snglhdtest"
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
    sim.run_simulation()

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
    S_adj = np.loadtxt('k123_layer001.dat')

    # now plot up both results

    diff = np.abs(S_per - S_adj)
    print(diff)
    print(diff.max())
    assert diff.max() < 1.0e-5

    os.chdir('..')

def freyberg_test():
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
    sim.run_simulation()

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
    org_d = "freyberg_mh_adj"
    test_d = "freyberg_mh_adj_test"
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
    adj = mf6adj.Mf6Adj("test.adj", local_lib_name, True,2)
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
    base_d = "freyberg_mh_adj_base"

    if True:
        # run MH's adj code
        
        if os.path.exists(base_d):
            shutil.rmtree(base_d)
        shutil.copytree(org_d,base_d)
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
            k1file = os.path.join(test_d,"pm-pm1_sens_k_kper{0:05d}_k000.dat".format(kper))
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


    a1 = np.loadtxt(os.path.join(test_d,"pm-pm1_dadk123_kper00000.dat"))
    a2 = np.loadtxt(os.path.join(base_d,"dadk123.dat"))
    d = np.abs(a1-a2)
    print(d.max())
    assert d.max() < 1.0e-6

    files = [f for f in os.listdir(test_d) if f.startswith("pm-pm1_dadk123_kper") and "_k0" not in f]
    assert len(files) == 25
    for f in files:
        a1 = np.loadtxt(os.path.join(test_d,f))
        d = np.abs(a1-a2)
        assert d.max() < 1.0e-6

    for kper in range(sim.tdis.nper.data):
        a1 = np.loadtxt(os.path.join(test_d,"pm-pm1_adjstates_kper{0:05d}_k000.dat".format(kper)))
        a2 = np.loadtxt(os.path.join(base_d,"adjstates_kper{0:05d}_k000.dat".format(kper)))
        a1[id==0] = 0
        a2[id==0] = 0
        d = np.abs(a1-a2)
        print(d.max())
        assert d.max() < 1e-6
        

    tags = ["comp_sens_k_k000.dat","comp_sens_k33_k000.dat","comp_sens_ss_k000.dat"]
    
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
    org_d = "freyberg_mf6_pestppv5"
    test_d = "freyberg_mf6_pestppv5"
    if os.path.exists(test_d):
       shutil.rmtree(test_d)
    shutil.copytree(org_d,test_d)

    #write the adj file using the truth values
    import pyemu
    pst = pyemu.Pst(os.path.join(test_d,"truth.pst"))
    res = pst.res
    print(res)
    return

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

    os.chdir(bd)


if __name__ == "__main__":
    #basic_freyberg()
    #twod_ss_hetero_coarsegrid()
    #twod_ss_hetero_head_at_point()
    #twod_ss_nested_hetero_head_at_point()
    #freyberg_test()
    #test_freyberg_mh()
    test_3d_freyberg()


