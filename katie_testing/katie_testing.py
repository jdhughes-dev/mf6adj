import os
import sys
import platform
import shutil
import string
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import modflowapi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0,".")
import flopy
import mf6adj

if "linux" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "linux", "libmf6.so")
    mf6_bin = os.path.join("..", "bin", "linux", "mf6")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
else:
    lib_name = os.path.join("..", "bin", "win", "libmf6.dll")
    mf6_bin = os.path.join("..", "bin", "win", "mf6.exe")

def get_analytical_adj_state(xs, ys, M, N):
    list_adj_state = []
    for j in range(Nrow):
        for i in range(Ncol):
            x = gwf.modelgrid.xcellcenters[j][i]
            y = gwf.modelgrid.ycellcenters[j][i]
            list_adj_state.append(phi_s(x, y, xs, ys, M, N))
    array_adj_state = np.array(list_adj_state)
    array_adj_state_2D = np.reshape(array_adj_state, (Nrow, Ncol))
    return array_adj_state_2D

def get_analytical_adj_state_averaged_head(C, M, N):
    list_adj_state = []
    for j in range(Nrow):
        for i in range(Ncol):
            x = gwf.modelgrid.xcellcenters[j][i]
            y = gwf.modelgrid.ycellcenters[j][i]
            list_adj_state.append(phi_s_Averaged_Head(x, y, C, M, N))
    array_adj_state = np.array(list_adj_state)
    array_adj_state_2D = np.reshape(array_adj_state, (Nrow, Ncol))
    return array_adj_state_2D

def a(C, n):
    return (C * L1 ** 2 / (((n * np.pi) ** 3) * T)) * (((-1) ** n) * (2 - (n * np.pi) ** 2) - 2)

def b(C, n):
    return - (C * L1 ** 2 / (n * np.pi * T)) * (((-1) ** n) - 1)

def alpha(m):
    return m * np.pi / L1

def beta(n):
    return n * np.pi / L2

def omega_square(m, n):
    return (alpha(m)) ** 2 + (beta(n)) ** 2

def phi_s(x, y, xs, ys, M, N):
    listofmindices = [item + 1 for item in range(M)]
    Sum = 0.0
    a = [0.5]
    for i in [item + 1 for item in range(N)]:
        a.append(1.0)
    for n in range(N+1):
        for m in listofmindices:
            Sum = Sum + a[n] * np.sin(alpha(m) * x) * np.cos(beta(n) * y) * np.sin(alpha(m) * xs) * np.cos(beta(n) * ys) / omega_square(m, n)
    Sum = - (4 / (T*D)) * Sum
    return Sum

def phi_s_Averaged_Head(x, y, C, M, N):
    listofmindices = [item + 1 for item in range(M)]
    listofnindices = [item + 1 for item in range(N)]
    Sum = 0.0
    for m in listofmindices:
        for n in listofnindices:
            Sum = Sum + ((-1)**m - 1) * ((-1)**n - 1) * np.sin(alpha(m) * x) * np.sin(beta(n) * y) / (alpha(m) * beta(n) * omega_square(m, n))
    Sum = - (4 * C / (L1 * L2 * T)) * Sum
    return Sum

def dFdh(k, j, i, ncells):
    mylist = np.zeros(ncells)
    m = gwf.modelgrid.get_node((k, j, i))[0]
    for ii in range(ncells):
        if ii == m:
            mylist[ii] = 1.0
    return mylist

def dFdh_avraged_head(ncells):
    mylist = np.zeros(ncells)
    for ii in range(ncells):
        (k, j, i) = gwf.modelgrid.get_lrc(ii)[0]
        if (k, j, i) in list_ch:
            mylist[ii] = 0.0
        else:
            mylist[ii] = CELLAREA[ii]
    return mylist

def J_averaged_head(ncells, X):
    sum = 0.0
    for ii in range(ncells):
        sum += X[ii] * CELLAREA[ii]
    return sum

def J_head(k,i,j, X):
    hd = X[k,i,j]
    return hd

def multiplyarray(A1, A2):
    n = len(A1)
    product = np.zeros(n)
    for i in range(n):
        product[i] = A1[i] * A2[i]
    return product

def substructarray(A1, A2):
    n = len(A1)
    mylist = [item1 - item2 for item1, item2 in zip(A1, A2)]
    return mylist


def SolveAdjointHeadAtPoint(k, j, i, ncells):
    rhs = dFdh(k, j, i, ncells)
    A = csr_matrix((MAT[0], JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
    At = A.transpose()
    lam = spsolve(At, rhs)
    # lam = np.linalg.solve(At, rhs)
    return lam

def SolveAdjointAveragedHead(ncells):
    rhs =  dFdh_avraged_head(ncells)
    A = csr_matrix((MAT[0], JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
    # At = A.transpose()
    lam = spsolve(A, rhs)
    # lam = spsolve(At, rhs)
    # lam = np.linalg.solve(At, rhs)
    return lam

def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
    return - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)

def d_amat_k():

    d_mat_k11 = np.zeros(len(MAT[0]))
    d_mat_k22 = np.zeros(len(MAT[0]))
    d_mat_k33 = np.zeros(len(MAT[0]))
    d_mat_k123 = np.zeros(len(MAT[0]))
    for nn in range(len(IA)-1):
        (k, j, i) = gwf.modelgrid.get_lrc(nn)[0]
        if (k, j, i) in list_ch:
            for ij in range(IAC[nn]-1):
                d_mat_k11[IA_p[nn] + ij] = 0.0
                d_mat_k22[IA_p[nn] + ij] = 0.0
                d_mat_k33[IA_p[nn] + ij] = 0.0
                d_mat_k123[IA_p[nn] + ij] = d_mat_k11[IA_p[nn] + ij] + d_mat_k22[IA_p[nn] + ij] + d_mat_k33[IA_p[nn] + ij]
        else:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            ii = 1
            for nc in range(IAC[nn])[1:]:
                (knc, jnc, inc) = gwf.modelgrid.get_lrc(JA_p[IA_p[nn]+nc])[0]
                if knc == k - 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = derivative_conductance_k1(K33[nn], K33[JA_p[IA_p[nn]+nc]], CELLTOP[nn] - CELLBOT[nn], CELLTOP[JA_p[IA_p[nn]+nc]] - CELLBOT[JA_p[IA_p[nn]+nc]], DELR[i], DELC[j])
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif jnc == j - 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = derivative_conductance_k1(K22[nn], K22[JA_p[IA_p[nn]+nc]], DELC[j], DELC[j-1], DELR[i], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif inc == i - 1:
                    d_mat_k11[IA_p[nn] + ii] = derivative_conductance_k1(K11[nn], K11[JA_p[IA_p[nn]+nc]], DELR[i], DELR[i-1], DELC[j], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif inc == i + 1:
                    d_mat_k11[IA_p[nn] + ii] = derivative_conductance_k1(K11[nn], K11[JA_p[IA_p[nn]+nc]], DELR[i], DELR[i+1], DELC[j], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif jnc == j + 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = derivative_conductance_k1(K22[nn], K22[JA_p[IA_p[nn]+nc]], DELC[j], DELC[j+1], DELR[i], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif knc == k + 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = derivative_conductance_k1(K33[nn], K33[JA_p[IA_p[nn]+nc]], CELLTOP[nn] - CELLBOT[nn], CELLTOP[JA_p[IA_p[nn]+nc]] - CELLBOT[JA_p[IA_p[nn]+nc]], DELR[i], DELC[j])
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
            d_mat_k11[IA_p[nn]] = - sum1
            d_mat_k22[IA_p[nn]] = - sum2
            d_mat_k33[IA_p[nn]] = - sum3
            d_mat_k123[IA_p[nn]] = d_mat_k11[IA_p[nn]] + d_mat_k22[IA_p[nn]] + d_mat_k33[IA_p[nn]]

    return d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k123


def lam_dAdk_h(lam, dAdk, h):
    my_list = []
    for k in range(len(lam)):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(IAC[k]):
            sum1 += dAdk[IA_p[k] + j] * h[JA_p[IA_p[k] + j]]
            # print(dAdk[IA_p[k] + j])
            # print(h[JA_p[IA_p[k] + j]])
        sum1 = lam[k] * sum1

        for j in list(range(IAC[k]))[1:]:
            (kk, jj, ii) = gwf.modelgrid.get_lrc(JA_p[IA_p[k] + j])[0]
            if (kk, jj, ii) in list_ch:
                sum2 += 0.0
            else:
                sum2 += lam[JA_p[IA_p[k] + j]] * dAdk[IA_p[k] + j] * (h[k] - h[JA_p[IA_p[k] + j]])
        sum = sum1 + sum2
        my_list.append(sum)
    return my_list

#now some plotting functions
def plot_contour(x, y, l_anal, l_num, contour_intervals, fname):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    c1 = ax.contour(x, y, l_anal, contour_intervals, colors="black", linestyles='solid')
    c2 = ax.contour(x, y, l_num, contour_intervals, colors="black", linestyles='dashed')
    plt.clabel(c2, fmt="%d")
    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    ax.legend([h1[0], h2[0]], ["Analytical", "Numerical (MF6-ADJ)"],
              bbox_to_anchor=(0.78, 1.16), fontsize=12)
    plt.savefig('{0}'.format(fname))

def plot_colorbar_2plts(x, y, l_anal, l_num, contour_intervals, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    # first subplot
    ax = axes[0]
    ax.set_title("Analytical", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(l_anal, vmin=-.08, vmax=0)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    contours = modelmap.contour_array(
        l_anal,
        levels=contour_intervals,
        colors="white",
    )
    # ax.clabel(contours, fmt="%1.1f")
    plt.clabel(contours, fmt="%d")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)

    # second subplot
    ax = axes[1]
    ax.set_title("Numerical (MF6-ADJ)", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    pa = modelmap.plot_array(l_num, vmin=-.08, vmax=0)
    contours = modelmap.contour_array(
        l_num,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%1.1f")
    cbcb = plt.colorbar(pa, shrink=1.0, ax=ax)
    plt.savefig('{0}'.format(fname))

def plot_colorbar_3plts(x, y, l_anal, l_num,lnum2, contour_intervals, fname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    # first subplot
    ax = axes[0]
    ax.set_title("Analytical", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(l_anal)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    contours = modelmap.contour_array(
        l_anal,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%1.1f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)

    # second subplot
    ax = axes[1]
    ax.set_title("Numerical (MF6-ADJ)", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    pa = modelmap.plot_array(l_num)
    contours = modelmap.contour_array(
        l_num,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%1.1f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)

    # third subplot
    ax = axes[2]
    ax.set_title("Numerical (mf6adj)", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    pa = modelmap.plot_array(l_num2)
    contours = modelmap.contour_array(
        l_num2,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%1.1f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)
    plt.savefig('{0}'.format(fname))

def plot_colorbar_sensitivity(x, y, Sadj, Sper,S_jdub, contour_intervals, fname):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    # first subplot
    ax = axes[0]
    ax.set_title("Mohamed-ADJ", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(Sadj)
    # quadmesh = modelmap.plot_bc("CHD")
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    contours = modelmap.contour_array(
        Sadj,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%2.2f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)

    # second subplot
    ax = axes[1]
    ax.set_title("Perturbation", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    pa = modelmap.plot_array(Sper)
    # quadmesh = modelmap.plot_bc("CHD")
    contours = modelmap.contour_array(
        Sper,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%2.2f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)

    # third subplot
    ax = axes[2]
    ax.set_title("MF6-ADJ", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(S_jdub)
    # quadmesh = modelmap.plot_bc("CHD")
    linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
    contours = modelmap.contour_array(
        S_jdub,
        levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%2.2f")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)
    plt.savefig('{0}'.format(fname))

def twod_ss_homo_finegrid():
    global name
    global N
    global L
    global L1
    global L2
    global T
    global D
    global Nlay
    global Nrow
    global Ncol
    global epsilon

    name = "2dsshomofgr"
    epsilon = 1.0e-05
    tf = 1.0
    h1 = 100
    N = 100
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

    #first set up model for analytical solution comparison
    # ### Create the FloPy simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=mf6_bin,
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
    global gwf
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )

    # Now that the overall simulation is set up, we can focus on building the
    # groundwater flow model. The groundwater flow model will be built by
    # adding packages to it that describe the model characteristics.

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
        # steady_state={0: False},
        # transient={0: True}
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
    ra
    global list_ch
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
    array_cond_3D = 10 * np.ones((Nlay, Nrow, Ncol))
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=array_cond_3D,
        # alternative_cell_averaging='logarithmic'
        # alternative_cell_averaging='amt-lmk',
        # alternative_cell_averaging='amt-hmk',
    )

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    sim.run_simulation()

    #now run with API
    mf6api = modflowapi.ModflowApi(lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
    global CELLAREA
    global IA_p
    global JA_p
    global IA
    global MAT
    global IAC
    CELLAREA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("AREA","%s/DIS" % name)))
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

    # print('JA = ', JA)
    # print('IA = ', IA)
    # print('IAC = ', IAC)

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
                    # print('****************************************************************')
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

    #then calculate analytical solution for head integral
    print('now calculating analytical adjoint state')
    lam_anal = get_analytical_adj_state_averaged_head(1.0, 50, 50)

    #then calculate mohamed solution for head integral
    print('now calculating mohamed adjoint state')
    list_AS = SolveAdjointAveragedHead(len(IA) - 1)

    #now make the comparison plot and save
    lam_3d = np.reshape(list_AS, (Nlay, Nrow, Ncol))
    x = np.linspace(0, L1, Ncol)
    y = np.linspace(0, L2, Nrow)
    y = y[::-1]
    minval = min(list_AS)
    maxval = max(list_AS)
    contour_intervals = np.linspace(minval, maxval, 10)
    plot_contour(x, y, lam_anal, lam_3d[0], contour_intervals, '{0}_contour.png'.format(name))
    plot_colorbar_2plts(x, y, lam_anal, lam_3d[0], contour_intervals, '{0}_colorbar.png'.format(name))

def twod_ss_hetero_coarsegrid():
    global name
    global N
    global L
    global L1
    global L2
    global T
    global D
    global Nlay
    global Nrow
    global Ncol
    global epsilon

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
        exe_name=mf6_bin,
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
    global gwf
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
    ra
    global list_ch
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

    kkk = np.array(array_cond_3D)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=kkk,
    )

    list_triplets = []
    for nn in range(Nlay * Nrow * Ncol):
        list_triplets.append(gwf.modelgrid.get_lrc(nn))

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    sim.run_simulation()

    # now run with API
    mf6api = modflowapi.ModflowApi(lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
    global CELLAREA
    global IA_p
    global JA_p
    global IA
    global MAT
    global IAC
    global K22
    global K11
    global DELC
    global DELR
    global CELLTOP
    global CELLBOT
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

    # print('JA = ', JA)
    # print('IA = ', IA)
    # print('IAC = ', IAC)

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

    h = gwf.output.head().get_alldata()[-1]
    hh = np.reshape(h[0], (Nlay * Nrow * Ncol))

    # then calculate mohamed solution for head integral
    print('now calculating mohamed adjoint state')
    lam = SolveAdjointAveragedHead(len(IA) - 1)
    print('lam = ', lam)

    d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k123 = d_amat_k()
    print('dJdk = ', lam_dAdk_h(lam, d_mat_k123, hh))

    print('{:.40f}'.format(J_averaged_head(len(IA) - 1, hh)))
    list_S_adj = lam_dAdk_h(lam, d_mat_k123, hh)

    # then calculate analytical solution for head integral
    print('now calculating perturbation sensitivity')
    count = 0
    for index_sens in list_triplets[0:105]:
        count += 1
        kkk = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            # print(f_sens.write('{:2.4E}\n'.format(0.0)))
            pass
        else:
            kkk[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=1,
                k=kkk,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(lib_name)
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
                sum += hh[ii] * CELLAREA[ii]
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
        kkk = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            print(f_sens.write('{:2.4E}\n'.format(0.0)))
            list_S_per.append(0.)
        else:
            kkk[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=1,
                k=kkk,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(lib_name)
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
                sum += hh[ii] * CELLAREA[ii]
            J = sum

            sens = (J - J_constant) / epsilon
            list_S_per.append(sens)
            print(f_sens.write('{:2.4E}\n'.format(sens)))
    f_sens.close()

    array_S_adj = np.array(list_S_adj)
    array_S_per = np.array(list_S_per)
    S_adj = np.reshape(array_S_adj, (Nlay, Nrow, Ncol))
    S_per = np.reshape(array_S_per, (Nlay, Nrow, Ncol))

    x = np.linspace(0, L1, Ncol)
    y = np.linspace(0, L2, Nrow)
    y = y[::-1]
    minval = min(list_S_per)
    maxval = max(list_S_per)
    contour_intervals = np.linspace(minval, maxval, 10)
    plot_colorbar_sensitivity(x, y, S_adj, S_per, contour_intervals, '2dssheterocgr_colorbar_sensitivity.png')

    f = open("sensitivity.dat", "w")
    print(f.write('  MF6-ADJ  Perturbation\n'))
    print(f.write('-----------------------\n'))
    for i in range(len(list_S_adj)):
        print(f.write('{:2.4E} '.format(list_S_adj[i])))
        print(f.write('{:2.4E}\n'.format(list_S_per[i])))
    f.close()

def twod_ss_homo_head_at_point():
    global name
    global N
    global L
    global L1
    global L2
    global T
    global D
    global Nlay
    global Nrow
    global Ncol
    global epsilon

    name = "snglhdtest"
    epsilon = 1.0e-05
    tf = 1.0
    h1 = 100
    N = 10
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
        exe_name=mf6_bin,
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
    global gwf
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )
    #
    # # ### Create the storage (`STO`) Package
    # sto = flopy.mf6.ModflowGwfsto(
    #     gwf,
    #     steady_state={0: True},
    #     ss_confined_only=True,
    # )

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
    ra
    global list_ch
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
    array_cond_3D = 10 * np.ones((Nlay, Nrow, Ncol))
    kkk= array_cond_3D
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=array_cond_3D,
    )

    list_triplets = []
    for nn in range(Nlay * Nrow * Ncol):
        list_triplets.append(gwf.modelgrid.get_lrc(nn))

    # # ### Write the datasets and run to make sure it works
    sim.write_simulation()
    sim.run_simulation()

    # now run with API
    mf6api = modflowapi.ModflowApi(lib_name)
    mf6api.initialize()
    current_time = mf6api.get_current_time()
    end_time = mf6api.get_end_time()
    max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
    global CELLAREA
    global IA_p
    global JA_p
    global IA
    global MAT
    global IAC
    global K22
    global K11
    global DELC
    global DELR
    global CELLTOP
    global CELLBOT
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

    # print('JA = ', JA)
    # print('IA = ', IA)
    # print('IAC = ', IAC)

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


    # # then calculate analytical solution for head at point
    # print('now calculating analytical adjoint state')
    # lam_anal = get_analytical_adj_state(int(L / 2), int(L / 2), 50, 50)

    # then calculate mohamed solution for head at point
    print('now calculating mohamed adjoint state')
    lam = SolveAdjointHeadAtPoint(0,int(N / 2),int(N / 2)-1,len(IA) - 1)
    # print(lam)

    # # now make the comparison plot and save
    # lam_3d = np.reshape(lam, (Nlay, Nrow, Ncol))
    # x = np.linspace(0, L1, Ncol)
    # y = np.linspace(0, L2, Nrow)
    # y = y[::-1]
    # minval = min(lam)
    # maxval = max(lam)
    # contour_intervals = np.linspace(minval, maxval, 10)
    # plot_contour(x, y, lam_anal, lam_3d[0], contour_intervals, '{0}_contour.png'.format(name))
    # plot_colorbar_2plts(x, y, lam_anal, lam_3d[0], contour_intervals, '{0}_colorbar.png'.format(name))

    h2 = gwf.output.head().get_alldata()[-1]
    hh = np.reshape(h2[0], (Nlay * Nrow * Ncol))
    d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k123 = d_amat_k()
    # print(d_mat_k11)
    # print('dJdk = ', lam_dAdk_h(lam, d_mat_k123, hh))
    # print('{:.40f}'.format(J_head(0,49,49, h2)))

    list_S_adj = lam_dAdk_h(lam, d_mat_k123, hh)

    # then calculate perturbation for head at point
    print('now calculating perturbation sensitivity')

    count = 0
    for index_sens in list_triplets[0:105]:
        count += 1
        kkk = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            # print(f_sens.write('{:2.4E}\n'.format(0.0)))
            pass
        else:
            kkk[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=1,
                k=kkk,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(lib_name)
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
            J_constant = h2[0,int(N / 2),int(N / 2)-1]
            print(J_constant)
            break
    # now set epsilon
    f_sens = open("sens_per_single_head.dat", "w")
    list_S_per = []
    epsilon = .1
    count = 0
    for index_sens in list_triplets:
        count += 1
        kkk = np.array(array_cond_3D)
        if index_sens[0] in list_ch:
            print(f_sens.write('{:2.4E}\n'.format(0.0)))
            list_S_per.append(0.)
        else:
            kkk[index_sens[0][0]][index_sens[0][1]][index_sens[0][2]] += epsilon
            npf = flopy.mf6.ModflowGwfnpf(
                gwf,
                icelltype=1,
                k=kkk,
            )

            # # ### Write the datasets
            sim.write_simulation()

            # ### API-----------------------------------------------------------------
            mf6api = modflowapi.ModflowApi(lib_name)
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
            J = h2[0,int(N / 2),int(N / 2)-1]
            sens = (J - J_constant) / epsilon
            list_S_per.append(sens)
            print(f_sens.write('{:2.4E}\n'.format(sens)))
    f_sens.close()

    # then calculate mfadj for head at point
    print('now calculating mfadj sensitivity from jeremy script')

    with open("test.adj",'w') as f:
        f.write("\nbegin options\n\nend options\n\n")
        f.write("begin performance_measure pm1 type direct\n")
        f.write("1 1 1 {0} {1} 1.0 \n".format(int(N / 2)+1,int(N / 2)))
        f.write("end performance_measure\n\n")

    adj = mf6adj.Mf6Adj("test.adj", lib_name)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()

    #now plot up all three results
    array_S_adj = np.array(list_S_adj)
    array_S_per = np.array(list_S_per)
    array_S_jdub = np.loadtxt('result.dat')

    S_adj = np.reshape(array_S_adj, (Nlay, Nrow, Ncol))
    S_per = np.reshape(array_S_per, (Nlay, Nrow, Ncol))
    S_jdub = np.reshape(array_S_jdub, (Nlay, Nrow, Ncol))

    x = np.linspace(0, L1, Ncol)
    y = np.linspace(0, L2, Nrow)
    y = y[::-1]
    minval = min(list_S_per)
    maxval = max(list_S_per)
    contour_intervals = np.linspace(minval, maxval, 5)
    plot_colorbar_sensitivity(x, y, S_adj, S_per,S_jdub, contour_intervals, 'snglhdtest_colorbar_sensitivity.png')

    # f = open("sensitivity.dat", "w")
    # print(f.write('  analytical  MF6-ADJ  mf6adj  Perturbation\n'))
    # print(f.write('-----------------------\n'))
    # for i in range(len(list_S_adj)):
    #     print(f.write('{:2.4E} '.format(list_S_adj[i])))
    #     print(f.write('{:2.4E} '.format(list_S_adj[i])))
    #     print(f.write('{:2.4E} '.format(list_S_adj[i])))
    #     print(f.write('{:2.4E}\n'.format(list_S_per[i])))
    # f.close()

    #then run perturbation method for head integral

    #then calculate mohamed solution for head at a point

    #then run perturbation method for head at a point

    #then run jeremy solution for head at a point


if __name__ == "__main__":
    # twod_ss_homo_finegrid()
    # twod_ss_hetero_coarsegrid()
    twod_ss_homo_head_at_point()