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
    gg_bin = os.path.join("..", "bin", "linux", "gridgen")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() and "arm" not in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
    local_lib_name = "./libmf6.dylib"
    local_mf6_bin = "./mf6"
    gg_bin = os.path.join("..", "bin", "mac", "gridgen")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower() and "arm" in platform.platform().lower():
    lib_name = os.path.join("..", "bin", "mac", "libmf6_arm.dylib")
    mf6_bin = os.path.join("..", "bin", "mac", "mf6")
    local_lib_name = "./libmf6_arm.dylib"
    local_mf6_bin = "./mf6"
    gg_bin = os.path.join("..", "bin", "mac", "gridgen")
else:
    lib_name = os.path.join("..", "bin", "win", "libmf6.dll")
    mf6_bin = os.path.join("..", "bin", "win", "mf6.exe")
    local_lib_name = "libmf6.dll"
    local_mf6_bin = "mf6.exe"
    gg_bin = os.path.join("..", "bin", "win", "gridgen.exe")


def setup_xd_box_model(new_d,sp_len=1.0,nper=1,hk=1.0,k33=1.0,q=-0.1,ss=1.0e-5,
                     nlay=1,nrow=10,ncol=10,delrowcol=1.0,icelltype=0,iconvert=0,newton=False,
                     top=1,botm=None,include_sto=True,include_id0=True,name = "freyberg6",
                       full_sat_bnd=True,alt_bnd=None):

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

    ims = flopy.mf6.ModflowIms(sim, pname="ims", complexity="COMPLEX", linear_acceleration="BICGSTAB",
                               inner_dvclose=1e-15, outer_dvclose=1e-15, outer_maximum=1000, inner_maximum=1000)

    model_nam_file = f"{name}.nam"
    newtonoptions = []
    if newton:
        newtonoptions = ["NEWTON"]
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file, save_flows=True,newtonoptions=newtonoptions)

    idm = np.ones((nlay, nrow, ncol))
    if ncol > 1 and nrow > 1 and include_id0:
        for k in range(nlay):
            idm[k, 0, 1] = 0  # just to have one in active cell...
    if ncol > 5 and nrow > 5 and include_id0:
        for k in range(nlay):
            idm[0, 1, 1] = 0
            idm[0, 3, 3] = 0

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
        transient = [True]
        if len(tdis_pd) > 1:
            steady_state = {kper:False for kper in range(len(tdis_pd))}
            steady_state[0] = True
            transient = {kper:True for kper in range(len(tdis_pd))}
            transient[0] = False

        sto = flopy.mf6.ModflowGwfsto(gwf, iconvert=iconvert, steady_state=steady_state,transient=transient,ss=ss,sy=sy)

    alt_rec = []
    if ncol > 1:
        stage = top
        if not full_sat_bnd:
            stage = (top-botm[0])/4.0

        for k in [nlay-1]:
            for i in range(nrow):
                if alt_bnd == "riv":
                    alt_rec.append(((k, i, 0), stage,1000.0,botm[0]))
                elif alt_bnd == "drn":
                    alt_rec.append(((k, i, 0), stage, 1000.0))
                #elif alt_bnd == "chd":
                #    alt_rec.append(((k, i, 0), stage))
                elif alt_bnd is None:
                    alt_rec.append(((k, i, 0), stage, 1000.0))
                else:
                    raise Exception()

        #chd = flopy.mf6.ModflowGwfghb(gwf, stress_period_data=chd_rec)
        if alt_bnd == "riv":
            _ = flopy.mf6.ModflowGwfriv(gwf,stress_period_data={kper:alt_rec for kper in range(nper)})
        elif alt_bnd == "drn":
            _ = flopy.mf6.ModflowGwfdrn(gwf, stress_period_data={kper: alt_rec for kper in range(nper)})
        #elif alt_bnd == "chd":
        #    _ = flopy.mf6.ModflowGwfchd(gwf, stress_period_data={kper: alt_rec for kper in range(nper)})
        elif alt_bnd is None:
            pass
        else:
            raise Exception()

    if alt_bnd is None:
        ghb_rec = alt_rec
    else:
        ghb_rec = []
    ghb_stage = top+1
    if not full_sat_bnd:
        ghb_stage = 3.*top/4.
    for k in [0]:
        for i in range(nrow):
            ghb_rec.append(((k, i, ncol - 1), ghb_stage, 1000.0))
    ghb_spd = {kper:ghb_rec for kper in range(nper)}
    ghb = flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd)

    if nrow > 1 and ncol > 1:
        #wel_rec = [(nlay - 1, int(nrow / 2), int(ncol / 2), q)]
        wspd ={kper:[(nlay - 1, int(nrow / 2), int(ncol / 2), q*(kper+1))] for kper in range(nper)}
        wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wspd)

    #flopy.mf6.ModflowGwfrcha(gwf, recharge=0.0001)

    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = {kper:[("HEAD", "ALL"), ("BUDGET", "ALL")] for kper in range(nper)}
    printrecord = {kper:[("HEAD", "ALL")] for kper in range(nper)}
    oc = flopy.mf6.ModflowGwfoc(gwf, saverecord=saverecord, head_filerecord=head_filerecord,
                                budget_filerecord=budget_filerecord, printrecord=printrecord)



    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=hk, k33=k33)

    rech = {kper:np.zeros((nrow,ncol))+0.0000001 for kper in range(nper)}
    rch = flopy.mf6.ModflowGwfrcha(gwf,recharge=rech)

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
    #nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data

    adj_summary_files = [os.path.join(new_d,f) for f in os.listdir(new_d) if f.startswith("adjoint_summary") and f.endswith(".csv")]

    #if nlay == 1:
    #    pm_files = [f for f in pm_files if "k33" not in f]
    assert len(adj_summary_files) > 0

    # temp filter
    #pm_files = [f for f in pm_files if "ghb" not in f and "wel" not in f]
    #pm_files.sort()
    #pert_files = [f.replace("pm-","pert-") for f in pm_files]
    pert_summary = pd.read_csv(os.path.join(new_d,"pert_results.csv"),index_col=0)
    for col in ["k","i","j"]:
        if col in pert_summary:
            pert_summary[col] = pert_summary[col].astype(int)
    if plot_compare:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(os.path.join(new_d,"compare.pdf"))

    skip = ["epsilon","k","i","j","addr"]
    pm_names = [c for c in pert_summary.columns if c not in skip]
    pm_names.sort()

    for pm_name in pm_names:
        adj_file = [f for f in adj_summary_files if pm_name in f]
        if len(adj_file) != 1:
            print(pm_name,adj_file)
            raise Exception()
        adj_file = adj_file[0]
        #print(adj_file,pm_name)
        adj = pd.read_csv(adj_file,index_col=0)
        adj_cols = adj.columns.tolist()
        adj_cols.sort()
        for col in adj_cols:
            #try to find the values in addr
            pertdf = pert_summary.loc[pert_summary.addr.str.contains(col),:].copy()
            if pertdf.shape[0] == 0:
                print("...WARNING: no values to compare for pm {0} and parameter {1}".format(pm_name,col))
                continue
            #print(pm_name,col)
            adjdf = adj.loc[pertdf.index.values,col].copy()

            dif = pertdf.loc[:,pm_name].values - adjdf.values
            print(pm_name, col, pertdf.shape[0], adjdf.shape[0])

            demon = (pertdf[pm_name].max() - pertdf[pm_name].min())

            #print(pm_name,col,demon,pertdf[pm_name].max(),pertdf[pm_name].min())
            if demon < 0.0:
                raise Exception()
            elif demon == 0:
                pertdf[pm_name].max()
                #demon = plt_zero_thres
            #abs_max_dif_percent = 100. * np.abs(dif)/max(np.abs(pertdf[pm_name].values).max(),np.abs(adjdf.values).max())
            if demon == 0:
                demon = 1.0
            abs_max_dif_percent = 100. * np.abs(dif) / demon

            print("...min vals:",pertdf[pm_name].values.min(),adjdf.values.min())
            print("...max vals:",pertdf[pm_name].values.max(), adjdf.values.max())
            print("...dif vals",dif.min(),dif.max(),np.abs(dif).min(),np.abs(dif).max())
            print("...abs max % dif:",np.nanmax(abs_max_dif_percent))
            #if not np.isinf(np.nanmax(abs_max_dif_percent)):
            #    assert np.nanmax(abs_max_dif_percent) < 10.0 #?
            if plot_compare and "i" in pertdf.columns and "j" in pertdf.columns:
                kvals =pertdf.k.unique()
                kvals.sort()
                pertdf.loc[:,"dif"] = dif
                pertdf.loc[:,"absdif"] = abs_max_dif_percent
                for k in kvals:
                    kpertdf = pertdf.loc[pertdf.k==k,:]
                    kadjdf = adjdf.loc[kpertdf.index]

                    adj_arr = np.zeros((gwf.dis.nrow.data,gwf.dis.ncol.data)) - 1e30
                    pert_arr = np.zeros_like(adj_arr) - 1e30
                    adj_arr[kpertdf.i,kpertdf.j] = kadjdf.values
                    pert_arr[kpertdf.i, kpertdf.j] = kpertdf[pm_name].values
                    dif_arr = np.zeros_like(adj_arr) - 1.0e30
                    dif_arr[kpertdf.i, kpertdf.j] = kpertdf.dif.values

                    abs_arr = np.zeros_like(adj_arr) - 1.0e30
                    abs_arr[kpertdf.i, kpertdf.j] = kpertdf.absdif.values

                    for arr in [adj_arr,pert_arr,dif_arr,abs_arr]:
                        arr[arr==-1e+30] = np.nan

                    fig,axes = plt.subplots(1,4,figsize=(12,2))

                    absd = np.abs(dif_arr)
                    absp = np.abs(abs_arr)

                    #absd[absd<plt_zero_thres] = 0
                    #pert_arr[absd==0] = np.nan
                    #pm_arr[absd==0] = np.nan
                    dif_arr[absd==0] = np.nan
                    abs_arr[absd==0] = np.nan
                    mx = max(np.nanmax(pert_arr),np.nanmax(adj_arr))
                    mn = min(np.nanmin(pert_arr), np.nanmin(adj_arr))
                    cb = axes[0].imshow(pert_arr,vmax=mx,vmin=mn)
                    plt.colorbar(cb,ax=axes[0])
                    axes[0].set_title(pm_name+" "+col+" pert k:"+str(k),loc="left")


                    cb = axes[1].imshow(adj_arr, vmax=mx, vmin=mn)
                    plt.colorbar(cb, ax=axes[1])
                    axes[1].set_title(pm_name+" "+col+" adj k:"+str(k),loc="left")
                    mx = np.nanmax(absd)

                    cb = axes[2].imshow(dif_arr,vmin=-mx,vmax=mx,cmap="coolwarm")
                    plt.colorbar(cb,ax=axes[2])
                    axes[2].set_title("pert - adj, not showing abs(diff) <= {0}".format(plt_zero_thres),loc="left")
                    mx = np.nanmax(absp)
                    cb = axes[3].imshow(abs_arr,vmin=-mx,vmax=mx,cmap="coolwarm")
                    plt.colorbar(cb, ax=axes[3])
                    axes[3].set_title("percent diff pert - adj, max:{0:4.2g}".format(np.nanmax(abs_arr)),loc="left")
                    if np.any(id==0):
                        idp = id[k,:,:].copy().astype(float)
                        idp[idp!=0] = np.nan
                        axes[0].imshow(idp,cmap="magma")
                        axes[1].imshow(idp, cmap="magma")
                    for ax in axes.flatten():
                        ax.set_xticks([])
                        ax.set_yticks([])
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

    include_ghb_flux_pm = True

    clean = True # run the pertbuation process
    run_pert = False # the pertubations
    plot_pert_results = True #plot the pertubation results

    run_adj = True
    plot_adj_results = False # plot adj result

    plot_compare = False
    new_d = 'xd_box_1_test'
    nrow = ncol = 5
    nlay = 2
    nper = 2
    delrowcol = 1

    botm = None#[-10,-100,-150]
    if clean:
        sim = setup_xd_box_model(new_d, nper=nper,include_sto=include_sto, include_id0=include_id0, nrow=nrow, ncol=ncol,
                                 nlay=nlay,q=-1, icelltype=1, iconvert=1, newton=True, delrowcol=delrowcol,
                                 full_sat_bnd=False,botm=botm,alt_bnd="drn")
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay,nrow,ncol = gwf.dis.nlay.data,gwf.dis.nrow.data,gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.001
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

        with open("test.adj",'w') as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = "direct_pk{1:03d}_pi{2:03d}_pj{3:03d}".format("0", k, i, j)
                f.write("begin performance_measure {0}\n".format(pm_name))
                for kper in range(sim.tdis.nper.data):
                    f.write("{0} 1 {1} {2} {3} head direct {4} -1e+30\n".format(kper + 1, k + 1, i + 1, j + 1, weight))
                f.write("end performance_measure\n\n")

                pm_name = "phi_pk{1:03d}_pi{2:03d}_pj{3:03d}".format("0",k,i,j)
                f.write("begin performance_measure {0}\n".format(pm_name))
                for kper in range(sim.tdis.nper.data):
                    f.write("{0} 1 {1} {2} {3} head residual {4} {5}\n".format(kper+1,k+1,i+1,j+1,weight,obsval))
                f.write("end performance_measure\n\n")

            if include_ghb_flux_pm:
                ghb = gwf.get_package("ghb_0").stress_period_data.array[0]
                for k in range(nlay):
                    kijs = [g[0] for g in ghb if g[0][0] == k]
                    #print(kijs)
                    if len(kijs) > 0:
                        pm_name = "ghb_0_k{0}_direct".format(k)
                        f.write("begin performance_measure {0}\n".format(pm_name))
                        for kper in range(sim.tdis.nper.data):
                            for k,i,j in kijs:
                                f.write("{0} 1 {1} {2} {3} ghb_0 direct 1.0 -1.0e+30\n".format(kper+1,k+1,i+1,j+1))
                        f.write("end performance_measure\n\n")

        adj = mf6adj.Mf6Adj("test.adj", local_lib_name,verbose_level=1)
        adj.solve_gwf()
        adj.solve_adjoint()
        adj._perturbation_test(pert_mult=pert_mult)
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

    #if run_pert:
    #    run_xd_box_pert(new_d,p_kijs,plot_pert_results,weight,pert_mult,obsval=obsval,pm_locs=pm_locs)

    xd_box_compare(new_d,plot_compare)
    return new_d


def test_xd_box_unstruct_1():

    # workflow flags
    include_id0 = True  # include an idomain = cell - has to be false for unstructured...
    include_sto = True

    clean = True # run the pertbuation process
    #run_pert = False # the pertubations
    #plot_pert_results = True #plot the pertubation results

    run_adj = True
    plot_adj_results = False # plot adj result

    #plot_compare = True

    new_d = 'xd_box_1_unstruct_test'
    nrow = ncol = 5
    nlay = 2
    nper = 2
    if clean:
        sim = setup_xd_box_model(new_d, nper=nper, include_sto=include_sto, include_id0=include_id0, nrow=nrow,
                                 ncol=ncol,
                                 nlay=nlay, q=-1, icelltype=1, iconvert=0, newton=True, delrowcol=100.0,
                                 full_sat_bnd=False)
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    obsval = 1.0
    shutil.copy2(gg_bin,os.path.join(new_d,os.path.split(gg_bin)[1]))

    pert_mult = 1.0001
    weight = 1.0
    xcc, ycc = np.atleast_2d(gwf.modelgrid.xcellcenters),np.atleast_2d(gwf.modelgrid.ycellcenters)

    from flopy.utils.gridgen import Gridgen
    g = Gridgen(gwf.dis, model_ws=new_d, exe_name=os.path.join(new_d,os.path.split(gg_bin)[1]))
    g.build(verbose=True)
    gridprops = g.get_gridprops_disv()
    idomain = gwf.dis.idomain.array.copy()

    disv_idomain = []
    for k in range(gwf.dis.nlay.data):
        disv_idomain.append(idomain[k].flatten())
    gwf.remove_package("dis")
    disv = flopy.mf6.ModflowGwfdisv(gwf,idomain=disv_idomain,**gridprops)

    disv.write()

    wel = gwf.get_package("wel")
    if wel is not None:
        f_wel = open(os.path.join(new_d, "freyberg6_disv.wel"), 'w')
        f_wel.write("begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n")

        f_wel.write(
            "begin dimensions\nmaxbound {0}\nend dimensions\n\n".format(wel.stress_period_data.data[0].shape[0]))

        wel_spd = {}
        for kper in range(sim.tdis.nper.data):
            rarray = wel.stress_period_data.data[kper]
            print(rarray.dtype)
            xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
            ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
            ilay = [cid[0] for cid in rarray.cellid]
            xys = [(x, y) for x, y in zip(xs, ys)]
            # use zero for the layer so that we get the cell2d value back
            inodes = [g.intersect([xy], "point", 0)[0][0] for xy, il in zip(xys, ilay)]
            #data = [[(il, inode), bhead, cond] for il, inode, bhead, cond in
            #        zip(ilay, inodes, rarray.flux)]
            #wel_spd[kper] = data
            f_wel.write("begin period {0}\n".format(kper + 1))
            [f_wel.write("{0:9d} {1:9d} {2:15.6E}\n".format(il + 1, inode + 1, q)) for
             il, inode, q in
             zip(ilay, inodes, rarray.q)]
            f_wel.write("end period {0}\n\n".format(kper + 1))
        f_wel.close()

    ghb = gwf.get_package("ghb")
    if ghb is not None:
        f_ghb = open(os.path.join(new_d, "freyberg6_disv.ghb"), 'w')
        f_ghb.write("begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n")

        f_ghb.write(
            "begin dimensions\nmaxbound {0}\nend dimensions\n\n".format(ghb.stress_period_data.data[0].shape[0]))

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
            elif "wel" in line.lower():
                line = "WEL6 freyberg6_disv.wel wel\n"
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


        # with open("test.adj",'w') as f:
        #     f.write("\nbegin options\n\nend options\n\n")
        #     for kper in range(sim.tdis.nper.data):
        #         for p_kinode in pm_locs:
        #             k,inode = p_kinode
        #             pm_name = "direct_kper{0:03d}_pk{1:03d}_pinode{2:03d}".format(kper,k,inode)
        #             f.write("begin performance_measure {0} type direct\n".format(pm_name))
        #             f.write("{0} 1 {1} {2} {3} \n".format(kper+1,k+1,inode+1,weight))
        #             f.write("end performance_measure\n\n")
        #
        #             pm_name = "phi_kper{0:03d}_pk{1:03d}_pinode{2:03d}".format(kper, k, inode)
        #             f.write("begin performance_measure {0} type residual\n".format(pm_name))
        #             f.write("{0} 1 {1} {2} {3} {4}\n".format(kper+1,k+1,inode+1,obsval,weight))
        #             f.write("end performance_measure\n\n")

        with open("test.adj",'w') as f:
            f.write("\nbegin options\n\nend options\n\n")
            for p_kinode in pm_locs:
                k, inode = p_kinode
                pm_name = "direct_pk{0:03d}_pinode{1:03d}".format(k, inode)
                f.write("begin performance_measure {0}\n".format(pm_name))
                for kper in range(sim.tdis.nper.data):
                    f.write("{0} 1 {1} {2} head direct {3} -1.0e+30\n".format(kper+1,k+1,inode+1,weight))
                f.write("end performance_measure\n\n")

                pm_name = "phi_pk{0:03d}_pinode{1:03d}".format(k, inode)
                f.write("begin performance_measure {0}\n".format(pm_name))
                for kper in range(sim.tdis.nper.data):
                    f.write("{0} 1 {1} {2} head direct {3} {4}\n".format(kper+1,k+1,inode+1,weight,obsval))
                f.write("end performance_measure\n\n")

        adj = mf6adj.Mf6Adj("test.adj", local_lib_name, verbose_level=1)
        adj.solve_gwf()
        adj.solve_adjoint()
        adj._perturbation_test(pert_mult=pert_mult)
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
    xd_box_compare(new_d,False)

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

    #os.chdir(new_d)
    #os.system("mf6")
    #os.chdir("..")
    pyemu.os_utils.run("mf6",cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    sfr_data = pd.DataFrame.from_records(gwf.sfr.packagedata.array)


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

        f.write("begin performance_measure pm1\n")
        for rval,lrc in zip(rvals,lrcs):
            for kper in range(sim.tdis.nper.data):
                #f.write("{0} 1 {1} 1.0  {2}\n".format(kper+1,lrc,rval))
                f.write("{0} 1 {1} head direct 1.0 -1.0e+30\n".format(kper + 1, lrc, rval))
        f.write("end performance_measure\n\n")

        bnames = sfr_data.boundname.unique()
        bnames.sort()
        for bname in bnames:
            bdf = sfr_data.loc[sfr_data.boundname==bname,:].copy()

            f.write("begin performance_measure {0}\n".format(bname))
            for kper in range(sim.tdis.nper.data):
                for kij in bdf.cellid.values:
                    f.write("{0} 1 {1} {2} {3} sfr_1 direct 1.0 -1.0e+30\n".format(kper+1,kij[0]+1,kij[1]+1,kij[2]+1))
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

    result_hdf = [f for f in os.listdir(new_d) if f.endswith("hd5") and f.startswith("adjoint_solution_pm1")]
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
    prep_run = True
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

        #os.chdir(new_d)
        #os.system("mf6")
        #os.chdir("..")
        pyemu.os_utils.run("mf6",cwd=new_d)

    if run_adj:
        df = pd.read_csv(os.path.join(new_d,"freyberg6.obs_continuous_heads.csv.txt"),header=None,names=["site","otype","layer","node"])
        df.loc[:,"layer"] = df.layer.astype(int)
        df.loc[:,"node"] = df.node.astype(int)

        np.random.seed(11111)
        rvals = np.random.random(df.shape[0]) + 36
        with open(os.path.join(new_d, "test.adj"), 'w') as f:

            f.write("begin performance_measure pm1\n")
            for rval, lay,node in zip(rvals, df.layer,df.node):
                for kper in range(25):
                    f.write("{0} 1 {1} {2} head residual 1.0  {3}\n".format(kper + 1, lay, node, rval))
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

    result_hdf = [f for f in os.listdir(new_d) if f.endswith("hd5") and f.startswith("adjoint_solution_pm")]

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
                    ax = m.dis.top.plot(colorbar=True)
                    h = head[k].copy()
                    h[idomain[k] == 0] = np.nan
                    m.dis.top = h
                    print(np.nanmin(h),np.nanmax(h))
                    m.dis.top.plot(colorbar=True)
                    print(h)
                    plt.show()
                    return
                #break


def freyberg_structured_highres_demo():
    if os.path.exists('mf6adj'):
        shutil.rmtree('mf6adj')
    # shutil.copytree(os.path.join('..','mf6adj'),os.path.join('mf6adj'))
    sys.path.insert(0, os.path.join(".."))
    import mf6adj

    org_d = "freyberg_highres"
    new_d = "freyberg_highres_test"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)
    shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))
    shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))
    # shutil.copytree(os.path.join('mf6adj'), os.path.join(new_d, 'mf6adj'))

    #os.chdir(new_d)
    #os.system("mf6")
    #os.chdir("..")
    pyemu.os_utils.run("mf6",cwd=new_d)
    # lrcs = []
    # k_dict = {}
    # with open(os.path.join(new_d, "head.obs"), 'r') as f:
    #     f.readline()
    #     for line in f:
    #         if line.strip().lower().startswith("end"):
    #             break
    #         raw = line.strip().split()
    #         lrcs.append(" ".join(raw[2:]))
    #         k = int(raw[2]) - 1
    #         i = int(raw[3]) - 1
    #         j = int(raw[4]) - 1
    #         if k not in k_dict:
    #             k_dict[k] = []
    #         k_dict[k].append([i, j])

    df = pd.read_csv(os.path.join(new_d, "freyberg6.obs_continuous_heads.csv.txt"), header=None,
                     names=["site", "otype", "layer", "row","col"],delim_whitespace=True)
    df.loc[:, "layer"] = df.layer.astype(int)
    df.loc[:, "row"] = df.row.astype(int)
    df.loc[:, "col"] = df.col.astype(int)

    np.random.seed(11111)
    rvals = np.random.random(df.shape[0]) + 36
    with open(os.path.join(new_d, "test.adj"), 'w') as f:

        f.write("begin performance_measure pm1\n")
        for rval, lay, row,col in zip(rvals, df.layer, df.row,df.col):
            for kper in range(25):
                #f.write("{0} 1 {1} {2} {3} 1.0  {4}\n".format(kper + 1, lay, row, col, rval))
                f.write("{0} 1 {1} {2} {3} head direct 1.0 -1.0e+30\n".format(kper + 1, lay, row, col, rval))
        f.write("end performance_measure\n\n")

    # np.random.seed(11111)
    # rvals = np.random.random(len(lrcs)) + 36
    # with open(os.path.join(new_d, "test.adj"), 'w') as f:
    #
    #     f.write("begin performance_measure pm1 type residual\n")
    #     for rval, lrc in zip(rvals, lrcs):
    #         for kper in range(25):
    #             f.write("{0} 1 {1} 1.0  {2}\n".format(kper + 1, lrc, rval))
    #     f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)
    adj = mf6adj.Mf6Adj("test.adj", os.path.split(local_lib_name)[1], verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)

    result_hdf = [f for f in os.listdir(new_d) if f.endswith("hd5") and f.startswith("adjoint_solution_pm1")]
    print(result_hdf)
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]
    import h5py
    hdf = h5py.File(os.path.join(new_d, result_hdf), 'r')
    keys = list(hdf.keys())
    keys.sort()
    print(keys)
    from matplotlib.backends.backend_pdf import PdfPages
    idomain = np.loadtxt(os.path.join(new_d, "freyberg6.dis_idomain_layer1.txt"))
    with PdfPages(os.path.join(new_d, "results.pdf")) as pdf:
        for key in keys:
            if key != "composite":
                continue

            grp = hdf[key]
            plot_keys = [i for i in grp.keys() if len(grp[i].shape) == 3]
            for pkey in plot_keys:

                arr = grp[pkey][:]
                for k, karr in enumerate(arr):
                    karr[idomain < 1] = np.nan
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    cb = ax.imshow(karr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(key + ", " + pkey + ", layer:{0}".format(k + 1), loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...", key, pkey, k + 1)


def freyberg_notional_unstruct_demo():
    if os.path.exists('mf6adj'):
        shutil.rmtree('mf6adj')
    # shutil.copytree(os.path.join('..','mf6adj'),os.path.join('mf6adj'))
    sys.path.insert(0, os.path.join(".."))
    import mf6adj

    org_d = "freyberg_structured"
    new_d = "freyberg_notional_unstructured_test"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)
    shutil.copy2(lib_name, os.path.join(new_d, os.path.split(lib_name)[1]))
    shutil.copy2(mf6_bin, os.path.join(new_d, os.path.split(mf6_bin)[1]))
    shutil.copy2(gg_bin, os.path.join(new_d, os.path.split(gg_bin)[1]))
    shutil.copytree(os.path.join('xmipy'), os.path.join(new_d, 'xmipy'))
    shutil.copytree(os.path.join('bmipy'), os.path.join(new_d, 'bmipy'))
    shutil.copytree(os.path.join('modflowapi'), os.path.join(new_d, 'modflowapi'))
    shutil.copytree(os.path.join('flopy'), os.path.join(new_d, 'flopy'))
    # shutil.copytree(os.path.join('mf6adj'), os.path.join(new_d, 'mf6adj'))

    #os.chdir(new_d)
    #os.system("mf6")
    #os.chdir("..")
    pyemu.os_utils.run("mf6",cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    nlay,nrow,ncol = gwf.dis.nlay.data,gwf.dis.nrow.data,gwf.dis.ncol.data
    xcc, ycc = np.atleast_2d(gwf.modelgrid.xcellcenters), np.atleast_2d(gwf.modelgrid.ycellcenters)

    from flopy.utils.gridgen import Gridgen
    g = Gridgen(gwf.dis, model_ws=new_d, exe_name=os.path.join(new_d,os.path.split(gg_bin)[1]))
    g.build(verbose=True)
    gridprops = g.get_gridprops_disv()
    idomain = gwf.dis.idomain.array.copy()

    disv_idomain = []
    for k in range(gwf.dis.nlay.data):
        disv_idomain.append(idomain[k].flatten())
    gwf.remove_package("dis")
    disv = flopy.mf6.ModflowGwfdisv(gwf, idomain=disv_idomain, **gridprops)

    disv.write()

    sfr = gwf.get_package("sfr")
    #f_wel = open(os.path.join(new_d, "freyberg6_disv.sfr"), 'w')
    #f_wel.write("begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n")
    sfr_pdata = pd.DataFrame.from_records(sfr.packagedata.array)
    print(sfr_pdata)
    sfr_pdata.loc[:,"cellid"] = sfr_pdata.cellid.apply(lambda x: (x[0],x[1]*ncol+x[2]))
    sfr.packagedata = sfr_pdata.to_records(index=False)
    sfr.write()

    wel = gwf.get_package("wel")
    if wel is not None:
        f_wel = open(os.path.join(new_d, "freyberg6_disv.wel"), 'w')
        f_wel.write("begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n")

        f_wel.write(
            "begin dimensions\nmaxbound {0}\nend dimensions\n\n".format(wel.stress_period_data.data[0].shape[0]))

        wel_spd = {}
        for kper in range(sim.tdis.nper.data):
            if kper not in wel.stress_period_data.data:
                continue
            rarray = wel.stress_period_data.data[kper]
            print(rarray.dtype)
            xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
            ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
            ilay = [cid[0] for cid in rarray.cellid]
            xys = [(x, y) for x, y in zip(xs, ys)]
            # use zero for the layer so that we get the cell2d value back
            inodes = [g.intersect([xy], "point", 0)[0][0] for xy, il in zip(xys, ilay)]
            # data = [[(il, inode), bhead, cond] for il, inode, bhead, cond in
            #        zip(ilay, inodes, rarray.flux)]
            # wel_spd[kper] = data
            f_wel.write("begin period {0}\n".format(kper + 1))
            [f_wel.write("{0:9d} {1:9d} {2:15.6E}\n".format(il + 1, inode + 1, q)) for
             il, inode, q in
             zip(ilay, inodes, rarray.q)]
            f_wel.write("end period {0}\n\n".format(kper + 1))
        f_wel.close()

    ghb = gwf.get_package("ghb")
    if ghb is not None:
        f_ghb = open(os.path.join(new_d, "freyberg6_disv.ghb"), 'w')
        f_ghb.write("begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n")

        f_ghb.write(
            "begin dimensions\nmaxbound {0}\nend dimensions\n\n".format(ghb.stress_period_data.data[0].shape[0]))

        ghb_spd = {}
        for kper in range(sim.tdis.nper.data):
            if kper not in ghb.stress_period_data.data:
                continue
            rarray = ghb.stress_period_data.data[kper]
            # print(rarray)
            xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
            ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
            ilay = [cid[0] for cid in rarray.cellid]
            xys = [(x, y) for x, y in zip(xs, ys)]
            # use zero for the layer so that we get the cell2d value back
            inodes = [g.intersect([xy], "point", 0)[0][0] for xy, il in zip(xys, ilay)]
            data = [[(il, inode), bhead, cond] for il, inode, bhead, cond in
                    zip(ilay, inodes, rarray.bhead, rarray.cond)]
            ghb_spd[kper] = data
            f_ghb.write("begin period {0}\n".format(kper + 1))
            [f_ghb.write("{0:9d} {1:9d} {2:15.6E} {3:15.6E}\n".format(il + 1, inode + 1, bhead, cond)) for
             il, inode, bhead, cond in
             zip(ilay, inodes, rarray.bhead, rarray.cond)]
            f_ghb.write("end period {0}\n\n".format(kper + 1))
        f_ghb.close()

    df = pd.read_csv(os.path.join(new_d,"head.obs"),skipfooter=1,skiprows=1,header=None,names=["site","otype","l","r","c"],delim_whitespace=True)
    df.loc[:,"node"] = df.apply(lambda x: (int(x.r)*ncol)+int(x.c),axis=1)
    with open(os.path.join(new_d,"head.obs"),'w') as f:
        f.write("BEGIN CONTINUOUS FILEOUT heads.csv\n")
        for site,otype,lay,node in zip(df.site,df.otype,df.l,df.node):
            f.write("{0} {1} {2} {3}\n".format(site,otype,lay,node))
        f.write("END CONTINUOUS")



    # now hack the nam file
    nam_file = os.path.join(new_d, "freyberg6.nam")
    lines = open(nam_file, 'r').readlines()

    with open(nam_file,'w') as f:
        for line in lines:
            if "dis" in line.lower():
                line = "DISV6 freyberg6.disv disv\n"
            elif "ghb" in line.lower():
                line = "GHB6 freyberg6_disv.ghb ghb\n"
            elif "wel" in line.lower():
                line = "WEL6 freyberg6_disv.wel wel\n"
            f.write(line)

    pyemu.os_utils.run("mf6",cwd=new_d)

    laynode = []
    k_dict = {}
    with open(os.path.join(new_d, "head.obs"), 'r') as f:
        f.readline()
        for line in f:
            if line.strip().lower().startswith("end"):
                break
            raw = line.strip().split()
            laynode.append(" ".join(raw[2:]))

            k = int(raw[2]) - 1
            inode = int(raw[3]) - 1
            if k not in k_dict:
                k_dict[k] = []
            k_dict[k].append([inode])

    np.random.seed(11111)
    rvals = np.random.random(len(laynode)) + 36
    with open(os.path.join(new_d, "test.adj"), 'w') as f:

        f.write("begin performance_measure pm1\n")
        for rval, ln in zip(rvals, laynode):
            for kper in range(25):
                # f.write("{0} 1 {1} 1.0  {2}\n".format(kper+1,lrc,rval))
                f.write("{0} 1 {1} head direct 1.0 {2}\n".format(kper + 1, ln, rval))
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

    result_hdf = [f for f in os.listdir(new_d) if f.endswith("hd5") and f.startswith("adjoint_solution_pm")]
    print(result_hdf)
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]
    import h5py
    hdf = h5py.File(os.path.join(new_d, result_hdf), 'r')
    keys = list(hdf.keys())
    keys.sort()
    print(keys)
    from matplotlib.backends.backend_pdf import PdfPages
    idomain = np.loadtxt(os.path.join(new_d, "freyberg6.dis_idomain_layer1.txt"))
    with PdfPages(os.path.join(new_d, "results.pdf")) as pdf:
        for key in keys:
            if key != "composite":
                continue

            grp = hdf[key]
            plot_keys = [i for i in grp.keys() if len(grp[i]) == nlay*nrow*ncol]
            for pkey in plot_keys:

                arr = grp[pkey][:].reshape((nlay,nrow,ncol))
                for k, karr in enumerate(arr):
                    karr[idomain < 1] = np.nan
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    cb = ax.imshow(karr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(key + ", " + pkey + ", layer:{0}".format(k + 1), loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...", key, pkey, k + 1)

if __name__ == "__main__":
    #test_xd_box_unstruct_1()
    new_d = test_xd_box_1()
    xd_box_compare(new_d,True)

    #freyberg_structured_demo()
    #freyberg_structured_highres_demo()
    #freyberg_notional_unstruct_demo()
    #freyberg_quadtree_demo()

