import os
import shutil
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import h5py
import modflowapi
import flopy


class PerfMeasRecord(object):
    def __init__(self, kper, kstp, nnode, k=None, i=None, j=None, weight=None, obsval=None):
        self._kper = int(kper)
        self._kstp = int(kstp)
        self.kperkstp = (self._kper, self._kstp)
        self.nnode = int(nnode)
        self._k = None
        if k is not None:
            self._k = int(k)
        self._i = None
        if i is not None:
            self._i = int(i)
        self._j = None
        if j is not None:
            self._j = int(j)
        self.weight = 1.0
        if weight is not None:
            self.weight = float(weight)
        self.obsval = None
        if obsval is not None:
            self.obsval = float(obsval)


class PerfMeas(object):
    """todo: preprocess all the connectivity in to faster look dict containers,
	including nnode to kij info for structured grids

	todo: convert several class methods to static methods - this might make testing easier

	todo: add a no-data value var to fill empty spots in output arrays.  currently using zero :(

	todo: check that each entry's kperkstp is in the dicts being passed to solve_adjoint()
	
	"""

    def __init__(self, name, type, entries, is_structured, verbose_level=1):
        self._name = name.lower().strip()
        self._type = type.lower().strip()
        self._entries = entries
        self.is_structured = is_structured
        self.verbose_level = int(verbose_level)

    @property
    def name(self):
        return str(self._name)

    @staticmethod
    def has_sto_iconvert(gwf):
        names = [n for n in list(gwf.get_input_var_names()) if "STO" in n and "ICONVERT" in n]
        if len(names) == 0:
            return False
        return True

    def solve_forward(self, head_dict):
        """for testing only"""
        result = 0.0
        for pfr in self._entries:
            if self._type == "direct":
                result += pfr.weight * head_dict[pfr.kperkstp][pfr.nnode]
            elif self._type == "residual":
                result += (pfr.weight * (head_dict[pfr.kperkstp][pfr.nnode] - pfr.obsval)) ** 2
        return result

    # def solve_adjoint_old(self, kperkstp, iss, deltat_dict, amat_dict, head_dict, head_old_dict,
    #                       sat_dict, sat_old_dict, gwf, gwf_name, mg_structured, gwf_package_dict):
    #     """
    #
    #
    # 	"""
    #     nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
    #
    #     lamb = np.zeros(nnodes)
    #
    #     ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
    #     ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
    #
    #     comp_k33_sens = np.zeros(nnodes)
    #     comp_k_sens = np.zeros(nnodes)
    #
    #     comp_ss_sens = None
    #
    #     has_sto = PerfMeas.has_sto_iconvert(gwf)
    #     if has_sto:
    #         comp_ss_sens = np.zeros(nnodes)
    #
    #     comp_welq_sens = None
    #     comp_ghb_head_sens = None
    #     comp_ghb_cond_sens = None
    #     comp_rch_sens = None
    #
    #     if "wel6" in gwf_package_dict:
    #         comp_welq_sens = np.zeros(nnodes)
    #     if "ghb6" in gwf_package_dict:
    #         comp_ghb_head_sens = np.zeros(nnodes)
    #         comp_ghb_cond_sens = np.zeros(nnodes)
    #     if "rch6" in gwf_package_dict or "rcha6" in gwf_package_dict:
    #         comp_rch_sens = np.zeros((nnodes))
    #
    #     data = {}
    #     for itime, kk in enumerate(kperkstp[::-1]):
    #         itime = kk[0]
    #         print('solving', self._name, "(kper,kstp)", kk)
    #         dfdh = self._dfdh_old(kk, gwf_name, gwf, head_dict)
    #         # jwhite: I think it should be sat old since the head (and therefore sat) from the last
    #         # timestep/stress period is used to scale T...
    #         dadk11, dadk33 = self._dadk(gwf_name, gwf, sat_dict[kk], amat_dict[kk])
    #
    #         if iss[kk] == 0:  # transient
    #             # if False:
    #             # get the derv of RHS WRT head
    #             drhsdh = self._drhsdh(gwf_name, gwf, deltat_dict[kk], sat_dict[kk])
    #             rhs = (drhsdh * lamb) - dfdh
    #         else:
    #             rhs = - dfdh
    #         if np.all(rhs == 0.0):
    #             print(
    #                 "WARNING: adjoint solve rhs is all zeros, adjoint states cannot be calculated for {0} at kperkstp {1}".format(
    #                     self._name, kk))
    #             continue
    #
    #         amat = amat_dict[kk]
    #         # for ii in range(ia.shape[0]-1):
    #         #	print(ii,ii+1,ia[ii],ia[ii+1],ja[ia[ii]:ia[ii+1]],amat[ja[ia[ii]:ia[ii+1]]])
    #         #	print()
    #         amat_sp = sparse.csr_matrix((amat.copy(), ja.copy(), ia.copy()), shape=(len(ia) - 1, len(ia) - 1))
    #         # amat_sp.eliminate_zeros()
    #         amat_sp_t = amat_sp.transpose()
    #         lamb = spsolve(amat_sp_t, rhs)
    #         if np.any(np.isnan(lamb)):
    #             print("WARNING: nans in adjoint states for pm {0} at kperkstp {1}".format(self._name, kk))
    #             continue
    #
    #         k_sens = self.lam_dAdk_h(gwf_name, gwf, lamb, dadk11, head_dict[kk])
    #         k33_sens = self.lam_dAdk_h(gwf_name, gwf, lamb, dadk33, head_dict[kk])
    #
    #         comp_k_sens += k_sens
    #         comp_k33_sens += k33_sens
    #
    #         if has_sto:
    #             ss_sens = self.sens_ss_indirect(gwf_name, gwf, lamb, head_dict[kk], head_old_dict[kk], deltat_dict[kk],
    #                                             sat_dict[kk], sat_old_dict[kk])
    #             comp_ss_sens += ss_sens
    #
    #         if "wel6" in gwf_package_dict and kk in gwf_package_dict["wel6"]:
    #             sens_welq = self.lam_drhs_dqwel(lamb, gwf_package_dict["wel6"][kk])
    #             if self.verbose_level > 1:
    #                 self.save_array_old("sens_welq_kper{0:05d}".format(itime), sens_welq, gwf_name, gwf, mg_structured)
    #                 data["welq_kper{0:05d}".format(itime)] = sens_welq
    #             comp_welq_sens += sens_welq
    #
    #         if "ghb6" in gwf_package_dict and kk in gwf_package_dict["ghb6"]:
    #             sens_ghb_head, sens_ghb_cond = self.lam_drhs_dghb(lamb, head_dict[kk], gwf_package_dict["ghb6"][kk])
    #             if self.verbose_level > 1:
    #                 self.save_array_old("sens_ghbhead_kper{0:05d}".format(itime), sens_ghb_head, gwf_name, gwf,
    #                                 mg_structured)
    #                 self.save_array_old("sens_ghbcond_kper{0:05d}".format(itime), sens_ghb_cond, gwf_name, gwf,
    #                                 mg_structured)
    #                 data["ghbhead_kper{0:05d}".format(itime)] = sens_ghb_head
    #                 data["ghbcond_kper{0:05d}".format(itime)] = sens_ghb_cond
    #
    #             comp_ghb_head_sens += sens_ghb_head
    #             comp_ghb_cond_sens += sens_ghb_cond
    #         # print("rch" in gwf_package_dict,kk,list(gwf_package_dict["rch6"].keys()))
    #         if "rch6" in gwf_package_dict and kk in gwf_package_dict["rch6"]:
    #             sens_rch = self.rch_sens(gwf_name, gwf, lamb, gwf_package_dict["rch6"][kk])
    #             if self.verbose_level > 1:
    #                 self.save_array_old("sens_rch_kper{0:05d}".format(itime), sens_rch, gwf_name, gwf, mg_structured)
    #                 if "rch_kper{0:05d}".format(itime) in data:
    #                     data["rch_kper{0:05d}".format(itime)] += sens_rch
    #                 else:
    #                     data["rch_kper{0:05d}".format(itime)] = sens_rch
    #
    #             comp_rch_sens += sens_rch
    #
    #         # todo: think about what it would do to have both rch and recha active..for now just accumulating them
    #         if "rcha6" in gwf_package_dict and kk in gwf_package_dict["rcha6"]:
    #             sens_rch = self.rch_sens(gwf_name, gwf, lamb, gwf_package_dict["rcha6"][kk])
    #             if self.verbose_level > 1:
    #                 self.save_array_old("sens_rch_kper{0:05d}".format(itime), sens_rch, gwf_name, gwf, mg_structured)
    #                 if "rch_kper{0:05d}".format(itime) in data:
    #                     data["rch_kper{0:05d}".format(itime)] += sens_rch
    #                 else:
    #                     data["rch_kper{0:05d}".format(itime)] = sens_rch
    #             comp_rch_sens += sens_rch
    #
    #         # this is just temp stuff - will be replaced with a more scalable solution...
    #         if self.verbose_level > 1:
    #             self.save_array_old("adjstates_kper{0:05d}".format(itime), lamb, gwf_name, gwf, mg_structured)
    #             data["adjstates_kper{0:05d}".format(itime)] = lamb
    #             self.save_array_old("sens_k33_kper{0:05d}".format(itime), k33_sens, gwf_name, gwf, mg_structured)
    #             data["sens_k33_kper{0:05d}".format(itime)] = k33_sens
    #             self.save_array_old("sens_k11_kper{0:05d}".format(itime), k_sens, gwf_name, gwf, mg_structured)
    #             data["sens_k11_kper{0:05d}".format(itime)] = k_sens
    #             if has_sto:
    #                 self.save_array_old("sens_ss_kper{0:05d}".format(itime), ss_sens, gwf_name, gwf, mg_structured)
    #                 data["sens_ss_kper{0:05d}".format(itime)] = ss_sens
    #             self.save_array_old("head_kper{0:05d}".format(itime), head_dict[kk], gwf_name, gwf, mg_structured)
    #             data["head_kper{0:05d}".format(itime)] = head_dict[kk]
    #             if self.verbose_level > 2:
    #                 self.save_array_old("dadk11_kper{0:05d}".format(itime), dadk11, gwf_name, gwf, mg_structured)
    #                 self.save_array_old("dadk33_kper{0:05d}".format(itime), dadk33, gwf_name, gwf, mg_structured)
    #                 np.savetxt("pm-{0}_amattodense_kper{1:04d}.dat".format(self._name, itime), amat_sp_t.todense(),
    #                            fmt="%15.6E")
    #                 np.savetxt("pm-{0}_amat_kper{1:04d}.dat".format(self._name, itime), amat, fmt="%15.6E")
    #                 np.savetxt("pm-{0}_rhs_kper{1:04d}.dat".format(self._name, itime), rhs, fmt="%15.6E")
    #                 np.savetxt("pm-{0}_ia_kper{1:04d}.dat".format(self._name, itime), ia)
    #                 np.savetxt("pm-{0}_ja_kper{1:04d}.dat".format(self._name, itime), ja)
    #                 # for arr,tag in zip([dadk11,dadk22,dadk33,dadk123],["dadk11","dadk22","dadk33","dadk123"]):
    #                 for arr, tag in zip([dadk11, dadk33],
    #                                     ["dadk11", "dadk33"]):
    #                     np.savetxt("pm-{0}_{1}_kper{2:05d}.dat".format(self._name, tag, itime), arr, fmt="%15.6E")
    #
    #     data["k11"] = comp_k_sens
    #     data["k33"] = comp_k33_sens
    #     self.save_array_old("comp_sens_k33", comp_k33_sens, gwf_name, gwf, mg_structured)
    #     self.save_array_old("comp_sens_k11", comp_k_sens, gwf_name, gwf, mg_structured)
    #     addr = ["NODEUSER", gwf_name, "DIS"]
    #     wbaddr = gwf.get_var_address(*addr)
    #     nuser = gwf.get_value(wbaddr) - 1
    #     if len(nuser) == 1:
    #         nuser = np.arange(nnodes, dtype=int)
    #     df = pd.DataFrame(data, index=nuser)
    #
    #     if has_sto:
    #         self.save_array_old("comp_sens_ss", comp_ss_sens, gwf_name, gwf, mg_structured)
    #         df.loc[:, "ss"] = comp_ss_sens
    #     if "wel6" in gwf_package_dict:
    #         self.save_array_old("comp_sens_welq", comp_welq_sens, gwf_name, gwf, mg_structured)
    #         df.loc[:, "welq"] = comp_welq_sens
    #     if "ghb6" in gwf_package_dict:
    #         self.save_array_old("comp_sens_ghbhead", comp_ghb_head_sens, gwf_name, gwf, mg_structured)
    #         self.save_array_old("comp_sens_ghbcond", comp_ghb_cond_sens, gwf_name, gwf, mg_structured)
    #         df.loc[:, "ghbhead"] = comp_ghb_head_sens
    #         df.loc[:, "ghbcond"] = comp_ghb_cond_sens
    #
    #     if "rch6" in gwf_package_dict or "rcha6" in gwf_package_dict:
    #         self.save_array_old("comp_sens_rch", comp_rch_sens, gwf_name, gwf, mg_structured)
    #         df.loc[:, "rch"] = comp_rch_sens
    #     df.to_csv("{0}_adj_results.csv".format(self._name))
    #     return df

    def solve_adjoint(self, hdf5_forward_solution_fname, hdf5_adjoint_solution_fname=None):
        """

		"""
        try:
            hdf = h5py.File(hdf5_forward_solution_fname, 'r')
        except Exception as e:
            raise Exception("error opening hdf5 file '{0}' for PerfMeas {1}: {2}". \
                            format(hdf5_forward_solution_fname, self._name, str(e)))
        if hdf5_adjoint_solution_fname is None:
            pth = os.path.split(hdf5_forward_solution_fname)[0]
            hdf5_adjoint_solution_fname = os.path.join(pth, "adj_{0}_".format(self._name) + hdf5_forward_solution_fname)

        if os.path.exists(hdf5_adjoint_solution_fname):
            print("WARNING: removing existing adjoint solution file '{0}'".format(hdf5_adjoint_solution_fname))
            os.remove(hdf5_adjoint_solution_fname)

        adf = h5py.File(hdf5_adjoint_solution_fname, "w")

        keys = list(hdf.keys())
        gwf_package_dict = {k: v for k, v in hdf["gwf_info"].attrs.items()}

        sol_keys = [k for k in keys if k.startswith("solution")]
        sol_keys.sort()
        if len(sol_keys) == 0:
            raise Exception("no 'solution' keys found")
        kperkstp = [(kper, kstp) for kper, kstp in zip(hdf["aux"]["kper"], hdf["aux"]["kstp"])]
        if len(kperkstp) != len(sol_keys):
            raise Exception(
                "number of solution datasets ({0}) != number of kper,kstp entries ({1})".format(len(sol_keys),
                                                                                                len(kperkstp)))
        kk_sol_map = {}
        for kk in kperkstp:
            sol = None
            for s in sol_keys:
                skper, skstp = hdf[s].attrs["kper"], hdf[s].attrs["kstp"]
                # print(skper, skstp)
                if skper == kk[0] and skstp == kk[1]:
                    sol = s
                    break
            if sol is None:
                raise Exception("no solution dataset found for kper,kstp:{0}".format(str(kk)))
            kk_sol_map[kk] = sol

        # nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
        nnodes = hdf["gwf_info"]["nnodes"]
        print(hdf["gwf_info"].keys())
        nodeuser = hdf["gwf_info"]["nodeuser"][:]
        lamb = np.zeros(nnodes)

        grid_shape = None
        if "nrow" in hdf["gwf_info"].keys():
            grid_shape = (hdf["gwf_info"]["nlay"][0],
                          hdf["gwf_info"]["nrow"][0],
                          hdf["gwf_info"]["ncol"][0])
            print("...structured grid found, shape:", grid_shape)

        ia = hdf["gwf_info"]["ia"][:]
        ja = hdf["gwf_info"]["ja"][:]

        comp_k33_sens = np.zeros(nnodes)
        comp_k_sens = np.zeros(nnodes)

        comp_ss_sens = None

        # has_sto = PerfMeas.has_sto_iconvert(gwf)
        has_sto = hdf[sol_keys[0]].attrs["has_sto"]
        if has_sto:
            comp_ss_sens = np.zeros(nnodes)

        comp_welq_sens = None
        comp_ghb_head_sens = None
        comp_ghb_cond_sens = None
        comp_rch_sens = None

        if "wel6" in gwf_package_dict:
            comp_welq_sens = np.zeros(nnodes)
        if "ghb6" in gwf_package_dict:
            comp_ghb_head_sens = np.zeros(nnodes)
            comp_ghb_cond_sens = np.zeros(nnodes)
        if "rch6" in gwf_package_dict or "rcha6" in gwf_package_dict:
            comp_rch_sens = np.zeros((nnodes))

        for itime, kk in enumerate(kperkstp[::-1]):
            data = {}

            print('solving adjoint solution for PerfMeas:', self._name, " (kper,kstp)", kk)
            sol_key = kk_sol_map[kk]
            # print(hdf[sol_key].keys())
            if sol_key in adf:
                raise Exception("solution key '{0}' already in adjoint hdf5 file".format(sol_key))

            dfdh = self._dfdh(kk, hdf[sol_key]["head"])
            data["dfdh"] = dfdh
            iss = hdf[sol_key]["iss"][0]

            if iss == 0:  # transient
                # if False:
                # get the derv of RHS WRT head
                drhsdh = hdf[sol_key]["drhsdh"][:]
                data["drhsdh"] = drhsdh
                rhs = (drhsdh * lamb) - dfdh
            else:
                rhs = - dfdh
            # if np.all(rhs == 0.0):
            #    print(
            #        "WARNING: adjoint solve rhs is all zeros, adjoint states cannot be calculated for {0} at kperkstp {1}".format(
            #            self._name, kk))
            # continue

            amat = hdf[sol_key]["amat"][:]
            amat_sp = sparse.csr_matrix((amat.copy(), ja.copy(), ia.copy()), shape=(len(ia) - 1, len(ia) - 1))
            amat_sp_t = amat_sp.transpose()
            lamb = spsolve(amat_sp_t, rhs)
            if np.any(np.isnan(lamb)):
                print("WARNING: nans in adjoint states for pm {0} at kperkstp {1}".format(self._name, kk))

            is_newton = hdf[sol_key].attrs["is_newton"]
            k_sens, k33_sens = PerfMeas.lam_dresdk_h(is_newton, lamb, hdf[sol_key]["sat"][:],
                                                     hdf[sol_key]["head"][:],
                                                     hdf["gwf_info"]["ihc"][:],
                                                     ia,
                                                     ja,
                                                     hdf["gwf_info"]["jas"][:],
                                                     hdf["gwf_info"]["cl1"][:],
                                                     hdf["gwf_info"]["cl2"][:],
                                                     hdf["gwf_info"]["hwva"][:],
                                                     hdf["gwf_info"]["top"][:],
                                                     hdf["gwf_info"]["bot"][:],
                                                     hdf["gwf_info"]["icelltype"][:],
                                                     hdf[sol_key]["k11"][:],
                                                     hdf[sol_key]["k33"][:]
                                                     )

            data["k11"] = k_sens
            data["k33"] = k33_sens
            comp_k_sens += k_sens
            comp_k33_sens += k33_sens

            if has_sto:
                ss_sens = lamb * hdf[sol_key]["dresdss_h"][:]
                data["ss"] = ss_sens
                comp_ss_sens += ss_sens

            data["welq"] = lamb
            comp_welq_sens += lamb

            if "ghb6" in gwf_package_dict and kk in gwf_package_dict["ghb6"]:
                sens_ghb_head, sens_ghb_cond = self.lam_drhs_dghb(lamb, head_dict[kk], gwf_package_dict["ghb6"][kk])
                data["ghbhead_" + sol_key] = sens_ghb_head
                data["ghbcond_" + sol_key] = sens_ghb_cond
                comp_ghb_head_sens += sens_ghb_head
                comp_ghb_cond_sens += sens_ghb_cond

            sens_rch = lamb * hdf["gwf_info"]["area"][:]
            comp_rch_sens += sens_rch
            data["rech"] = sens_rch

            data["lambda"] = lamb
            data["head"] = hdf[sol_key]["head"][:]

            if self.verbose_level > 2:
                data["amat"] = amat
                data["rhs"] = rhs
            PerfMeas.write_group_to_hdf(adf, sol_key, data, nodeuser=nodeuser,grid_shape=grid_shape)

        data = {}
        data["k11"] = comp_k_sens
        data["k33"] = comp_k33_sens
        # self.save_array("comp_sens_k33", comp_k33_sens, gwf_name, gwf, mg_structured)
        # self.save_array("comp_sens_k11", comp_k_sens, gwf_name, gwf, mg_structured)
        # addr = ["NODEUSER", gwf_name, "DIS"]
        # wbaddr = gwf.get_var_address(*addr)
        # nuser = gwf.get_value(wbaddr) - 1
        # if len(nuser) == 1:
        #    nuser = np.arange(nnodes, dtype=int)
        # df = pd.DataFrame(data, index=nuser)

        if has_sto:
            data["ss"] = comp_ss_sens
        data["welq"] = comp_welq_sens
        if "ghb6" in gwf_package_dict:
            data["ghbhead"] = comp_ghb_head_sens
            data["ghbcond"] = comp_ghb_cond_sens
        data["rch"] = comp_rch_sens
        PerfMeas.write_group_to_hdf(adf, "composite", data,nodeuser=nodeuser,grid_shape=grid_shape)
        adf.close()
        hdf.close()

        df = pd.DataFrame({"k11":comp_k_sens,"k33":comp_k33_sens,"welq":comp_welq_sens,"rch":comp_rch_sens},index=nodeuser)
        df.to_csv("{0}_adj_summary.csv".format(self._name))
        return df

    @staticmethod
    def write_group_to_hdf(hdf, group_name, data_dict, attr_dict={},grid_shape=None,nodeuser=None):
        if group_name in hdf:
            raise Exception("group_name {0} already in hdf file".format(group_name))
        grp = hdf.create_group(group_name)
        for name, val in attr_dict.items():
            grp.attrs[name] = val
        kijs = None
        if grid_shape is not None and nodeuser is not None:
            kijs = PerfMeas.get_lrc(grid_shape, list(nodeuser))
        for tag, item in data_dict.items():
            if isinstance(item, list):
                item = np.array(item)
            if isinstance(item, np.ndarray):
                if grid_shape is not None and nodeuser is not None:
                    if len(item) == len(nodeuser): #3D
                        arr = np.zeros(grid_shape,dtype=item.dtype)
                        for kij, v in zip(kijs, item):
                            arr[kij] = v
                        dset = grp.create_dataset(tag, grid_shape, dtype=item.dtype, data=arr)
                    else:
                        raise Exception("doh!")
                else:
                    dset = grp.create_dataset(tag, item.shape, dtype=item.dtype, data=item)
            elif isinstance(item, dict):
                if "nodelist" in item:
                    iitem = item["nodelist"]
                    dset = grp.create_dataset(tag, iitem.shape, dtype=iitem.dtype, data=iitem)
                elif "bound" in item:
                    iitem = item["bound"]
                    dset = grp.create_dataset(tag, iitem.shape, dtype=iitem.dtype, data=iitem)
                else:
                    print("Mf6Adj._write_group_to_hdf(): unused data_dict item {0}".format(tag))
            else:
                raise Exception("unrecognized data_dict entry: {0},type:{1}".format(tag, type(item)))
        if nodeuser is not None:
            dset = grp.create_dataset("nodeuser", nodeuser.shape, dtype=nodeuser.dtype, data=nodeuser)
        if kijs is not None:
            for idx,name in enumerate(["k","i","j"]):
                arr = np.array([kij[idx] for kij in kijs],dtype=int)
                dset = grp.create_dataset(name, arr.shape, dtype=arr.dtype, data=arr)


    @staticmethod
    def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
        # todo: upstream weighting - could use height1 and height2 to check...
        # todo: vertically staggered
        d = (width * cl1 * height1 * (height2 ** 2) * (k2 ** 2)) / (
                ((cl2 * height1 * k1) + ((cl1 * height2 * k2))) ** 2)
        return d

    @staticmethod
    def smooth_sat(sat):
        # satomega = self._gwf.get_value(self._gwf.get_var_address("SATOMEGA", self._gwf_name, "NPF"))
        satomega = 1.0e-6
        A_omega = 1 / (1 - satomega)
        s_sat = 1.0
        if sat < 0:
            s_sat = 0
        elif sat >= 0 and sat < satomega:
            s_sat = (A_omega / (2 * satomega)) * sat ** 2
        elif sat >= satomega and sat < 1 - satomega:
            s_sat = A_omega * sat + 0.5 * (1 - A_omega)
        elif sat >= 1 - satomega and sat < 1:
            s_sat = 1 - (A_omega / (2 * satomega)) * ((1 - sat) ** 2)
        return s_sat

    @staticmethod
    def d_smooth_sat_dh(sat, top, bot):
        satomega = 1.0e-6
        A_omega = 1 / (1 - satomega)
        d_s_sat_dh = 0.0
        if sat >= 0 and sat < satomega:
            d_s_sat_dh = (A_omega / satomega) * sat / (top - bot)
        elif sat >= satomega and sat < 1 - satomega:
            d_s_sat_dh = A_omega / (top - bot)
        elif sat >= 1 - satomega and sat < 1:
            d_s_sat_dh = (A_omega / satomega) * (1 - sat) / (top - bot)
        return d_s_sat_dh

    @staticmethod
    def _smooth_sat(sat1, sat2, h1, h2):
        if h1 >= h2:
            value = PerfMeas.smooth_sat(sat1)
        else:
            value = PerfMeas.smooth_sat(sat2)
        return value

    @staticmethod
    def _d_smooth_sat_dh(sat, h1, h2, top, bot):
        value = 0.0
        if h1 >= h2:
            value = PerfMeas.d_smooth_sat_dh(sat, top, bot)
        return value

    @staticmethod
    def lam_dresdk_h(is_newton, lamb, sat, head, ihc, ia, ja, jas, cl1, cl2, hwva, top, bot,
                     icelltype, k11, k33):

        # ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
        # # IHC tells us whether connection is vertical (and if so, whether connection is above or below) or horizontal (and if so, whether it is a vertically staggered grid).
        # # It is of size NJA (or number of connections)
        # ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
        # # IA is the number of connections, plus 1 (for self), for each node in grid. it is of size NNODES + 1
        # ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
        # # JA is an array containing all cells for which there is a connection (including self) for each node. it is of size NJA
        # jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
        # cl1 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
        # # distance from node to cell m boundary (size NJA)
        # cl2 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
        # # distance from cell m to node boundary (size NJA)
        # hwva = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)
        # # Width perpendicular to flow for a horizontal connection, or the face area for a vertical connection. size NJA
        # top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        # # top elevation for all nodes
        # bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        # bottom elevation for all nodes
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        # array of number of connections per node (size ndoes)

        # icelltype = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "ICELLTYPE", gwf)

        sat_mod = sat.copy()
        sat_mod[icelltype == 0] = 1.0

        # height = sat_mod * (top - bot)
        height = top - bot

        result33 = np.zeros_like(head)
        result = np.zeros_like(head)

        # k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
        # k22 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K22", gwf)
        # assert np.all(k11 == k22)
        # k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)

        for node, (offset, ncon) in enumerate(zip(ia, iac)):
            sum1 = 0.
            sum2 = 0.
            height1 = height[node]
            pp = 1
            for ii in range(offset + 1, offset + ncon):
                mnode = ja[ii]
                height2 = height[mnode]

                jj = jas[ii]
                if jj < 0:
                    raise Exception()
                iihc = ihc[jj]

                if iihc == 0:  # vertical con
                    dconddk33 = PerfMeas._dconddhk(k33[node], k33[mnode], 0.5 * height1, 0.5 * height2, hwva[jj],
                                                   1.0, 1.0)
                    t2 = dconddk33 * (head[mnode] - head[node]) * (lamb[node] - lamb[mnode])
                    sum1 += t2

                else:
                    if is_newton:
                        dconddk = PerfMeas._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj], height1,
                                                     height2)
                        SF = PerfMeas._smooth_sat(sat_mod[node], sat_mod[mnode], head[node], head[mnode])

                    else:
                        dconddk = PerfMeas._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj],
                                                     height1 * sat_mod[node], height2 * sat_mod[mnode])
                        SF = 1.0

                    t1 = SF * dconddk * (head[mnode] - head[node]) * (lamb[node] - lamb[mnode])
                    sum2 += t1

            result33[node] = sum1
            result[node] = sum2
        return result, result33

    def rch_sens_old(self, gwf_name, gwf, lamb, sp_dict):
        result = np.zeros_like(lamb)
        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)

        for id in sp_dict:
            n = id["node"] - 1
            result[n] = lamb[n] * area[n]
        return result

    def lam_drhs_dghb(self, lamb, head, sp_dict):
        result_head = np.zeros_like(lamb)
        result_cond = np.zeros_like(lamb)

        for id in sp_dict:
            n = id["node"] - 1
            # the second item in bound should be cond
            result_head[n] = lamb[n] * id["bound"][1]
            # the first item in bound should be head
            lam_drhs_dcond = lamb[n] * id["bound"][0]
            lam_dadcond_h = -1.0 * lamb[n] * head[n]
            result_cond[n] = lam_drhs_dcond + lam_dadcond_h

        return result_head, result_cond

    def lam_drhs_dqwel(self, lamb, sp_dict):
        result = np.zeros_like(lamb)
        for id in sp_dict:
            n = id["node"] - 1
            result[n] = lamb[n]
        return result

    def save_array_old(self, filetag, avec, gwf_name, gwf, structured_mg):
        nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "NODEUSER", gwf) - 1
        nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
        jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf)
        filetag = "pm-" + self._name + "_" + filetag
        # if not a reduced node scheme
        if len(nodeuser) == 1:
            nodeuser = np.arange(nnodes)
        if structured_mg is not None:
            kijs = structured_mg.get_lrc(list(nodeuser))
            arr = np.zeros((structured_mg.nlay, structured_mg.nrow, structured_mg.ncol))
            for kij, v in zip(kijs, avec):
                arr[kij] = v
            for k, karr in enumerate(arr):
                filename = filetag + "_k{0:03d}.dat".format(k)
                np.savetxt(filename, karr, fmt="%15.6E")
        else:
            filename = filetag + ".dat"
            if avec.shape[0] == nodeuser.shape[0]:
                rarr = np.array((nodeuser, avec)).transpose()
            elif avec.shape[0] == jas.shape[0]:
                rarr = np.array((jas, avec)).transpose()
            else:
                raise Exception(
                    "unrecognized unstructed vector length: {0} for filename {1}".format(avec.shape[0], filename))
            np.savetxt(filename, rarr, fmt=["%10d", "%15.6E"])

    def save_array_old(self, filetag, avec, gwf_name, gwf, structured_mg):
        nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "NODEUSER", gwf) - 1
        nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
        jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf)
        filetag = "pm-" + self._name + "_" + filetag
        # if not a reduced node scheme
        if len(nodeuser) == 1:
            nodeuser = np.arange(nnodes)
        if structured_mg is not None:
            kijs = structured_mg.get_lrc(list(nodeuser))
            arr = np.zeros((structured_mg.nlay, structured_mg.nrow, structured_mg.ncol))
            for kij, v in zip(kijs, avec):
                arr[kij] = v
            for k, karr in enumerate(arr):
                filename = filetag + "_k{0:03d}.dat".format(k)
                np.savetxt(filename, karr, fmt="%15.6E")
        else:
            filename = filetag + ".dat"
            if avec.shape[0] == nodeuser.shape[0]:
                rarr = np.array((nodeuser, avec)).transpose()
            elif avec.shape[0] == jas.shape[0]:
                rarr = np.array((jas, avec)).transpose()
            else:
                raise Exception(
                    "unrecognized unstructed vector length: {0} for filename {1}".format(avec.shape[0], filename))
            np.savetxt(filename, rarr, fmt=["%10d", "%15.6E"])

    def _dadk(self, gwf_name, gwf, sat, amat):
        """partial of A matrix WRT K
		"""
        is_chd = False
        chd_list = []
        # names = list(gwf.get_input_var_names())
        # chds = [name for name in names if 'CHD' in name and 'NODELIST' in name]
        # for name in chds:
        # 	chd = np.array(PerfMeas.get_ptr_from_gwf(gwf_name,name.split('/')[1],"NODELIST",gwf)-1)
        # 	chd_list.extend(list(chd))
        # 	is_chd = True
        # chd_list = set(chd_list)

        # nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
        # ib = np.array(PerfMeas.get_value_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf)).reshape(-1)
        # nlay = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NLAY", gwf)[0]
        ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
        # IHC tells us whether connection is vertical (and if so, whether connection is above or below) or horizontal (and if so, whether it is a vertically staggered grid).
        # It is of size NJA (or number of connections)
        ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
        # IA is the number of connections, plus 1 (for self), for each node in grid. it is of size NNODES + 1
        ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
        # JA is an array containing all cells for which there is a connection (including self) for each node. it is of size NJA
        jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
        cl1 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
        # distance from node to cell m boundary (size NJA)
        cl2 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
        # distance from cell m to node boundary (size NJA)
        hwva = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)
        # Width perpendicular to flow for a horizontal connection, or the face area for a vertical connection. size NJA
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        # top elevation for all nodes
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        # bottom elevation for all nodes
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        # array of number of connections per node (size ndoes)

        icelltype = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "ICELLTYPE", gwf)

        sat_mod = sat.copy()
        sat_mod[icelltype == 0] = 1.0

        # height = sat_mod * (top - bot)
        height = top - bot

        # TODO: check here for converible cells

        d_mat_k11 = np.zeros(ja.shape[0])
        d_mat_k33 = np.zeros(ja.shape[0])

        k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
        k22 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K22", gwf)
        assert np.all(k11 == k22)
        k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)

        for node, (offset, ncon) in enumerate(zip(ia, iac)):

            # ncon -= 1 # for self

            # if ib[node]==0:
            #	pass
            # if is_chd and node in chd_list:
            #		continue
            if False:
                pass
            else:
                sum1 = 0.
                sum2 = 0.
                height1 = height[node]
                pp = 1
                for ii in range(offset + 1, offset + ncon):
                    mnode = ja[ii]
                    # if is_chd and mnode in chd_list:
                    #		continue
                    height2 = height[mnode]
                    # if icelltype[mnode] != 0:
                    # height2 *= sat[mnode]

                    jj = jas[ii]
                    if jj < 0:
                        raise Exception()
                    iihc = ihc[jj]

                    if iihc == 0:  # vertical con
                        # v1 = PerfMeas._dconddvk(k33[node],height1,sat[node],k33[mnode],
                        #							height2,sat[mnode],hwva[jj],amat[jj])
                        # def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
                        #	d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
                        #	return d
                        v2 = PerfMeas.derivative_conductance_k1(k33[node], k33[mnode], height1, height2,
                                                                cl1[jj] + cl2[jj], hwva[jj])
                        # derivative_conductance_k33(k1, k2, w1, w2, area)
                        # v3 = PerfMeas.derivative_conductance_k33(k33[node],k33[mnode],height1, height2, hwva[jj])

                        d_mat_k33[ia[node] + pp] += v2
                        # d_mat_k123[ia[node]+pp] += v3
                        sum1 += v2
                        pp += 1

                    else:
                        v1 = PerfMeas._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj],
                                                height1 * sat_mod[node], height2 * sat_mod[mnode])
                        # v1 = PerfMeas._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj],
                        #							height1, height2)
                        # v2 = -PerfMeas.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj]+cl2[jj], cl1[jj]+cl2[jj], hwva[jj],height2*sat_mod[mnode])
                        # v2 = PerfMeas.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj],cl2[jj], hwva[jj],height1)

                        d_mat_k11[ia[node] + pp] += v1
                        # d_mat_k123[ia[node]+pp] += v1
                        sum2 += v1
                        pp += 1

                d_mat_k11[ia[node]] = -sum2
                d_mat_k33[ia[node]] = -sum1
            # d_mat_k22[ia[node]] = -sum3
            # d_mat_k123[ia[node]] = d_mat_k11[ia[node]] + d_mat_k22[ia[node]] + d_mat_k33[ia[node]]

        return d_mat_k11, -d_mat_k33

    # @staticmethod
    # def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
    # 	# todo: upstream weighting - could use height1 and height2 to check...
    # 	# todo: vertically staggered
    # 	d = (width * cl1 * height1 * (height2 ** 2) * (k2**2)) / (((cl2 * height1 * k1) + ((cl1 * height2 * k2))) ** 2)
    # 	return d
    #
    # @staticmethod
    # def _dconddvk(k1,height1,sat1,k2,height2,sat2,area,vcond_12):
    # 	"""need to think about how k1 and k2 are combined to
    # 	form the average k between the two cells
    # 	from MH:
    # 	dcond_n,m / dk_n,m = cond_n,m**2 / ((area * k_n,m**2)/(0.5*(top_n - bot_n)))
    #
    # 	todo: VARIABLE CV and DEWATER options
    #
    # 	"""
    #
    # 	#condsq = (1./((1./((area*k1)/(0.5*(top1-bot1)))) + (1./((area*k2)/(0.5*(top2-bot2))))))**2
    # 	#return condsq / ((area * k1**2)/(0.5*(top1-bot1)))
    # 	d = (sat1*vcond_12**2) / (((area * k1)**2)/(0.5*(height1)))
    # 	return d

    @staticmethod
    def derivative_conductance_k33(k1, k2, w1, w2, area):
        d = - 2.0 * w1 * area / ((w1 + w2 * k1 / k2) ** 2)
        return d

    @staticmethod
    def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
        d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
        return d

    def lam_dAdss_h(self, gwf_name, gwf, lamb, head, dt, sat):
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
        iconvert = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)

        # handle iconvert
        sat_mod = sat.copy()
        sat_mod[iconvert == 0] = 1.0

        # correction for solved head below cell bot
        # not sure about this one...
        # head[head<bot] = bot[head<bot]
        # head[head < bot] = bot[head < bot]

        # result = -1. * lamb * head * area * ((top - bot) * sat_mod) / dt
        result = -1. * lamb * head * area * (top - bot) / dt
        return result

    def sens_ss_indirect(self, gwf_name, gwf, lamb, head, head_old, dt, sat, sat_old):
        # todo: check that sy is equivalent to ss - I think it might be...but maybe not...
        return self.lam_dAdss_h(gwf_name, gwf, lamb, head, dt, sat) - self.lam_dAdss_h(gwf_name, gwf, lamb, head_old,
                                                                                       dt, sat_old)

    def lam_dAdk_h(self, gwf_name, gwf, lamb, dAdk, head):
        ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
        # IA is the number of connections, plus 1 (for self), for each node in grid. it is of size NNODES + 1
        ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
        # JA is an array containing all cells for which there is a connection (including self) for each node. it is of size NJA
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        # ib = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf).reshape(-1)

        is_chd = False
        chd_list = []
        # names = list(gwf.get_input_var_names())
        # chds = [name for name in names if 'CHD' in name and 'NODELIST' in name]
        # for name in chds:
        # 	chd = np.array(PerfMeas.get_ptr_from_gwf(gwf_name,name.split('/')[1],"NODELIST",gwf)-1)
        # 	chd_list.extend(list(chd))
        # 	is_chd = True
        # chd_list = set(chd_list)

        result = np.zeros_like(lamb)
        for i in range(len(lamb)):
            sum1 = 0.0
            sum2 = 0.0
            for ii in range(iac[i]):
                sum1 += dAdk[ia[i] + ii] * head[ja[ia[i] + ii]]
            sum1 *= lamb[i]
            for ii in list(range(iac[i]))[1:]:
                # if is_chd and ja[ia[i]+ii] in chd_list:
                #	continue
                # elif ib[ja[ia[i]+ii]] == 0:
                # if ib[ja[ia[i] + ii]] == 0:
                #		pass
                # else:
                sum2 += lamb[ja[ia[i] + ii]] * dAdk[ia[i] + ii] * (head[i] - head[ja[ia[i] + ii]])
            sums = sum1 + sum2
            # print(list(range(iac[i]))[1:])
            # print(sum1,sum2,sums)
            result[i] = sums

        return result

    def _drhsdh(self, gwf_name, gwf, dt, sat):
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
        storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
        sat_mod = sat.copy()
        if PerfMeas.has_sto_iconvert(gwf):
            iconvert = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)
            sat_mod[iconvert == 0] = 1.0
        else:
            sat_mod = np.ones_like(top)
        # drhsdh = -1. * storage * area * ((top - bot)*sat_mod) / dt
        drhsdh = -1. * storage * area * (top - bot) / dt
        return drhsdh

    def _dfdh_old(self, kk, gwf_name, gwf, head_dict):
        nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
        dfdh = np.zeros(nnodes)
        for pfr in self._entries:
            if pfr.kperkstp == kk:
                if self._type == "direct":
                    # dfdh[pfr.nnode] = -1. * pfr.weight / deltat_dict[kk]
                    dfdh[pfr.nnode] = pfr.weight
                elif self._type == "residual":
                    dfdh[pfr.nnode] = 2.0 * pfr.weight * (head_dict[kk][pfr.nnode] - pfr.obsval)
        return dfdh

    def _dfdh(self, kk, head):
        dfdh = np.zeros_like(head)
        for pfr in self._entries:
            if pfr.kperkstp == kk:
                if self._type == "direct":
                    # dfdh[pfr.nnode] = -1. * pfr.weight / deltat_dict[kk]
                    dfdh[pfr.nnode] = pfr.weight
                elif self._type == "residual":
                    dfdh[pfr.nnode] = 2.0 * pfr.weight * (head[pfr.nnode] - pfr.obsval)
        return dfdh

    @staticmethod
    def get_value_from_gwf(gwf_name, pak_name, prop_name, gwf):
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value(addr)

    @staticmethod
    def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf):
        addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
        return gwf.get_value_ptr(addr)

    @staticmethod
    def get_node(shape, lrc_list):

        if not isinstance(lrc_list, list):
            lrc_list = [lrc_list]
        multi_index = tuple(np.array(lrc_list).T)
        return np.ravel_multi_index(multi_index, shape).tolist()

    @staticmethod
    def get_lrc(shape, nodes):
        if isinstance(nodes, int):
            nodes = [nodes]
        return list(zip(*np.unravel_index(nodes, shape)))
# @staticmethod
# def get_input_var_names():
# 	return gwf.get_input_var_names()
