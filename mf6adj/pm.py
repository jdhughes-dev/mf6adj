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
    def __init__(self, kper, kstp, inode, pm_type, pm_form, weight, obsval, k=None, i=None, j=None, ):
        self._kper = int(kper)
        self._kstp = int(kstp)
        self.kperkstp = (self._kper, self._kstp)
        self.inode = int(inode)
        self._k = None
        if k is not None:
            self._k = int(k)
        self._i = None
        if i is not None:
            self._i = int(i)
        self._j = None
        if j is not None:
            self._j = int(j)
        self.weight = float(weight)
        self.obsval = float(obsval)
        self.pm_type = pm_type.lower().strip()
        self.pm_form = pm_form.lower().strip()
        if self.pm_form not in ["direct","residual"]:
            raise Exception("PerfMeasRecord.pm_form must be 'direct' or 'residual', not '{0}'".format(self.pm_form))
    def __repr__(self):
        s = "kperkstp:{0}, inode:{1}, k:{2}, type:{3}, form:{4}".format(self.kperkstp,self.inode, self._k,self.pm_tpye,self.pm_form)
        if self._i is not None:
            s += ", i:{0}".format(self._i)
        if self._j is not None:
            s += ", j:{0}".format(self._j)
        return s


class PerfMeas(object):
    """todo: preprocess all the connectivity in to faster look dict containers,
	including nnode to kij info for structured grids

	todo: convert several class methods to static methods - this might make testing easier

	todo: add a no-data value var to fill empty spots in output arrays.  currently using zero :(

	todo: check that each entry's kperkstp is in the dicts being passed to solve_adjoint()
	
	"""

    def __init__(self, pm_name, pm_entries, is_structured, verbose_level=1):
        self._name = pm_name.lower().strip()
        self._entries = pm_entries
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

    def solve_forward(self, head_dict,sp_package_dict):
        """for testing only"""
        result = 0.0
        for pfr in self._entries:
            if pfr.pm_type == "head":
                if pfr.pm_form == "direct":
                    result += pfr.weight * head_dict[pfr.kperkstp][pfr.inode]
                elif pfr.pm_form == "residual":
                    result += (pfr.weight * (head_dict[pfr.kperkstp][pfr.inode] - pfr.obsval)) ** 2
                else:
                    raise Exception("something is wrong")
            else:
                for gwf_ptype, bnd_d in sp_package_dict.items():
                    for kk,kk_list in bnd_d.items():
                        for kk_d in kk_list:
                            if kk == pfr.kperkstp:
                                if kk_d["packagename"] == pfr.pm_type:
                                    if pfr.pm_form == "direct":
                                        result += pfr.weight * kk_d["simval"]
                                    elif pfr.pm_form == "residual":
                                        result += (pfr.weight * (kk_d["simval"] - pfr.obsval)) ** 2

        return result


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
            hdf5_adjoint_solution_fname = os.path.join(pth, "adjoint_solution_{0}_".format(self._name) + hdf5_forward_solution_fname)

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
        nnodes = hdf["gwf_info"]["nnodes"][:]
        #print(hdf["gwf_info"].keys())
        nodeuser = hdf["gwf_info"]["nodeuser"][:]
        nodereduced = hdf["gwf_info"]["nodereduced"][:]
        if len(nodeuser) == 1:
            nodeuser = np.arange(nnodes,dtype=int)
        lamb = np.zeros(nnodes)

        grid_shape = None
        if "nrow" in hdf["gwf_info"].keys():
            grid_shape = (hdf["gwf_info"]["nlay"][0],
                          hdf["gwf_info"]["nrow"][0],
                          hdf["gwf_info"]["ncol"][0])
            print("...structured grid found, shape:", grid_shape)

        ia = hdf["gwf_info"]["ia"][:]
        ja = hdf["gwf_info"]["ja"][:]
        ihc = hdf["gwf_info"]["ihc"][:]
        jas = hdf["gwf_info"]["jas"][:]
        cl1 = hdf["gwf_info"]["cl1"][:]
        cl2 = hdf["gwf_info"]["cl2"][:]
        hwva = hdf["gwf_info"]["hwva"][:]
        top = hdf["gwf_info"]["top"][:]
        bot = hdf["gwf_info"]["bot"][:]
        icelltype = hdf["gwf_info"]["icelltype"][:]
        area = hdf["gwf_info"]["area"][:]
        idomain = hdf["gwf_info"]["idomain"][:]

        comp_k33_sens = np.zeros(nnodes)
        comp_k_sens = np.zeros(nnodes)

        comp_ss_sens = None

        # has_sto = PerfMeas.has_sto_iconvert(gwf)
        has_sto = hdf[sol_keys[0]].attrs["has_sto"]
        if has_sto:
            comp_ss_sens = np.zeros(nnodes)

        comp_welq_sens = np.zeros(nnodes)
        comp_ghb_head_sens = None
        comp_ghb_cond_sens = None
        comp_rch_sens = np.zeros((nnodes))

        if "ghb6" in gwf_package_dict:
            comp_ghb_head_sens = np.zeros(nnodes)
            comp_ghb_cond_sens = np.zeros(nnodes)

        for itime, kk in enumerate(kperkstp[::-1]):
            data = {}

            print('solving adjoint solution for PerfMeas:', self._name, " (kper,kstp)", kk)
            sol_key = kk_sol_map[kk]
            #print(hdf[sol_key].keys())
            if sol_key in adf:
                raise Exception("solution key '{0}' already in adjoint hdf5 file".format(sol_key))

            dfdh = self._dfdh(kk, hdf[sol_key])
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
            head = hdf[sol_key]["head"][:]
            amat_sp = sparse.csr_matrix((amat.copy(), ja.copy(), ia.copy()), shape=(len(ia) - 1, len(ia) - 1))
            amat_sp_t = amat_sp.transpose()
            lamb = spsolve(amat_sp_t, rhs)
            if np.any(np.isnan(lamb)):
                print("WARNING: nans in adjoint states for pm {0} at kperkstp {1}".format(self._name, kk))

            is_newton = hdf[sol_key].attrs["is_newton"]
            k_sens, k33_sens = PerfMeas.lam_dresdk_h(is_newton, lamb, hdf[sol_key]["sat"][:],
                                                     head, ihc, ia, ja, jas, cl1, cl2,
                                                     hwva, top, bot, icelltype,
                                                     hdf[sol_key]["k11"][:],
                                                     hdf[sol_key]["k33"][:]
                                                     )

            data["k11"] = k_sens
            data["k33"] = k33_sens
            comp_k_sens += k_sens
            comp_k33_sens += k33_sens

            if has_sto:
                if iss == 0:
                    ss_sens = lamb * hdf[sol_key]["dresdss_h"][:]
                else:
                    ss_sens = np.zeros_like(lamb)
                data["ss"] = ss_sens
                comp_ss_sens += ss_sens

            data["wel"] = lamb
            comp_welq_sens += lamb

            #if "ghb6" in gwf_package_dict and kk in gwf_package_dict["ghb6"]:
            # look for ghb packages
            ghb_keys = []
            for key in hdf[sol_key]:
                dset = hdf[sol_key][key]
                ptype = dset.attrs.get("ptype",None)
                if ptype is not None and ptype.lower() == "ghb6":
                    ghb_keys.append(key)

            #if "ghb" in hdf[sol_key]:
            for ghb_key in ghb_keys:

                #print(hdf[sol_key]["ghb"].keys())
                sp_ghb_dict = {"bound":hdf[sol_key][ghb_key]["bound"][:],
                               "node":hdf[sol_key][ghb_key]["nodelist"][:]}
                sens_ghb_head, sens_ghb_cond = self.lam_drhs_dghb(lamb, head, sp_ghb_dict)
                data[ghb_key+ "_" + sol_key] = sens_ghb_head
                data[ghb_key + "_" + sol_key] = sens_ghb_cond
                comp_ghb_head_sens += sens_ghb_head
                comp_ghb_cond_sens += sens_ghb_cond

            sens_rch = lamb
            comp_rch_sens += sens_rch
            data["rech"] = sens_rch

            data["lambda"] = lamb
            data["head"] = hdf[sol_key]["head"][:]

            if self.verbose_level > 2:
                data["amat"] = amat
                data["rhs"] = rhs

            PerfMeas.write_group_to_hdf(adf, sol_key, data, nodeuser=nodeuser, grid_shape=grid_shape,nodereduced=nodereduced)

        data = {}
        data["k11"] = comp_k_sens
        data["k33"] = comp_k33_sens

        if has_sto:
            data["ss"] = comp_ss_sens
        data["wel"] = comp_welq_sens
        if comp_ghb_head_sens is not None:
            data["ghbhead"] = comp_ghb_head_sens
            data["ghbcond"] = comp_ghb_cond_sens
        data["rch"] = comp_rch_sens
        PerfMeas.write_group_to_hdf(adf, "composite", data, nodeuser=nodeuser, grid_shape=grid_shape,nodereduced=nodereduced)
        adf.close()
        hdf.close()

        df = pd.DataFrame({"k11": comp_k_sens, "k33": comp_k33_sens, "wel": comp_welq_sens, "rch": comp_rch_sens},
                          index=nodeuser+1)
        if comp_ghb_head_sens is not None:
            df.loc[:,"ghb_head"] = comp_ghb_head_sens
            df.loc[:,"ghb_cond"] = comp_ghb_cond_sens
        if has_sto:
            df["ss"] = comp_ss_sens


        df.index.name = "node"
        df.to_csv("adjoint_summary_{0}.csv".format(self._name))
        return df

    @staticmethod
    def write_group_to_hdf(hdf, group_name, data_dict, attr_dict={}, grid_shape=None, nodeuser=None,nodereduced=None):
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
                    if len(item) == len(nodeuser):  # 3D
                        arr = np.zeros(grid_shape, dtype=item.dtype)
                        for kij, v in zip(kijs, item):
                            arr[kij] = v

                        dset = grp.create_dataset(tag, grid_shape, dtype=item.dtype, data=arr)
                    else:
                        raise Exception("doh! "+str(tag))
                elif nodeuser is not None and nodereduced is not None:
                    arr = np.zeros_like(nodereduced, dtype=item.dtype)
                    for inode,v in zip(nodeuser,item):
                        arr[inode] = v
                    dset = grp.create_dataset(tag, arr.shape, dtype=item.dtype, data=arr)
                else:
                    dset = grp.create_dataset(tag, item.shape, dtype=item.dtype, data=item)
            elif isinstance(item, dict):
                subgrp = grp.create_group(tag)
                for k,v in item.items():
                    if isinstance(v,np.ndarray):
                        dset = subgrp.create_dataset(k, v.shape, dtype=v.dtype, data=v)
                    else:
                        #print("WARNING: unknown dtype, setting as attribute for group {2}: {0} {1}".format(k,type(v), tag))
                        subgrp.attrs[k] = v
                        #print()
                # if "nodelist" in item:
                #     iitem = item["nodelist"]
                #     dset = grp.create_dataset(tag, iitem.shape, dtype=iitem.dtype, data=iitem)
                # elif "bound" in item:
                #     iitem = item["bound"]
                #     dset = grp.create_dataset(tag, iitem.shape, dtype=iitem.dtype, data=iitem)
                # else:
                #     print("Mf6Adj._write_group_to_hdf(): unused data_dict item {0}".format(tag))
            else:
                raise Exception("unrecognized data_dict entry: {0},type:{1}".format(tag, type(item)))
        if nodeuser is not None:
            dset = grp.create_dataset("nodeuser", nodeuser.shape, dtype=nodeuser.dtype, data=nodeuser)
        if kijs is not None:
            for idx, name in enumerate(["k", "i", "j"]):
                arr = np.array([kij[idx] for kij in kijs], dtype=int)
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

        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        # array of number of connections per node (size ndoes)

        sat_mod = sat.copy()
        sat_mod[icelltype == 0] = 1.0

        # height = sat_mod * (top - bot)
        height = top - bot

        result33 = np.zeros_like(head)
        result = np.zeros_like(head)

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

    def lam_drhs_dghb(self, lamb, head, sp_dict):
        result_head = np.zeros_like(lamb)
        result_cond = np.zeros_like(lamb)

        #for id in sp_dict:
        for node,bound in zip(sp_dict["node"],sp_dict["bound"]):
            n = node - 1
            # the second item in bound should be cond
            result_head[n] = lamb[n] * bound[1]
            # the first item in bound should be head
            lam_drhs_dcond = lamb[n] * bound[0]
            lam_dadcond_h = -1.0 * lamb[n] * head[n]
            result_cond[n] = lam_drhs_dcond + lam_dadcond_h

        return result_head, result_cond

    @staticmethod
    def derivative_conductance_k33(k1, k2, w1, w2, area):
        d = - 2.0 * w1 * area / ((w1 + w2 * k1 / k2) ** 2)
        return d

    @staticmethod
    def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
        d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
        return d


    def _dfdh(self, kk, sol_dataset):
        head = sol_dataset["head"][:]
        dfdh = np.zeros_like(head)
        for pfr in self._entries:
            if pfr.kperkstp == kk:
                if pfr.pm_type == "head":
                    if pfr.pm_form == "direct":
                    # dfdh[pfr.nnode] = -1. * pfr.weight / deltat_dict[kk]
                        dfdh[pfr.inode] = pfr.weight
                    elif pfr.pm_form == "residual":
                        dfdh[pfr.inode] = 2.0 * pfr.weight * (head[pfr.inode] - pfr.obsval)
                else:
                    hcof = sol_dataset[pfr.pm_type]["hcof"][:]
                    inodelist = sol_dataset[pfr.pm_type]["nodelist"][:] - 1
                    idx = np.where(inodelist == pfr.inode)[0][0]
                    dfdh[pfr.inode] = hcof[idx]
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

