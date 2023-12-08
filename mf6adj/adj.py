import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import h5py
import modflowapi
import flopy

from .pm import PerfMeasRecord, PerfMeas

DT_FMT = "%Y-%m-%d %H:%M:%S"


class Mf6Adj(object):
    def __init__(self, adj_filename, lib_name, is_structured, verbose_level=1):

        """todo:

        check for unsupported horizontal conductance formulation
        check for unsupported vertical conductance formulation
        check for unsupported aniso options
        check for unsupported xt3d option/horizontal anisotropy!

        add chd,drn,and riv to bc types

        """
        self.verbose_level = int(verbose_level)
        if not os.path.exists(adj_filename):
            raise Exception("adj_filename '{0}' not found".format(adj_filename))
        self.adj_filename = adj_filename

        # process the flow model
        # make sure the lib exists
        if not os.path.exists(lib_name):
            raise Exception("MODFLOW-6 shared library  '{0}' not found".format(lib_name))
        # find the model name
        self._gwf_model_dict, namfile_dict = Mf6Adj.get_model_names_from_mfsim(".")
        if len(self._gwf_model_dict) != 1:
            raise Exception("only one model is current supported")
        self._gwf_name = list(self._gwf_model_dict.keys())[0]
        self._gwf_namfile = namfile_dict[self._gwf_name]
        self._gwf_package_dict = Mf6Adj.get_package_names_from_gwfname(self._gwf_namfile)
        if self._gwf_model_dict[self._gwf_name] != "gwf6":
            raise Exception("model is not a gwf6 type: {0}". \
                            format(self._gwf_model_dict[self._gwf_name]))
        self._gwf = None
        self._lib_name = lib_name
        self._flow_dir = "."
        self._gwf = self._initialize_gwf(lib_name, self._flow_dir)

        self._structured_mg = None
        self.is_structured = is_structured  # hard coded for now...
        if self.is_structured:
            nlay = self._gwf.get_value(self._gwf.get_var_address("NLAY", self._gwf_name.upper(), "DIS"))[0]
            nrow = self._gwf.get_value(self._gwf.get_var_address("NROW", self._gwf_name.upper(), "DIS"))[0]
            ncol = self._gwf.get_value(self._gwf.get_var_address("NCOL", self._gwf_name.upper(), "DIS"))[0]
            self._structured_mg = flopy.discretization.StructuredGrid(nrow=nrow,
                                                                      ncol=ncol,
                                                                      nlay=nlay)

        self._performance_measures = []

        self._read_adj_file()

        self._amat = {}
        self._head = {}
        self._head_old = {}

        self._kperkstp = []
        self._deltat = {}
        self._iss = {}
        self._sat = {}
        self._sat_old = {}
        self._gwf_package_types = ["wel6", "ghb6", "rch6", "rcha6"]

    def _read_adj_file(self):

        # clear any existing PMs
        self._performance_measures = []

        """read the adj input file

                """
        # used to detect location-pak duplicates
        current_period = -1
        current_entries = []

        addr = ["NODEUSER", self._gwf_name.upper(), "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nuser = self._gwf.get_value(wbaddr) - 1

        addr = ["NODEREDUCED", self._gwf_name.upper(), "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nred = self._gwf.get_value(wbaddr)

        with open(self.adj_filename, 'r') as f:
            count = 0
            while True:
                line = f.readline()
                count += 1
                # eof
                if line == "":
                    break

                # skip empty lines or comment lines
                if len(line.strip()) == 0 or line.strip()[0] == "#":
                    continue

                # read the options block
                if line.lower().strip().startswith("begin options"):
                    while True:
                        line2 = f.readline()
                        count += 1

                        if line2 == "":
                            raise EOFError("EOF while reading options")
                        elif len(line.strip()) == 0 or line.strip()[0] == "#":
                            continue
                        elif line2.lower().strip().startswith("begin"):
                            raise Exception("a new begin block found while parsing options")
                        elif line2.lower().strip().startswith("end options"):
                            break

                # parse a new performance measure block
                elif line.lower().strip().startswith("begin performance_measure"):
                    raw = line.lower().strip().split()
                    if len(raw) != 5:
                        raise Exception(
                            "wrong number of entries on 'begin performance_measure' line number {0}: '{1}'".format(
                                count, line))
                    pm_name = raw[2].strip().lower()
                    if raw[3].strip().lower() != "type":
                        raise Exception("4th entry on line {0} should be 'type', not '{1}'".format(count, raw[3]))
                    pm_type = raw[4].strip().lower()
                    if pm_type not in ["direct", "residual"]:
                        raise Exception("unrecognized PM type:'{0}', should be 'direct' or 'residual'")
                    pm_entries = []
                    while True:
                        line2 = f.readline()
                        count += 1
                        if line2 == "":
                            raise EOFError("EOF while reading performance_measure block '{0}'".format(line))
                        elif len(line.strip()) == 0 or line.strip()[0] == "#":
                            continue
                        elif line2.lower().strip().startswith("begin"):
                            raise Exception(
                                "a new begin block found while parsing performance_measure block '{0}'".format(line))
                        elif line2.lower().strip().startswith("end performance_measure"):
                            break

                        raw = line2.lower().strip().split()
                        kper = int(raw[0]) - 1
                        kstp = int(raw[1]) - 1
                        # todo: check limits of kper, kstp

                        if self.is_structured:
                            if len(raw) < 5:
                                raise Exception(
                                    "performance measure {0} line {1} has too few entries, need at least 5".format(
                                        pm_name, line2))
                            weight = None
                            if pm_type == "direct":
                                if len(raw) < 6:
                                    raise Exception(
                                        "direct performance measure {0} line {1} has wrong number of entries, should be 6".format(
                                            pm_name, line2))
                                weight = float(raw[5])
                            obsval = None
                            if pm_type == "residual":
                                if len(raw) < 7:
                                    raise Exception(
                                        "residual performance measure {0} line {1} has wrong number of entries, should be 7".format(
                                            pm_name, line2))
                                weight = float(raw[5])
                                obsval = float(raw[6])

                            kij = []
                            for i in range(3):
                                try:
                                    kij.append(int(raw[i + 2]) - 1)
                                except:
                                    raise Exception("error casting k-i-j info on line {0}: '{1}'".format(count, line2))

                            # convert to node number
                            n = self._structured_mg.get_node([kij])[0]
                            # if there is a reduced node scheme
                            if len(nuser) > 1:
                                nn = np.where(nuser == n)[0]
                                if nn.shape[0] != 1:
                                    print(n, nn)
                                    if self.is_structured:
                                        print(kij)
                                    raise Exception("node num {0} not in reduced node num".format(n))

                                pm_entries.append(
                                    PerfMeasRecord(kper, kstp, nn[0], k=kij[0], i=kij[1], j=kij[2], weight=weight,
                                                   obsval=obsval))
                            else:
                                pm_entries.append(
                                    PerfMeasRecord(kper, kstp, n, k=kij[0], i=kij[1], j=kij[2], weight=weight,
                                                   obsval=obsval))
                        else:
                            # raise NotImplementedError("only structured grids currently supported")
                            if len(raw) < 3:
                                raise Exception(
                                    "performance measure {0} line {1} has too few entries, need at least 3".format(
                                        pm_name, line2))
                            weight = None
                            if pm_type == "direct":
                                if len(raw) < 4:
                                    raise Exception(
                                        "direct performance measure {0} line {1} has wrong number of entries, should be 4".format(
                                            pm_name, line2))
                                weight = float(raw[3])
                            obsval = None
                            if pm_type == "residual":
                                if len(raw) < 5:
                                    raise Exception(
                                        "residual performance measure {0} line {1} has wrong number of entries, should be 5".format(
                                            pm_name, line2))
                                weight = float(raw[4])
                                obsval = float(raw[3])

                            try:
                                n = int(raw[2])
                            except:
                                raise Exception("error casting node info info on line {0}: '{1}'".format(count, line2))

                            # if there is a reduced node scheme
                            if len(nuser) > 1:
                                nn = np.where(nuser == n)[0]
                                if nn.shape[0] != 1:
                                    raise Exception("node num {0} not in reduced node num".format(n))
                                pm_entries.append(PerfMeasRecord(kper, kstp, nn, weight=weight, obsval=obsval))
                            else:
                                pm_entries.append(
                                    PerfMeasRecord(kper, kstp, n - 1, weight=weight, obsval=obsval))

                    if len(pm_entries) == 0:
                        raise Exception("no entries found for PM {0}".format(pm_name))

                    if pm_name in [pm._name for pm in self._performance_measures]:
                        raise Exception("PM {0} multiply defined".format(pm_name))
                    self._performance_measures.append(
                        PerfMeas(pm_name, pm_type, pm_entries, self.is_structured, self.verbose_level))


                else:
                    raise Exception("unrecognized adj file input on line {0}: '{1}'".format(count, line))
        if len(self._performance_measures) == 0:
            raise Exception("no PMs found in adj file")

    @staticmethod
    def get_model_names_from_mfsim(sim_ws):
        """return the model names from an mfsim.nam file

        Parameters:
            sim_ws (str): the simulation path

        Returns:
            dict,dict: a pair of dicts, first is model-name:model-type (e.g. {"gwf-1":"gwf"},
                the second is model namfile: model-type (e.g. {"gwf-1":"gwf_1.nam"})

        """
        sim_nam = os.path.join(sim_ws, "mfsim.nam")
        if not os.path.exists(sim_nam):
            raise Exception("simulation nam file '{0}' not found".format(sim_nam))
        model_dict = {}
        namfile_dict = {}
        with open(sim_nam, 'r') as f:
            while True:
                line = f.readline()
                if line == "":
                    raise EOFError("EOF when looking for 'models' block")
                if line.strip().lower().startswith("begin") and "models" in line.lower():
                    while True:
                        line2 = f.readline()
                        if line2 == "":
                            raise EOFError("EOF when reading 'models' block")
                        elif line2.strip().lower().startswith("end") and "models" in line2.lower():
                            break
                        raw = line2.strip().lower().split()
                        if raw[-1] in model_dict:
                            raise Exception("duplicate model name found: '{0}'".format(raw[-1]))
                        model_dict[raw[2]] = raw[0]
                        namfile_dict[raw[2]] = raw[1]
                    break
        return model_dict, namfile_dict

    @staticmethod
    def get_package_names_from_gwfname(gwf_nam_file):

        if not os.path.exists(gwf_nam_file):
            raise Exception("gwf nam file '{0}' not found".format(gwf_nam_file))
        package_dict = {}
        count_dict = {}
        with open(gwf_nam_file, 'r') as f:
            while True:
                line = f.readline()
                if line == "":
                    raise EOFError("EOF when looking for 'packages' block")
                if line.strip().lower().startswith("begin") and "packages" in line.lower():
                    while True:
                        line2 = f.readline()
                        if line2 == "":
                            raise EOFError("EOF when reading 'packages' block")
                        elif line2.strip().lower().startswith("end") and "packages" in line2.lower():
                            break
                        raw = line2.strip().lower().split()
                        if raw[0].startswith("#"):
                            continue
                        if "#" in line2:
                            raw = line2.split("#")[0].lower().split()
                        if len(raw) < 2:
                            raise Exception("wrong number of items on line: {0}".format(line2))
                        tag_name = None
                        if len(raw) > 2:
                            tag_name = raw[2]
                        package_type = raw[0]
                        if package_type not in count_dict:
                            count_dict[package_type] = 1

                        if package_type not in package_dict:
                            package_dict[package_type] = []
                        filename = raw[1]
                        if tag_name is None:
                            tag_name = package_type.replace("6", "") + "-{0}".format(count_dict[package_type])
                        package_dict[package_type].append(tag_name)
                        count_dict[package_type] += 1

                    break
        return package_dict

    def _write_group_to_hdf(self, hdf, group_name, data_dict, attr_dict={}):
        if group_name in hdf:
            raise Exception("group_name {0} already in hdf file".format(group_name))
        grp = hdf.create_group(group_name)
        for tag, item in data_dict.items():
            if isinstance(item, list):
                item = np.array(item)
            if isinstance(item, np.ndarray):
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
        for name, val in attr_dict.items():
            grp[name] = val

    def _open_hdf(self, tag):
        fname = tag + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".hd5"
        if os.path.exists(fname):
            # raise Exception("hdf5 file '{0}' exists somehow...".format(fname))
            os.remove(fname)
        f = h5py.File(fname, 'w')
        return f

    def _add_gwf_info_to_hdf(self, fhd):
        """
        todo: add kij info for structure grids
        """
        gwf_name = self._gwf_name
        gwf = self._gwf
        data_dict = {}
        ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
        data_dict["ihc"] = ihc
        ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
        data_dict["ia"] = ia
        ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
        data_dict["ja"] = ja
        jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
        data_dict["jas"] = jas
        cl1 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
        data_dict["cl1"] = cl1
        cl2 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
        data_dict["cl2"] = cl2
        hwva = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)
        data_dict["hwva"] = hwva
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        data_dict["top"] = top
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        data_dict["bot"] = bot
        iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
        data_dict["iac"] = iac
        icelltype = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "ICELLTYPE", gwf)
        data_dict["icelltype"] = icelltype

        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
        data_dict["area"] = area
        iconvert = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)
        data_dict["iconvert"] = iconvert
        storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
        data_dict["storage"] = storage
        nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "NODEUSER", gwf) - 1
        data_dict["nodeuser"] = nodeuser
        nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
        data_dict["nnodes"] = nnodes
        self._write_group_to_hdf(fhd, "gwf_info", data_dict)

    @staticmethod
    def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
        # todo: upstream weighting - could use height1 and height2 to check...
        # todo: vertically staggered
        d = (width * cl1 * height1 * (height2 ** 2) * (k2 ** 2)) / (
                ((cl2 * height1 * k1) + ((cl1 * height2 * k2))) ** 2)
        return d

    @staticmethod
    def _dconddvk(k1, height1, sat1, k2, height2, sat2, area, vcond_12):
        """need to think about how k1 and k2 are combined to
        form the average k between the two cells
        from MH:
        dcond_n,m / dk_n,m = cond_n,m**2 / ((area * k_n,m**2)/(0.5*(top_n - bot_n)))

        todo: VARIABLE CV and DEWATER options

        """

        # condsq = (1./((1./((area*k1)/(0.5*(top1-bot1)))) + (1./((area*k2)/(0.5*(top2-bot2))))))**2
        # return condsq / ((area * k1**2)/(0.5*(top1-bot1)))
        d = (sat1 * vcond_12 ** 2) / (((area * k1) ** 2) / (0.5 * (height1)))
        return d

    @staticmethod
    def derivative_conductance_k33(k1, k2, w1, w2, area):
        d = - 2.0 * w1 * area / ((w1 + w2 * k1 / k2) ** 2)
        return d

    @staticmethod
    def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
        d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
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
            value = self.smooth_sat(sat1)
        else:
            value = self.smooth_sat(sat2)
        return value

    @staticmethod
    def _d_smooth_sat_dh(sat, h1, h2, top, bot):
        value = 0.0
        if h1 >= h2:
            value = self.d_smooth_sat_dh(sat, top, bot)
        return value

    @staticmethod
    def dresdk_h(gwf_name, gwf, sat, head, is_newton=False):
        """partial of residual  WRT K times h.  just need to
        mult times lambda in PerfMeas.solve_adjoint()
        """

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

        result33 = np.zeros_like(lamb)
        result = np.zeros_like(lamb)

        k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
        k22 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K22", gwf)
        assert np.all(k11 == k22)
        k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)

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
                    dconddk33 = Mf6Adj._dconddhk(k33[node], k33[mnode], 0.5 * height1, 0.5 * height2, hwva[jj],
                                                 1.0, 1.0)
                    t2 = dconddk33 * (head[mnode] - head[node])
                    sum1 += t2

                else:
                    if is_newton:
                        dconddk = Mf6Adj._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj], height1,
                                                   height2)
                        SF = Mf6Adj._smooth_sat(sat_mod[node], sat_mod[mnode], head[node], head[mnode])

                    else:
                        dconddk = Mf6Adj._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj],
                                                   height1 * sat_mod[node], height2 * sat_mod[mnode])
                        SF = 1.0

                    t1 = SF * dconddk * (head[mnode] - head[node])
                    sum2 += t1

            result33[node] = sum1
            result[node] = sum2
        return result, result33

    @staticmethod
    def dresdss_h(gwf_name, gwf, head, head_old, dt, sat, sat_old):
        """partial of residual wrt ss times h.  Just need to mult
        times lambda in the PerfMeas.solve_adjoint()
        """
        top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
        bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
        area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
        iconvert = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)
        ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1

        # handle iconvert
        sat_mod = sat.copy()
        sat_mod[iconvert == 0] = 1.0
        sat_old_mod = sat_old.copy()
        sat_old_mod[iconvert == 0] = 1.0

        result = np.zeros_like(lamb)

        for node in range(len(ia) - 1):
            dSC1 = area[node] * (top[node] - bot[node])
            term = (dSC1 / dt) * (sat_old_mod[node] * head_old[node] - sat_mod[node] * head[node]) + (dSC1 / dt) * bot[
                node] * (sat_mod[node] - sat_old_mod[node]) + (dSC1 / (2.0 * dt)) * (top[node] - bot[node]) * (
                               (sat_mod[node]) ** 2 - (sat_old_mod[node]) ** 2)
            #
            result[node] = term
        return result

    @staticmethod
    def dadk(gwf_name, gwf, sat, amat):
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
        # k22 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K22", gwf)
        # assert np.all(k11 == k22)
        k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)

        # vector form of the calcs:
        k11conn1 = np.array([k11[node] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                             range(offset, offset + ncon)])
        k11conn2 = np.array([k11[ja[ii]] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                             range(offset, offset + ncon)])
        k33conn1 = np.array([k33[node] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                             range(offset, offset + ncon)])
        k33conn2 = np.array([k33[ja[ii]] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                             range(offset, offset + ncon)])
        heightconn1 = np.array([height[node] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                                range(offset, offset + ncon)])
        heightconn2 = np.array([height[ja[ii]] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                                range(offset, offset + ncon)])
        satconn1 = np.array([sat[node] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                             range(offset, offset + ncon)])
        satconn2 = np.array([sat[ja[ii]] for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
                             range(offset, offset + ncon)])
        hwvaconn1 = np.array(
            [hwva[jas[ii]] if ii != 0 else 0 for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
             range(offset, offset + ncon)])
        hwvaconn2 = np.array(
            [hwva[jas[ii]] if ii != 0 else 0 for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
             range(offset, offset + ncon)])
        cl1conn1 = np.array(
            [cl1[jas[ii]] if ii != 0 else 0 for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
             range(offset, offset + ncon)])
        cl2conn2 = np.array(
            [cl2[jas[ii]] if ii != 0 else 0 for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
             range(offset, offset + ncon)])
        ihcconn1 = np.array(
            [ihc[jas[ii]] if ii != 0 else 0 for node, (offset, ncon) in enumerate(zip(ia, iac)) for ii in
             range(offset, offset + ncon)])
        d_mat_k11_temp = Mf6Adj._dconddhk(k11conn1, k11conn2, cl1conn1, cl2conn2, hwvaconn1, heightconn1 * satconn1,
                                          heightconn2 * satconn2)
        d_mat_k11_temp[ihcconn1 == 0] = 0
        d_mat_k11_temp[ia[:-1]] = 0
        sums = np.array(
            [d_mat_k11_temp[offset:offset + ncon].sum() for node, (offset, ncon) in enumerate(zip(ia, iac))])
        d_mat_k11_temp[ia[:-1]] -= sums

        d_mat_k33_temp = Mf6Adj.derivative_conductance_k1(k33conn1, k33conn2, heightconn1, heightconn2,
                                                          cl1conn1 + cl2conn2, hwvaconn1)
        d_mat_k33_temp[ihcconn1 != 0] = 0
        d_mat_k33_temp[ia[:-1]] = 0
        sums = np.array(
            [d_mat_k33_temp[offset:offset + ncon].sum() for node, (offset, ncon) in enumerate(zip(ia, iac))])
        d_mat_k33_temp[ia[:-1]] -= sums

        # original form
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
                        # v1 = Mf6Adj._dconddvk(k33[node],height1,sat[node],k33[mnode],
                        #							height2,sat[mnode],hwva[jj],amat[jj])
                        # def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
                        #	d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
                        #	return d
                        v2 = Mf6Adj.derivative_conductance_k1(k33[node], k33[mnode], height1, height2,
                                                              cl1[jj] + cl2[jj], hwva[jj])
                        # derivative_conductance_k33(k1, k2, w1, w2, area)
                        # v3 = Mf6Adj.derivative_conductance_k33(k33[node],k33[mnode],height1, height2, hwva[jj])

                        d_mat_k33[ia[node] + pp] += v2
                        # d_mat_k123[ia[node]+pp] += v3
                        sum1 += v2
                        pp += 1

                    else:
                        v1 = Mf6Adj._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj],
                                              height1 * sat_mod[node], height2 * sat_mod[mnode])
                        # v1 = Mf6Adj._dconddhk(k11[node], k11[mnode], cl1[jj], cl2[jj], hwva[jj],
                        #							height1, height2)
                        # v2 = -Mf6Adj.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj]+cl2[jj], cl1[jj]+cl2[jj], hwva[jj],height2*sat_mod[mnode])
                        # v2 = Mf6Adj.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj],cl2[jj], hwva[jj],height1)

                        d_mat_k11[ia[node] + pp] += v1
                        # d_mat_k123[ia[node]+pp] += v1
                        sum2 += v1
                        pp += 1

                d_mat_k11[ia[node]] = -sum2
                d_mat_k33[ia[node]] = -sum1
            # d_mat_k22[ia[node]] = -sum3
            # d_mat_k123[ia[node]] = d_mat_k11[ia[node]] + d_mat_k22[ia[node]] + d_mat_k33[ia[node]]

        return d_mat_k11, -d_mat_k33

    def solve_gwf(self, verbose=True, _force_k_update=False, _sp_pert_dict=None):
        """solve the flow across the modflow sim times

        todo: move to hdf5

        """
        if self._gwf is None:
            raise Exception("gwf is None")
            self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
        fhd = self._open_hdf(self._gwf_name)
        sim_start = datetime.now()
        if verbose:
            print("...starting flow solution at {0}".format(sim_start.strftime(DT_FMT)))
        # get current sim time
        ctime = self._gwf.get_current_time()
        # get ending sim time
        etime = self._gwf.get_end_time()
        # max number of iterations
        max_iter = self._gwf.get_value(self._gwf.get_var_address("MXITER", "SLN_1"))
        # let's do it!
        num_fails = 0
        self._amat = {}
        self._head = {}
        self._kperkstp = []
        self._deltat = {}
        self._iss = {}
        self._sat = {}
        self._sat_old = {}
        sat_old = None
        visited = list()
        ctimes = []
        dts = []
        kpers, kstps = [], []

        is_newton = self._gwf.get_value(self._gwf.get_var_address("INEWTON", self._gwf_name))[0]

        # nodekchange = self._gwf.get_value_ptr(self._gwf.get_var_address("NODEKCHANGE", self._gwf_name, "NPF"))
        # k11 = self._gwf.get_value_ptr(self._gwf.get_var_address("K11", self._gwf_name, "NPF"))
        # k11input = self._gwf.get_value_ptr(self._gwf.get_var_address("K11INPUT", self._gwf_name, "NPF"))
        # condsat = self._gwf.get_value_ptr(self._gwf.get_var_address("CONDSAT", self._gwf_name, "NPF"))

        self._sp_package_data = {}
        while ctime < etime:
            sol_start = datetime.now()
            # the length of this sim time
            dt = self._gwf.get_time_step()
            # prep the current time step
            self._gwf.prepare_time_step(dt)
            kiter = 0
            # prep to solve
            stress_period = self._gwf.get_value(self._gwf.get_var_address("KPER", "TDIS"))[0]
            time_step = self._gwf.get_value(self._gwf.get_var_address("KSTP", "TDIS"))[0]
            kper, kstp = stress_period - 1, time_step - 1
            kperkstp = (kper, kstp)

            # this is to force mf6 to update cond sat using the k11 and k33 arrays
            # which is needed for the perturbation testing
            if kper == 0 and kstp == 0 and _force_k_update:
                kchangeper = self._gwf.get_value_ptr(self._gwf.get_var_address("KCHANGEPER", self._gwf_name, "NPF"))
                kchangestp = self._gwf.get_value_ptr(self._gwf.get_var_address("KCHANGESTP", self._gwf_name, "NPF"))
                kchangestp[0] = time_step
                kchangeper[0] = stress_period
                nodekchange = self._gwf.get_value_ptr(self._gwf.get_var_address("NODEKCHANGE", self._gwf_name, "NPF"))
                nodekchange[:] = 1

            # apply any boundary condition perturbation info
            if _sp_pert_dict is not None:
                if _sp_pert_dict["kperkstp"] != kperkstp:
                    continue
                addr = ["BOUND", self._gwf_name, _sp_pert_dict["packagename"].upper()]
                wbaddr = self._gwf.get_var_address(*addr)
                bnd_ptr = self._gwf.get_value_ptr(wbaddr)
                wbaddr = self._gwf.get_var_address("NODELIST", self._gwf_name, _sp_pert_dict["packagename"].upper())
                nodelist = self._gwf.get_value_ptr(wbaddr)
                idx = np.where(nodelist == _sp_pert_dict["node"])
                bnd_ptr[idx] = _sp_pert_dict["bound"]

            self._gwf.prepare_solve(1)
            sat = self._gwf.get_value(self._gwf.get_var_address("SAT", self._gwf_name, "NPF"))
            if sat_old is None:
                sat_old = self._gwf.get_value(self._gwf.get_var_address("SAT", self._gwf_name, "NPF"))

            # solve until converged
            while kiter < max_iter:
                convg = self._gwf.solve(1)
                if convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    if verbose:
                        print(
                            "flow stress period,time step {0},{1} converged with {2} iters, took {3:10.5G} mins".format(
                                stress_period, time_step, kiter, td))
                    break
                kiter += 1

            if not convg:
                td = (datetime.now() - sol_start).total_seconds() / 60.0
                if verbose:
                    print(
                        "flow stress period,time step {0},{1} did not converge, {2} iters, took {3:10.5G} mins".format(
                            stress_period, time_step, kiter, td))
                num_fails += 1
            try:
                self._gwf.finalize_solve(1)
            except:
                pass

            self._gwf.finalize_time_step()
            # update current sim time
            ctime = self._gwf.get_current_time()
            dt1 = self._gwf.get_time_step()

            ctimes.append(ctime)
            dts.append(dt1)
            kpers.append(kper)
            kstps.append(kstp)

            if kperkstp in visited:
                raise Exception("{0} already visited".format(kperkstp))
            visited.append(kperkstp)

            amat = self._gwf.get_value(self._gwf.get_var_address("AMAT", "SLN_1")).copy()
            data_dict = {"amat": amat}

            self._amat[kperkstp] = amat
            head = self._gwf.get_value(self._gwf.get_var_address("X", self._gwf_name.upper()))
            self._head[kperkstp] = head
            data_dict["head"] = head

            head_old = self._gwf.get_value(self._gwf.get_var_address("XOLD", self._gwf_name.upper()))
            self._head_old[kperkstp] = head_old
            data_dict["head_old"] = head_old

            # k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
            k11 = self._gwf.get_value(self._gwf.get_var_address("K11", self._gwf_name.upper(), "NPF"))
            data_dict["k11"] = k11
            # k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)
            k33 = self._gwf.get_value(self._gwf.get_var_address("K33", self._gwf_name.upper(), "NPF"))
            data_dict["k33"] = k33
            # condsat = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "CONDSAT", gwf)
            condsat = self._gwf.get_value(self._gwf.get_var_address("CONDSAT", self._gwf_name.upper(), "NPF"))
            data_dict["condsat"] = condsat

            dadk11, dadk33 = Mf6Adj.dadk(self._gwf_name, self._gwf, sat, amat)
            data_dict["dadk11"] = dadk11
            data_dict["dadk33"] = dadk33

            self._kperkstp.append(kperkstp)
            self._deltat[kperkstp] = dt1

            iss = self._gwf.get_value(self._gwf.get_var_address("ISS", self._gwf_name.upper()))
            self._iss[kperkstp] = iss
            data_dict["iss"] = iss

            sat = self._gwf.get_value(self._gwf.get_var_address("SAT", self._gwf_name, "NPF"))
            self._sat[kperkstp] = sat
            self._sat_old[kperkstp] = sat_old
            data_dict["sat"] = sat
            data_dict["sat_old"] = sat_old

            dresdk_h, dresdk33_h = Mf6Adj.dresdk_h(self._gwf_name,self._gwf,sat,head,is_newton=is_newton)
            data_dict["dresdk_h"] = dresdk_h
            data_dict["dresdk33_h"] = dresdk33_h

            dresdss_h = Mf6Adj.dresdss_h(self._gwf_name,self._gwf,head,head_old,sat,sat_old)
            data_dict["dresdss_h"] = dresdss_h

            sat_old = sat.copy()
            for package_type in self._gwf_package_types:
                if package_type in self._gwf_package_dict:
                    if package_type not in self._sp_package_data:
                        self._sp_package_data[package_type] = {}
                    for tag in self._gwf_package_dict[package_type]:
                        nbound = self._gwf.get_value(self._gwf.get_var_address("NBOUND", self._gwf_name, tag.upper()))[
                            0]
                        if nbound > 0:
                            if kperkstp in self._sp_package_data[package_type]:
                                if len(self._gwf_package_dict[package_type]) == 1:
                                    raise Exception("kperkstp '{0}' already in sp_package_data".format(str(kperkstp)))
                                else:
                                    pass
                            else:
                                self._sp_package_data[package_type][kperkstp] = []
                            nodelist = self._gwf.get_value_ptr(
                                self._gwf.get_var_address("NODELIST", self._gwf_name, tag.upper()))
                            bound = self._gwf.get_value_ptr(
                                self._gwf.get_var_address("BOUND", self._gwf_name, tag.upper()))
                            hcof = self._gwf.get_value_ptr(
                                self._gwf.get_var_address("HCOF", self._gwf_name, tag.upper()))
                            rhs = self._gwf.get_value_ptr(self._gwf.get_var_address("RHS", self._gwf_name, tag.upper()))
                            for i in range(nbound):
                                # note bound is an array!
                                self._sp_package_data[package_type][kperkstp].append(
                                    {"node": nodelist[i], "bound": bound[i],
                                     "hcof": hcof[i], "rhs": rhs[i], "packagename": tag})
                            data_dict[tag] = {"ptype": package_type, "nodelist": nodelist, "bound": bound}
            attr_dict = {"ctime": ctime, "dt": dt1, "kper": kper, "kstp": kstp}
            self._write_group_to_hdf(fhd, group_name="kper:{0}_kstp:{1}".format(kper, kstp), data_dict=data_dict,
                                     attr_dict=attr_dict)

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0
        if verbose:
            print("\n...flow solution finished at {0}, took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT), td))
            if num_fails > 0:
                print("...failed to converge {0} times".format(num_fails))
            print("\n")

        self._write_group_to_hdf(fhd, "aux", {"totime": ctimes, "dt": dts, "kper": kpers, "kstp": kstps})
        self._add_gwf_info_to_hdf(fhd)
        fhd.close()

    def solve_adjoint(self):
        if len(self._kperkstp) == 0:
            raise Exception("need to call solve_gwf() first")
        dfs = {}
        for pm in self._performance_measures:
            df = pm.solve_adjoint(self._kperkstp, self._iss, self._deltat, self._amat,
                                  self._head, self._head_old, self._sat, self._sat_old, self._gwf,
                                  self._gwf_name, self._structured_mg, self._sp_package_data)
            dfs[pm.name] = df
        return dfs

    def _initialize_gwf(self, lib_name, flow_dir):
        # instantiate the flow model api
        if self._gwf is not None:
            self._gwf.finalize()
            self._gwf = None
        gwf = modflowapi.ModflowApi(os.path.join(flow_dir, lib_name), working_directory=flow_dir)
        gwf.initialize()
        return gwf

    def finalize(self):
        """close the api and file handles

        """
        try:
            self._gwf.finalize()
        except:
            pass
        self._gwf = None

    def _perturbation_test(self, pert_mult=1.01):
        """run the pertubation testing - this is for dev and testing only"""

        self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
        gwf_name = self._gwf_name.upper()

        self.solve_gwf()
        base_results = {pm.name: pm.solve_forward(self._head) for pm in self._performance_measures}
        assert len(base_results) == len(self._performance_measures)

        addr = ["NODEUSER", gwf_name, "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nuser = self._gwf.get_value(wbaddr) - 1
        if len(nuser) == 1:
            nuser = np.arange(self._head[self._kperkstp[0]].shape[0], dtype=int)

        kijs = None
        if self.is_structured:
            kijs = self._structured_mg.get_lrc(list(nuser))

        addr = ["NLAY", gwf_name, "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nlay = self._gwf.get_value(wbaddr)[0]

        dfs = []

        # boundary condition perturbations
        org_sp_package_data = self._sp_package_data.copy()
        for paktype, pdict in org_sp_package_data.items():
            epsilons = []
            bound_idx = []
            nodes = []
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            print("running perturbations for ", paktype)
            for kk, infolist in pdict.items():
                for infodict in infolist:
                    bnd_items = infodict["bound"].shape[0]
                    for ibnd in range(bnd_items):
                        new_bound = infodict["bound"].copy()
                        delt = new_bound[ibnd] * pert_mult
                        epsilons.append(delt - new_bound[ibnd])
                        new_bound[ibnd] = delt
                        pakname = infodict["packagename"]
                        pert_dict = {"kperkstp": kk, "packagename": pakname, "node": infodict["node"],
                                     "bound": new_bound}
                        # print(pert_dict)
                        self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
                        self.solve_gwf(verbose=False, _sp_pert_dict=pert_dict)
                        pert_results = {pm.name: (pm.solve_forward(self._head) - base_results[pm.name]) / epsilons[-1]
                                        for pm in self._performance_measures}
                        for pm, result in pert_results.items():
                            pert_results_dict[pm].append(result)
                        bound_idx.append(ibnd)
                        nodes.append(infodict["node"])
            df = pd.DataFrame(pert_results_dict)
            df.loc[:, "node"] = nodes
            df.loc[:, "epsiloon"] = epsilons
            df.index = df.pop("node")
            if kijs is not None:
                for idx, lab in zip([0, 1, 2], ["k", "i", "j"]):
                    df.loc[:, lab] = df.index.map(lambda x: kijs[x - 1][idx])
            dfs.append(df)

        # property perturbations
        address = [["K11", gwf_name, "NPF"]]
        if nlay > 1:
            address.append(["K33", gwf_name, "NPF"])

        if PerfMeas.has_sto_iconvert(self._gwf):
            address.append(["SS", gwf_name, "STO"])
        for addr in address:
            print("running perturbations for ", addr)
            pert_results_dict = {pm.name: [] for pm in self._performance_measures}
            wbaddr = self._gwf.get_var_address(*addr)
            inodes = self._gwf.get_value_ptr(wbaddr).shape[0]
            epsilons = []

            for inode in range(inodes):
                self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
                pert_arr = self._gwf.get_value_ptr(wbaddr)

                delt = pert_arr[inode] * pert_mult
                epsilons.append(delt - pert_arr[inode])
                pert_arr[inode] = delt
                # if "K11" in wbaddr.upper():
                #     wbaddr1 = wbaddr.upper().replace("K11","K22")    
                #     pert_arr1 = self._gwf.get_value_ptr(wbaddr1)
                #     pert_arr1 = pert_arr.copy()
                #     print("K22",pert_arr1)
                # addr1 = ["IK22OVERK",gwf_name,"NPF"]
                # wbaddr1 = self._gwf.get_var_address(*addr1)
                # k22 = self._gwf.get_value_ptr(wbaddr1)
                # print("ik22 flag",k22)

                self.solve_gwf(verbose=False, _force_k_update=True)
                pert_results = {pm.name: (pm.solve_forward(self._head) - base_results[pm.name]) / epsilons[-1] for pm in
                                self._performance_measures}
                for pm, result in pert_results.items():
                    pert_results_dict[pm].append(result)

            df = pd.DataFrame(pert_results_dict)
            df.index = [nuser[inode] for inode in range(inodes)]
            df.index.name = "node"
            df.loc[:, "epsilon"] = epsilons
            if kijs is not None:
                for idx, lab in zip([0, 1, 2], ["k", "i", "j"]):
                    df.loc[:, lab] = [kij[idx] for kij in kijs]
            tag = '_'.join(addr).lower()
            df.loc[:, "addr"] = tag
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_csv("pert_results.csv")
