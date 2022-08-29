import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import modflowapi
import flopy

from .pm import PerfMeasRecord,PerfMeas

DT_FMT = "%Y-%m-%d %H:%M:%S"

class Mf6Adj(object):
    def __init__(self, adj_filename, lib_name, is_structured=True):
  
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
        if self._gwf_model_dict[self._gwf_name] != "gwf6":
            raise Exception("model is not a gwf6 type: {0}". \
                            format(self._gwf_model_dict[self._gwf_name]))
        self._gwf = None
        self._lib_name = lib_name
        self._flow_dir = "."
        self._gwf = self._initialize_gwf(lib_name,self._flow_dir)

        self._structured_mg = None
        self.is_structured = True  # hard coded for now...
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



    def _read_adj_file(self,):

        # clear any existing PMs
        self._performance_measures = []

        """read the adj input file

                """
        # used to detect location-pak duplicates
        current_period = -1
        current_entries = []

        addr = ["NODEUSER", self._gwf_name.upper(), "DIS"]
        wbaddr = self._gwf.get_var_address(*addr)
        nuser = self._gwf.get_value(wbaddr)

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
                        raise Exception("wrong number of entries on 'begin performance_measure' line number {0}: '{1}'".format(count,line))
                    pm_name = raw[2].strip().lower()
                    if raw[3].strip().lower() != "type":
                        raise Exception("4th entry on line {0} should be 'type', not '{1}'".format(count,raw[3]))
                    pm_type = raw[4].strip().lower()
                    if pm_type not in ["direct","residual"]:
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
                            raise Exception("a new begin block found while parsing performance_measure block '{0}'".format(line))
                        elif line2.lower().strip().startswith("end performance_measure"):
                            break

                        raw = line2.lower().strip().split()
                        kper = int(raw[0]) - 1
                        kstp = int (raw[1]) - 1
                        #todo: check limits of kper, kstp

                        if self.is_structured:
                            if len(raw) < 5:
                                raise Exception(
                                    "performance measure {0} line {1} has too few entries, need at least 5".format(
                                        pm_name, line2))
                            weight = None
                            if pm_type == "direct":
                                if len(raw) != 6:
                                    raise Exception(
                                        "direct performance measure {0} line {1} has wrong number of entries, should be 6".format(
                                            pm_name, line2))
                                weight = float(raw[5])
                            obsval = None
                            if pm_type == "residual":
                                if len(raw) != 7:
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
                            n = self._structured_mg.get_node([kij])[0] + 1
                            # if there is a reduced node scheme
                            if len(nuser) > 1:
                                nn = np.where(nuser == n)[0]
                                if nn.shape[0] != 1:
                                    raise Exception("node num {0} not in reduced node num".format(n))
                                pm_entries.append(PerfMeasRecord(kper,kstp,nn,k=kij[0],i=kij[1],j=kij[2],weight=weight,obsval=obsval))
                            else:
                                pm_entries.append(
                                    PerfMeasRecord(kper,kstp,n - 1,k=kij[0],i=kij[1],j=kij[2],weight=weight,obsval=obsval))
                        else:
                            raise NotImplementedError("only structured grids currently supported")
                    if len(pm_entries) == 0:
                        raise Exception("no entries found for PM {0}".format(pm_name))

                    if pm_name in [pm._name for pm in self._performance_measures]:
                        raise Exception("PM {0} multiply defined".format(pm_name))
                    self._performance_measures.append(PerfMeas(pm_name,pm_type,pm_entries))


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

    def solve_gwf(self):
        """solve the flow across the modflow sim times

        """
        if self._gwf is None:
            # raise Exception("gwf is None")
            self._gwf = self._initialize_gwf(self._lib_name, self._flow_dir)
        sim_start = datetime.now()
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

        while ctime < etime:
            sol_start = datetime.now()
            # the length of this sim time
            dt = self._gwf.get_time_step()
            # prep the current time step
            self._gwf.prepare_time_step(dt)
            kiter = 0
            # prep to solve
            self._gwf.prepare_solve(1)
            # the current one-based stress period number
            stress_period = self._gwf.get_value(self._gwf.get_var_address("KPER", "TDIS"))[0]
            time_step = self._gwf.get_value(self._gwf.get_var_address("KSTP", "TDIS"))[0]
            # solve until converged
            while kiter < max_iter:
                convg = self._gwf.solve(1)
                if convg:
                    td = (datetime.now() - sol_start).total_seconds() / 60.0
                    print("flow stress period,time step {0},{1} converged with {2} iters, took {3:10.5G} mins".format(
                        stress_period, time_step, kiter, td))
                    break

                kiter += 1

            if not convg:
                td = (datetime.now() - sol_start).total_seconds() / 60.0
                print("flow stress period,time step {0},{1} did not converge, {2} iters, took {3:10.5G} mins".format(
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
            amat = self._gwf.get_value(self._gwf.get_var_address("AMAT", "SLN_1"))
            kper,kstp = stress_period - 1,time_step - 1
            kperkstp = (kper,kstp)
            self._amat[kperkstp] = amat
            head = self._gwf.get_value(self._gwf.get_var_address("X", self._gwf_name.upper()))
            self._head[kperkstp] = head
            head_old = self._gwf.get_value(self._gwf.get_var_address("XOLD", self._gwf_name.upper()))
            self._head_old[kperkstp] = head_old
            self._kperkstp.append((kperkstp))
            self._deltat[(kper,kstp)] = dt1

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0
        print("\n...flow solution finished at {0}, took: {1:10.5G} mins".format(sim_end.strftime(DT_FMT), td))
        if num_fails > 0:
            print("...failed to converge {0} times".format(num_fails))
        print("\n")

    def solve_adjoint(self):
        if len(self._kperkstp) == 0:
            raise Exception("need to call solve_gwf() first")
        for pm in self._performance_measures:
            pm.solve_adjoint(self._kperkstp,self._deltat,self._amat,self._head,self._head_old, self._gwf)

    def _initialize_gwf(self,lib_name,flow_dir):
        # instantiate the flow model api
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

