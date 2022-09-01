import os
import shutil
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import modflowapi
import flopy


class PerfMeasRecord(object):
	def __init__(self,kper,kstp,nnode,k=None,i=None,j=None,weight=None,obsval=None):
		self._kper = int(kper)
		self._kstp = int(kstp)
		self.kperkstp = (self._kper,self._kstp)
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
	def __init__(self,name,type,entries):
		self._name = name.lower().strip()
		self._type = type.lower().strip()
		self._entries = entries

	def solve_adjoint(self, kperkstp, iss, deltat_dict, amat_dict, head_dict, head_old_dict, sat_dict, gwf, gwf_name):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name,"DIS","NODES",gwf)[0]

		d_amat_k = self._d_amat_k(gwf_name, gwf, sat_dict)
		# top = PerfMeas.get_ptr_from_gwf(gwf_name,"DIS","TOP",gwf)
		# botm = PerfMeas.get_ptr_from_gwf(gwf_name,"DIS","BOT",gwf)
		# area = PerfMeas.get_ptr_from_gwf(gwf_name,"DIS","AREA",gwf)

		#print(hwva)
		#exit()
		# storage = np.zeros(nnodes)
		# try:
		# 	storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
		# except Exception as e:
		# 	print("no storage found, must steady state solution(s) only")
		#ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)

		lamb = np.zeros(nnodes)
		lambs = np.zeros((len(kperkstp),nnodes))
		ia = PerfMeas.get_value_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		ja = PerfMeas.get_value_from_gwf(gwf_name, "CON", "JA", gwf) - 1
		for itime,kk in enumerate(kperkstp[::-1]):
			print(kk)
			# get the derv of the PM WRT head
			dFdh = self._dFdh(kk, gwf_name, gwf, kperkstp,deltat_dict,head_dict)
			if iss[kk] == 0: #transient
				# get the derv of RHS WRT head
				drhsdh = self._drhsdh(gwf_name,gwf,deltat_dict[kk])
				rhs = (drhsdh * lamb) - dFdh
			else:
				rhs = -1 * dFdh

			amat = amat_dict[kk]
			amat_sp_t = sparse.csr_matrix((amat,ja,ia),shape=(len(ia)-1,len(ia)-1)).transpose()
			lamb = spsolve(amat_sp_t,rhs)
			#print(dFdh.shape,drhsdh.shape)
			lambs[itime,:] = lamb


	@staticmethod
	def _derv_cond(k1,k2,cl1,cl2,area):
		return - 2.0 * cl1 * area / ((cl1 + cl2 * k1 / k2) ** 2)

	def _d_amat_k(self,gwf_name,gwf, sat):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
		ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
		jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
		cl1 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
		cl2 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
		hwva = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)

		top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
		bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)

		d_mat_k11 = np.zeros(nnodes)
		d_mat_k33 = np.zeros(nnodes)

		#jwhite 30 aug 2022 - dont know how to support horizontal aniso in unstructured...

		k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
		k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)

		for node in range(nnodes):
			start_ia = ia[node]+1
			end_ia = ia[node+1]
			height1 = sat[node] * (top[node] - bot[node])
			for ii in range(start_ia,end_ia):
				mnode = ja[ii]
				height2 = sat[mnode] * (top[mnode] - bot[mnode])
				jj = jas[ii]
				iihc = ihc[jj]
				#print(node,jj,iihc,ihc.shape,hwva.shape)
				if iihc == 0: # vertical con
					#todo: deal with upstream weigthing
					#todo: deal with water table
					interface_area = hwva[jj]
					d_mat_k33[node] = PerfMeas._derv_cond(k33[node],k33[mnode],cl1[jj],cl2[jj],interface_area)
				else: #horizontal con
					width = hwva[jj]
					# todo: deal with water table
					# todo: deal with upstream weighting?
					narea = (top[node] - bot[node]) / 2.
					marea = (top[mnode] - bot[mnode]) / 2.0
					interface_area = (narea + marea) / 2.0
					d_mat_k11[node] = PerfMeas._derv_cond(k11[node],k11[mnode],cl1[jj],cl2[jj],interface_area)
		return d_mat_k11,d_mat_k33

	def _drhsdh(self, gwf_name,gwf, dt):
		top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
		bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
		area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
		storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
		drhsdh = -1. * storage * area * (top - bot) / dt
		return drhsdh

	def _dFdh(self, kk, gwf_name, gwf, kperkstp,deltat_dict,head_dict):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
		dFdh = np.zeros(nnodes)
		for pfr in self._entries:
			if pfr.kperkstp == kk:
				if self._type == "direct":
					dFdh[pfr.nnode] = -1. * pfr.weight / deltat_dict[kk]
				elif self._type == "residual":
					dFdh[pfr.nnode] = - 2.0 * pfr.weight * (head_dict[kk][pfr.nnode] - pfr.obsval)
		return dFdh


	@staticmethod
	def get_value_from_gwf( gwf_name, pak_name,prop_name, gwf):
		addr = gwf.get_var_address(prop_name, gwf_name,pak_name)
		return gwf.get_value(addr)

	@staticmethod
	def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf):
		addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
		return gwf.get_value_ptr(addr)






# class PerfMeasDirect(PerfMeas):
# 	def __init__(self,name,type,entries):
# 		super(PerfMeasDirect).__init__(name,type,entries)
#
# 	@staticmethod
# 	def parse_entry(line,count,header_line):
# 		pass
#
#
# class PerfMeasResidual(PerfMeas):
# 	def __init__(self, name, type, entries):
# 		super(PerfMeasResidual).__init__(name, type, entries)
#
# 	@staticmethod
# 	def parse_entry(line, count, header_line):
# 		pass

