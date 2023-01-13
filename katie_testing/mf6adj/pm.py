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


	def solve_adjoint(self, kperkstp, iss, deltat_dict, amat_dict, head_dict, head_old_dict, sat_dict, gwf, gwf_name,mg_structured):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name,"DIS","NODES",gwf)[0]

		lamb = np.zeros(nnodes)
		#lambs = np.zeros((len(kperkstp),nnodes))

		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1

		dfdk11 = np.zeros(nnodes)
		dfdk33 = np.zeros(nnodes)

		for itime,kk in enumerate(kperkstp[::-1]):
			print('solving',self._name,kk)
			dfdh = self._dfdh(kk, gwf_name, gwf, deltat_dict, head_dict)
			dadk11,dadk33 = self._dadk(gwf_name, gwf, sat_dict[kk])

			if iss[kk] == 0: #transient
				# get the derv of RHS WRT head
				drhsdh = self._drhsdh(gwf_name,gwf,deltat_dict[kk])
				rhs = (drhsdh * lamb) - dfdh
			else:
				rhs = -1 * dfdh

			amat = amat_dict[kk]
			amat_sp_t = sparse.csr_matrix((amat,ja,ia),shape=(len(ia)-1,len(ia)-1)).transpose()
			lamb = spsolve(amat_sp_t,rhs)
			#lambs[itime,:] = lamb

			dfdk11 += np.dot(lamb,dadk11.dot(head_dict[kk]))
			dfdk33 += np.dot(lamb, dadk33.dot(head_dict[kk]))

		self.save_array("k11",dfdk11,gwf_name,gwf,mg_structured)
		self.save_array("k33", dfdk33, gwf_name, gwf, mg_structured)


	def save_array(self,filetag,avec,gwf_name,gwf,structured_mg):
		nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name,"DIS","NODEUSER",gwf)
		nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
		
		# if not a reduced node scheme
		if len(nodeuser) == 1:
			nodeuser = np.arange(nnodes)
		if structured_mg is not None:
			kijs = structured_mg.get_lrc(list(nodeuser-1))

			arr = np.zeros((structured_mg.nlay,structured_mg.nrow,structured_mg.ncol))
			for kij,v in zip(kijs,avec):
				arr[kij] = v
			for k,karr in enumerate(arr):
				filename = filetag + "_layer{0:03d}.dat".format(k+1)
				np.savetxt(filename,karr,fmt="%15.6E")
		else:
			filename = filetag + ".dat"
			with open(filename,'w') as f:
				f.write("node,value\n")
				for n,v in zip(nodeuser,avec):
					f.write("{0},{1:15.6E}\n".format(n,v))


	def _dadk(self,gwf_name,gwf, sat):
		"""partial of A matrix WRT K
		"""
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
		d_mat_k11 = np.zeros(ja.shape[0])
		d_mat_k33 = np.zeros(ja.shape[0])

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
				if iihc == 0: # vertical con
					d_mat_k33[jj] = PerfMeas._dconddvk(k33[node],top[node],bot[node],sat[node],
													   k33[mnode],top[mnode],bot[mnode],sat[mnode],hwva[jj])
				else: #horizontal con
					d_mat_k11[jj] = PerfMeas._dconddhk(k11[node],k11[mnode],cl1[jj],cl2[jj],hwva[jj],height1,height2)
		d_mat_k11 = sparse.csr_matrix((d_mat_k11, ja, ia), shape=(len(ia) - 1, len(ia) - 1))
		d_mat_k33 = sparse.csr_matrix((d_mat_k33, ja, ia), shape=(len(ia) - 1, len(ia) - 1))
		return d_mat_k11,d_mat_k33

	@staticmethod
	def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
		# todo: upstream weighting - could use height1 and height2 to check...
		# todo: vertically staggered
		return (width * cl1 * height1 * height2 ** 2 * k2) / ((cl2 * height1 * k1) + (cl1 * height2 * k2) ** 2)

	@staticmethod
	def _dconddvk(k1,top1,bot1,sat1,k2,top2,bot2,sat2,area):
		# todo: VARIABLE CV and DEWATER options
		condsq = (1./((1./((area*k1)/(0.5*(top1-bot1)))) + (1./((area*k2)/(0.5*(top2-bot2))))))**2
		return condsq / ((area * k1**2)/(0.5*(top1-bot1)))

	def _drhsdh(self, gwf_name,gwf, dt):
		top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
		bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
		area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
		storage = PerfMeas.get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
		drhsdh = -1. * storage * area * (top - bot) / dt
		return drhsdh

	def _dfdh(self, kk, gwf_name, gwf, deltat_dict,head_dict):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
		dfdh = np.zeros(nnodes)
		for pfr in self._entries:
			if pfr.kperkstp == kk:
				if self._type == "direct":
					dfdh[pfr.nnode] = -1. * pfr.weight / deltat_dict[kk]
				elif self._type == "residual":
					dfdh[pfr.nnode] = - 2.0 * pfr.weight * (head_dict[kk][pfr.nnode] - pfr.obsval)
		return dfdh

	@staticmethod
	def get_value_from_gwf( gwf_name, pak_name,prop_name, gwf):
		addr = gwf.get_var_address(prop_name, gwf_name,pak_name)
		return gwf.get_value(addr)

	@staticmethod
	def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf):
		addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
		return gwf.get_value_ptr(addr)

