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
	"""todo: preprocess all the connectivity in to faster look dict containers, 
	including nnode to kij info for structured grids

	todo: convert several class methods to static methods - this might make testing easier

	todo: add a no-data value var to fill empty spots in output arrays.  currently using zero :(
	
	"""
	def __init__(self,name,type,entries,is_structured,verbose_level=1):
		self._name = name.lower().strip()
		self._type = type.lower().strip()
		self._entries = entries
		self.is_structured = is_structured
		self.verbose_level = int(verbose_level)


	def solve_adjoint(self, kperkstp, iss, deltat_dict, amat_dict, head_dict, head_old_dict, 
		   			  sat_dict, gwf, gwf_name,mg_structured, gwf_package_dict):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name,"DIS","NODES",gwf)[0]

		lamb = np.zeros(nnodes)
		
		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1

		comp_k33_sens = np.zeros(nnodes)
		comp_k_sens = np.zeros(nnodes)
		comp_ss_sens = np.zeros(nnodes)

		comp_welq_sens = None
		comp_ghb_head_sens = None
		comp_ghb_cond_sens = None

		if "wel6" in gwf_package_dict:
			comp_welq_sens = np.zeros(nnodes)
		if "ghb6" in gwf_package_dict:
			comp_ghb_head_sens = np.zeros(nnodes)
			comp_ghb_cond_sens = np.zeros(nnodes)
			
		for itime,kk in enumerate(kperkstp[::-1]):
			itime = kk[0]
			print('solving',self._name,"(kper,kstp)",kk)
			dfdh = self._dfdh(kk, gwf_name, gwf, deltat_dict, head_dict)
			dadk11,dadk22,dadk33,dadk123 = self._dadk(gwf_name, gwf, sat_dict[kk],amat_dict[kk])
				
			if iss[kk] == 0: #transient
				# get the derv of RHS WRT head
				drhsdh = self._drhsdh(gwf_name,gwf,deltat_dict[kk])
				rhs = (drhsdh * lamb) - dfdh
			else:
				rhs = dfdh
			
			amat = amat_dict[kk]
			amat_sp = sparse.csr_matrix((amat,ja.copy(),ia.copy()),shape=(len(ia)-1,len(ia)-1))
			amat_sp_t = amat_sp.transpose()			
			lamb = spsolve(amat_sp_t,rhs)
				
			k_sens = self.lam_dAdk_h(gwf_name,gwf,lamb, dadk123,head_dict[kk])
			k33_sens = self.lam_dAdk_h(gwf_name,gwf,lamb, dadk33,head_dict[kk])
			
			ss_sens = self.sens_ss_indirect(gwf_name,gwf,lamb, head_dict[kk],head_old_dict[kk],deltat_dict[kk])
			comp_k_sens += k_sens
			comp_k33_sens += k33_sens
			comp_ss_sens += ss_sens

			if "wel6" in gwf_package_dict and kk in gwf_package_dict["wel6"]:
				sens_welq = self.lam_drhs_dqwel(lamb,gwf_package_dict["wel6"][kk])
				if self.verbose_level > 1:
					self.save_array("sens_welq_kper{0:05d}".format(itime),sens_welq,gwf_name,gwf,mg_structured)
				comp_welq_sens += sens_welq				

			if "ghb6" in gwf_package_dict and kk in gwf_package_dict["ghb6"]:
				sens_ghb_head,sens_ghb_cond = self.lam_drhs_dghb(lamb,head_dict[kk],gwf_package_dict["ghb6"][kk])
				if self.verbose_level > 1:
					self.save_array("sens_ghbhead_kper{0:05d}".format(itime),sens_ghb_head,gwf_name,gwf,mg_structured)
					self.save_array("sens_ghbcond_kper{0:05d}".format(itime),sens_ghb_cond,gwf_name,gwf,mg_structured)
				comp_ghb_head_sens += sens_ghb_head
				comp_ghb_cond_sens += sens_ghb_cond
			
			if "rch" in gwf_package_dict and kk in gwf_package_dict["rch6"]:
				pass

			if "rcha" in gwf_package_dict and kk in gwf_package_dict["rcha6"]:
				pass

			if self.verbose_level > 1:
				self.save_array("adjstates_kper{0:05d}".format(itime),lamb,gwf_name,gwf,mg_structured)
				self.save_array("sens_k33_kper{0:05d}".format(itime), k33_sens, gwf_name, gwf, mg_structured)
				self.save_array("sens_k_kper{0:05d}".format(itime),k_sens,gwf_name,gwf,mg_structured)
				self.save_array("sens_ss_kper{0:05d}".format(itime),ss_sens,gwf_name,gwf,mg_structured)
				self.save_array("head_kper{0:05d}".format(itime),head_dict[kk],gwf_name,gwf,mg_structured)
				if self.verbose_level > 2:
					self.save_array("dadk11_kper{0:05d}".format(itime),dadk11,gwf_name,gwf,mg_structured)
					self.save_array("dadk22_kper{0:05d}".format(itime),dadk22,gwf_name,gwf,mg_structured)
					self.save_array("dadk33_kper{0:05d}".format(itime),dadk33,gwf_name,gwf,mg_structured)
					self.save_array("dadk123_kper{0:05d}".format(itime),dadk123,gwf_name,gwf,mg_structured)
					np.savetxt("pm-{0}_amattodense_kper{1:04d}.dat".format(self._name,itime),amat_sp_t.todense(),fmt="%15.6E")
					np.savetxt("pm-{0}_amat_kper{1:04d}.dat".format(self._name,itime),amat,fmt="%15.6E")
					np.savetxt("pm-{0}_rhs_kper{1:04d}.dat".format(self._name,itime),rhs,fmt="%15.6E")
					np.savetxt("pm-{0}_ia_kper{1:04d}.dat".format(self._name,itime),ia)
					np.savetxt("pm-{0}_ja_kper{1:04d}.dat".format(self._name,itime),ja)
					for arr,tag in zip([dadk11,dadk22,dadk33,dadk123],["dadk11","dadk22","dadk33","dadk123"]):
						np.savetxt("pm-{0}_{1}_kper{2:05d}.dat".format(self._name,tag,itime),arr,fmt="%15.6E")
		
		self.save_array("comp_sens_k33", comp_k33_sens, gwf_name, gwf, mg_structured)
		self.save_array("comp_sens_k",comp_k_sens,gwf_name,gwf,mg_structured)
		self.save_array("comp_sens_ss",comp_ss_sens,gwf_name,gwf,mg_structured)
		

	def lam_drhs_dghb(self,lamb,head,sp_dict):
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

		return result_head,result_cond


	def lam_drhs_dqwel(self,lamb, sp_dict):
		result = np.zeros_like(lamb)
		for id in sp_dict:
			n = id["node"] - 1
			result[n] = lamb[n]
		return result	
	

	def save_array(self,filetag,avec,gwf_name,gwf,structured_mg):
		nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name,"DIS","NODEUSER",gwf)-1
		nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
		jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf)
		filetag = "pm-"+self._name + "_" + filetag
		# if not a reduced node scheme
		if len(nodeuser) == 1:
			nodeuser = np.arange(nnodes)
		if structured_mg is not None:
			kijs = structured_mg.get_lrc(list(nodeuser))

			arr = np.zeros((structured_mg.nlay,structured_mg.nrow,structured_mg.ncol))
			for kij,v in zip(kijs,avec):
				arr[kij] = v
			for k,karr in enumerate(arr):
				filename = filetag + "_k{0:03d}.dat".format(k)
				np.savetxt(filename,karr,fmt="%15.6E")
		else:
			filename = filetag + ".dat"
			if avec.shape[0] == nodeuser.shape[0]:
				rarr = np.array((nodeuser,avec)).transpose()
			elif avec.shape[0] == jas.shape[0]:
				rarr = np.array((jas,avec)).transpose()
			else:
				raise Exception("unrecognized unstructed vector length: {0} for filename {1}".format(avec.shape[0],filename))
			#print(rarr)
			np.savetxt(filename,rarr,fmt=["%10d","%15.6E"])
			# with open(filename,'w') as f:
			# 	f.write("node,value\n")
			# 	for n,v in zip(nodeuser,avec):
			# 		f.write("{0},{1:15.6E}\n".format(n,v))


	def _dadk(self,gwf_name,gwf, sat, amat):
		"""partial of A matrix WRT K
		"""
		is_chd = False
		chd_list = []
		names = list(gwf.get_input_var_names())
		chds = [name for name in names if 'CHD' in name and 'NODELIST' in name]
		for name in chds:
			chd = np.array(PerfMeas.get_ptr_from_gwf(gwf_name,name.split('/')[1],"NODELIST",gwf)-1)
			chd_list.append(chd)
			is_chd = True

		nnodes = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NODES", gwf)[0]
		ib = np.array(PerfMeas.get_value_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf)).reshape(-1)
		nlay = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "NLAY", gwf)[0]
		ihc = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IHC", gwf)
		#IHC tells us whether connection is vertical (and if so, whether connection is above or below) or horizontal (and if so, whether it is a vertically staggered grid). 
		#It is of size NJA (or number of connections)
		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		#IA is the number of connections, plus 1 (for self), for each node in grid. it is of size NNODES + 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
		#JA is an array containing all cells for which there is a connection (including self) for each node. it is of size NJA
		jas = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JAS", gwf) - 1
		cl1 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL1", gwf)
		#distance from node to cell m boundary (size NJA)
		cl2 = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "CL2", gwf)
		#distance from cell m to node boundary (size NJA)
		hwva = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "HWVA", gwf)
		#Width perpendicular to flow for a horizontal connection, or the face area for a vertical connection. size NJA
		top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
		#top elevation for all nodes
		bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
		#bottom elevation for all nodes
		iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
		#array of number of connections per node (size ndoes)
		anglex = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "ANGLEX", gwf)

		height = sat * (top - bot)	

		d_mat_k11 = np.zeros(ja.shape[0])
		d_mat_k22 = np.zeros(ja.shape[0])
		d_mat_k33 = np.zeros(ja.shape[0])
		d_mat_k123 = np.zeros(ja.shape[0])

		k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
		k22 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K22", gwf)
		assert np.all(k11==k22)
		k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)

		for node,(offset,ncon) in enumerate(zip(ia,iac)):
			
			#ncon -= 1 # for self
			
			if ib[node]==0:	
				pass
			if is_chd and node in chd_list[0]:
				pass
			else:
				sum1 = 0.
				sum2 = 0.
				sum3 = 0.
				height1 = height[node]
				pp = 1
				for ii in range(offset+1,offset+ncon):
					mnode = ja[ii]
					height2 = height[mnode]
					jj = jas[ii]
					cond_nm = amat[jj]
					iihc = ihc[jj]
					
					if iihc == 0: # vertical con
						#v1 = PerfMeas._dconddvk(k33[node],height1,sat[node],k33[mnode],
			      		#						height2,sat[mnode],hwva[jj],amat[jj])
						v2 = PerfMeas.derivative_conductance_k1(k33[node],k33[mnode],height1, height2, cl1[jj]+cl2[jj], hwva[jj])
						d_mat_k33[ia[node]+pp] += v2
						d_mat_k123[ia[node]+pp] += v2
						sum1 += v1
						pp+=1
						
					else:
						v1 = PerfMeas._dconddhk(k11[node],k11[mnode],cl1[jj],cl2[jj],hwva[jj],height1,height2)
						v2 = PerfMeas.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj]+cl2[jj], cl1[jj]+cl2[jj], hwva[jj],height1)
						d_mat_k11[ia[node]+pp] += v1
						d_mat_k123[ia[node]+pp]  += v1
						sum2 += v1
						pp+=1
					
				d_mat_k11[ia[node]] = -sum2
				d_mat_k33[ia[node]] = -sum1
				d_mat_k22[ia[node]] = -sum3
				d_mat_k123[ia[node]] = d_mat_k11[ia[node]] + d_mat_k22[ia[node]] + d_mat_k33[ia[node]]

		return d_mat_k11,d_mat_k22,d_mat_k33, d_mat_k123

	@staticmethod
	def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
		# todo: upstream weighting - could use height1 and height2 to check...
		# todo: vertically staggered
		d = (width * cl1 * height1 * (height2 ** 2) * (k2**2)) / (((cl2 * height1 * k1) + ((cl1 * height2 * k2))) ** 2)
		return d

	@staticmethod
	def _dconddvk(k1,height1,sat1,k2,height2,sat2,area,vcond_12):
		"""need to think about how k1 and k2 are combined to 
		form the average k between the two cells
		from MH:
		dcond_n,m / dk_n,m = cond_n,m**2 / ((area * k_n,m**2)/(0.5*(top_n - bot_n)))

		"""

		# todo: VARIABLE CV and DEWATER options
		#condsq = (1./((1./((area*k1)/(0.5*(top1-bot1)))) + (1./((area*k2)/(0.5*(top2-bot2))))))**2
		#return condsq / ((area * k1**2)/(0.5*(top1-bot1)))
		d = (vcond_12**2) / ((area * k1**2)/(0.5*(height1)))
		return d

	@staticmethod
	def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
		d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
		return d

	def lam_dAdss_h(self,gwf_name,gwf,lamb,head,dt):
		top = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
		bot = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
		area = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
		result = -1. * lamb * head * area * (top - bot) / dt
		return result
	
	def sens_ss_indirect(self,gwf_name,gwf,lamb,head,head_old,dt):
		return self.lam_dAdss_h(gwf_name,gwf,lamb,head,dt) - self.lam_dAdss_h(gwf_name,gwf,lamb,head_old,dt)

	def lam_dAdk_h(self, gwf_name, gwf, lamb, dAdk, head):
		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		#IA is the number of connections, plus 1 (for self), for each node in grid. it is of size NNODES + 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
		#JA is an array containing all cells for which there is a connection (including self) for each node. it is of size NJA
		iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
		ib = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf).reshape(-1)

		is_chd = False
		chd_list = []
		names = list(gwf.get_input_var_names())
		chds = [name for name in names if 'CHD' in name and 'NODELIST' in name]
		for name in chds:
			chd = np.array(PerfMeas.get_ptr_from_gwf(gwf_name,name.split('/')[1],"NODELIST",gwf)-1)
			chd_list.append(chd)
			is_chd = True

		
		result = np.zeros_like(lamb)
		for i in range(len(lamb)):
			sum1 = 0.0
			sum2 = 0.0
			for ii in range(iac[i]):
			    sum1 += dAdk[ia[i] + ii] * head[ja[ia[i] + ii]]        
			sum1 *= lamb[i]
			for ii in list(range(iac[i]))[1:]:
				if is_chd and ja[ii] in chd_list[0]:
					pass
				#elif ib[ja[ia[i]+ii]] == 0:
				#		pass
				else:
					sum2 += lamb[ja[ia[i] + ii]] * dAdk[ia[i] + ii] * (head[i] - head[ja[ia[i] + ii]])
			sums = sum1 + sum2
			#print(list(range(iac[i]))[1:])
			#print(sum1,sum2,sums)
			result[i] = sums

		return result

	
	


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
					#dfdh[pfr.nnode] = -1. * pfr.weight / deltat_dict[kk]
					dfdh[pfr.nnode] = pfr.weight
				elif self._type == "residual":
					dfdh[pfr.nnode] = 2.0 * pfr.weight * (head_dict[kk][pfr.nnode] - pfr.obsval)
		return dfdh

	@staticmethod
	def get_value_from_gwf( gwf_name, pak_name,prop_name, gwf):
		addr = gwf.get_var_address(prop_name, gwf_name,pak_name)
		return gwf.get_value(addr)

	@staticmethod
	def get_ptr_from_gwf(gwf_name, pak_name, prop_name, gwf):
		addr = gwf.get_var_address(prop_name, gwf_name, pak_name)
		return gwf.get_value_ptr(addr)

	# @staticmethod
	# def get_input_var_names():
	# 	return gwf.get_input_var_names()


	
