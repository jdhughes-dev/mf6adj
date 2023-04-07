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

	def __init__(self,name,type,entries,is_structured):
		self._name = name.lower().strip()
		self._type = type.lower().strip()
		self._entries = entries
		self.is_structured = is_structured


	def solve_adjoint(self, kperkstp, iss, deltat_dict, amat_dict, head_dict, head_old_dict, sat_dict, gwf, gwf_name,mg_structured):
		nnodes = PerfMeas.get_value_from_gwf(gwf_name,"DIS","NODES",gwf)[0]

		lamb = np.zeros(nnodes)
		#lambs = np.zeros((len(kperkstp),nnodes))

		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1

		dfdk11 = np.zeros(nnodes)
		dfdk22 = np.zeros(nnodes)
		dfdk33 = np.zeros(nnodes)
		dfdk123 = np.zeros(nnodes)

		for itime,kk in enumerate(kperkstp[::-1]):
			print('solving',self._name,kk)
			dfdh = self._dfdh(kk, gwf_name, gwf, deltat_dict, head_dict)
			dadk11,dadk22,dadk33,dadk123 = self._dadk(gwf_name, gwf, sat_dict[kk])

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

			# dfdk11 += np.dot(lamb,dadk11.dot(head_dict[kk]))
			# dfdk33 += np.dot(lamb, dadk33.dot(head_dict[kk]))
			dfdk11 = self.lam_dAdk_h(gwf_name,gwf,lamb, dadk11,head_dict[kk])
			dfdk22 = self.lam_dAdk_h(gwf_name,gwf,lamb, dadk22,head_dict[kk])
			dfdk33 = self.lam_dAdk_h(gwf_name,gwf,lamb, dadk33,head_dict[kk])
			dfdk123 = self.lam_dAdk_h(gwf_name,gwf,lamb, dadk123,head_dict[kk])

		# np.savetxt('result.dat',dfdk123)
		self.save_array("k11",dfdk11,gwf_name,gwf,mg_structured)
		self.save_array("k22",dfdk22,gwf_name,gwf,mg_structured)
		self.save_array("k33", dfdk33, gwf_name, gwf, mg_structured)
		self.save_array("k123",dfdk123,gwf_name,gwf,mg_structured)

	def save_array(self,filetag,avec,gwf_name,gwf,structured_mg):
		nodeuser = PerfMeas.get_ptr_from_gwf(gwf_name,"DIS","NODEUSER",gwf)
		nnodes = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "NODES", gwf)
		
		# if not a reduced node scheme
		if len(nodeuser) == 1:
			nodeuser = np.arange(nnodes)
		if structured_mg is not None:
			kijs = structured_mg.get_lrc(list(nodeuser))

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

		is_chd = False
		names = list(gwf.get_input_var_names())
		if '{0}/CHD_0/NODELIST'.format(gwf_name.upper()) in names:
			chd1 =  PerfMeas.get_ptr_from_gwf(gwf_name, "CHD_0", "NODELIST", gwf) - 1
			chd = np.array(chd1)
			is_chd = True
		if '{0}/CHD_1/NODELIST'.format(gwf_name.upper()) in names:
			chd2 = 	PerfMeas.get_ptr_from_gwf(gwf_name, "CHD_1", "NODELIST", gwf) - 1
			chd = np.append(chd1,chd2)
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

		if self.is_structured:
			cellx = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "CELLX", gwf)
			celly = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "CELLY", gwf)
			# delr = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "DELR", gwf)
			# delc = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "DELC", gwf)
			xs = np.repeat(np.tile(cellx, len(celly)),nlay) #NOTE THIS WILL BREAK IF THE CELLX/Y CHANGES LAYER TO LAYER!!!
			ys = np.repeat(np.repeat(celly,len(cellx)), nlay)	#NOTE THIS WILL BREAK IF THE CELLX/Y CHANGES LAYER TO LAYER!!!
		else:
			cellxy = PerfMeas.get_value_from_gwf(gwf_name, "DIS", "CELLXY", gwf)
			cellx = []
			celly = []
			for i in range(len(cellxy)):
				cellx.append(cellxy[i][0])
				celly.append(cellxy[i][1])
			xs = np.repeat(np.tile(cellx, len(celly)),nlay) #NOTE THIS WILL BREAK IF THE CELLX/Y CHANGES LAYER TO LAYER!!!
			ys = np.repeat(np.repeat(celly,len(cellx)), nlay)		#NOTE THIS WILL BREAK IF THE CELLX/Y CHANGES LAYER TO LAYER!!!

		d_mat_k11 = np.zeros(ja.shape[0])
		d_mat_k22 = np.zeros(ja.shape[0])
		d_mat_k33 = np.zeros(ja.shape[0])
		d_mat_k123 = np.zeros(ja.shape[0])

		k11 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K11", gwf)
		k22 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K22", gwf)
		k33 = PerfMeas.get_ptr_from_gwf(gwf_name, "NPF", "K33", gwf)


		for node in range(nnodes):
			xnode = xs[node]
			ynode = ys[node]
			start_ia = ia[node]+1
			end_ia = ia[node+1]
			if ib[node]==0:	
				for ij in range(iac[node]-1):
					d_mat_k33[ia[node] + ij] = 0.
					d_mat_k11[ia[node] + ij] = 0.
					d_mat_k22[ia[node] + ij] = 0.
					d_mat_k123[ia[node] + ij] = 0.
			if is_chd:
				if node in chd:
					for ij in range(iac[node]-1):
						d_mat_k33[ia[node] + ij] = 0.
						d_mat_k11[ia[node] + ij] = 0.
						d_mat_k22[ia[node] + ij] = 0.
						d_mat_k123[ia[node] + ij] = 0.
				else:
					sum1 = 0.
					sum2 = 0.
					sum3 = 0.
					height1 = sat[node] * (top[node] - bot[node])
					# for ii in range(iac[nn])[1:]:
					pp = 1
					for ii in range(start_ia, end_ia):
						mnode = ja[ii]
						xmnode = xs[mnode]
						ymnode = ys[mnode]
						xdiff = xnode - xmnode
						ydiff = ynode - ymnode

						height2 = sat[mnode] * (top[mnode] - bot[mnode])
						jj = jas[ii]
						iihc = ihc[jj]
						if iihc == 0: # vertical con
							d_mat_k11[ia[node]+pp] = 0.
							d_mat_k22[ia[node]+pp] = 0.
							# d_mat_k33[jj] = PerfMeas._dconddvk(k33[node],top[node],bot[node],sat[node],
							# 								   k33[mnode],top[mnode],bot[mnode],sat[mnode],hwva[jj])
							d_mat_k33[ia[node]+pp] = PerfMeas.derivative_conductance_k1(k33[node],k33[mnode],height1, height2, cl1[jj]+cl1[jj], hwva[jj])
							d_mat_k123[ia[node]+pp]  = d_mat_k11[ia[node]+pp] + d_mat_k22[ia[node]+pp] + d_mat_k33[ia[node]+pp]
							sum1 += d_mat_k33[ia[node]+pp]
							sum2 += d_mat_k11[ia[node]+pp]
							sum3 += d_mat_k22[ia[node]+pp]
							pp+=1
						elif np.abs(xdiff) > np.abs(ydiff):
							# d_mat_k11[jj] = PerfMeas._dconddhk(k11[node],k11[mnode],cl1[jj],cl2[jj],hwva[jj],height1,height2)
							d_mat_k11[ia[node]+pp] = PerfMeas.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj]+cl1[jj], cl2[jj]+cl2[jj], hwva[jj],height1)
							d_mat_k33[ia[node]+pp] = 0.
							d_mat_k22[ia[node]+pp] = 0.
							d_mat_k123[ia[node]+pp]  = d_mat_k11[ia[node]+pp] + d_mat_k22[ia[node]+pp] + d_mat_k33[ia[node]+pp]
							sum1 += d_mat_k33[ia[node]+pp]
							sum2 += d_mat_k11[ia[node]+pp]
							sum3 += d_mat_k22[ia[node]+pp]
							pp+=1
						else: #this is K22 in Mohamed's code
							# d_mat_k11[jj] = PerfMeas._dconddhk(k11[node],k11[mnode],cl1[jj],cl2[jj],hwva[jj],height1,height2)
							d_mat_k11[ia[node]+pp] = 0.
							d_mat_k33[ia[node]+pp] = 0.
							d_mat_k22[ia[node]+pp] = PerfMeas.derivative_conductance_k1(k22[node],k22[mnode],hwva[jj],hwva[jj], cl1[jj]+cl1[jj],height1)
							d_mat_k123[ia[node]+pp]  = d_mat_k11[ia[node]+pp] + d_mat_k22[ia[node]+pp] + d_mat_k33[ia[node]+pp]
							sum1 += d_mat_k33[ia[node]+pp]
							sum2 += d_mat_k11[ia[node]+pp]
							sum3 += d_mat_k22[ia[node]+pp]
							pp+=1
						d_mat_k11[ia[node]] = -sum2
						d_mat_k33[ia[node]] = -sum1
						d_mat_k22[ia[node]] = -sum3
						d_mat_k123[ia[node]] = d_mat_k11[ia[node]] + d_mat_k22[ia[node]] + d_mat_k33[ia[node]]
			else:
				sum1 = 0.
				sum2 = 0.
				sum3 = 0.
				height1 = sat[node] * (top[node] - bot[node])
				# for ii in range(iac[nn])[1:]:
				pp = 1
				for ii in range(start_ia, end_ia):
					mnode = ja[ii]
					xmnode = xs[mnode]
					ymnode = ys[mnode]
					xdiff = xnode - xmnode
					ydiff = ynode - ymnode

					height2 = sat[mnode] * (top[mnode] - bot[mnode])
					jj = jas[ii]
					iihc = ihc[jj]
					if iihc == 0: # vertical con
						d_mat_k11[ia[node]+pp] = 0.
						d_mat_k22[ia[node]+pp] = 0.
						# d_mat_k33[jj] = PerfMeas._dconddvk(k33[node],top[node],bot[node],sat[node],
						# 								   k33[mnode],top[mnode],bot[mnode],sat[mnode],hwva[jj])
						d_mat_k33[ia[node]+pp] = PerfMeas.derivative_conductance_k1(k33[node],k33[mnode],height1, height2, cl1[jj]+cl2[jj], hwva[jj])
						d_mat_k123[ia[node]+pp]  = d_mat_k11[ia[node]+pp] + d_mat_k22[ia[node]+pp] + d_mat_k33[ia[node]+pp]
						sum1 += d_mat_k33[ia[node]+pp]
						sum2 += d_mat_k11[ia[node]+pp]
						sum3 += d_mat_k22[ia[node]+pp]
						pp+=1
					elif np.abs(xdiff) > np.abs(ydiff):
						# d_mat_k11[jj] = PerfMeas._dconddhk(k11[node],k11[mnode],cl1[jj],cl2[jj],hwva[jj],height1,height2)
						d_mat_k11[ia[node]+pp] = PerfMeas.derivative_conductance_k1(k11[node],k11[mnode],cl1[jj]+cl2[jj], cl1[jj]+cl2[jj], hwva[jj],height1)
						d_mat_k33[ia[node]+pp] = 0.
						d_mat_k22[ia[node]+pp] = 0.
						d_mat_k123[ia[node]+pp]  = d_mat_k11[ia[node]+pp] + d_mat_k22[ia[node]+pp] + d_mat_k33[ia[node]+pp]
						sum1 += d_mat_k33[ia[node]+pp]
						sum2 += d_mat_k11[ia[node]+pp]
						sum3 += d_mat_k22[ia[node]+pp]
						pp+=1
					else: #this is K22 in Mohamed's code
						# d_mat_k11[jj] = PerfMeas._dconddhk(k11[node],k11[mnode],cl1[jj],cl2[jj],hwva[jj],height1,height2)
						d_mat_k11[ia[node]+pp] = 0.
						d_mat_k33[ia[node]+pp] = 0.
						d_mat_k22[ia[node]+pp] = PerfMeas.derivative_conductance_k1(k22[node],k22[mnode],hwva[jj],hwva[jj], cl1[jj]+cl2[jj],height1)
						d_mat_k123[ia[node]+pp]  = d_mat_k11[ia[node]+pp] + d_mat_k22[ia[node]+pp] + d_mat_k33[ia[node]+pp]
						sum1 += d_mat_k33[ia[node]+pp]
						sum2 += d_mat_k11[ia[node]+pp]
						sum3 += d_mat_k22[ia[node]+pp]
						pp+=1
					d_mat_k11[ia[node]] = -sum2
					d_mat_k33[ia[node]] = -sum1
					d_mat_k22[ia[node]] = -sum3
					d_mat_k123[ia[node]] = d_mat_k11[ia[node]] + d_mat_k22[ia[node]] + d_mat_k33[ia[node]]

		# d_mat_k11 = sparse.csr_matrix((d_mat_k11, ja, ia), shape=(len(ia) - 1, len(ia) - 1))
		# d_mat_k33 = sparse.csr_matrix((d_mat_k33, ja, ia), shape=(len(ia) - 1, len(ia) - 1))
		return d_mat_k11,d_mat_k22,d_mat_k33, d_mat_k123

	# @staticmethod
	# def _dconddhk(k1, k2, cl1, cl2, width, height1, height2):
	# 	# todo: upstream weighting - could use height1 and height2 to check...
	# 	# todo: vertically staggered
	# 	return (width * cl1 * height1 * height2 ** 2 * k2) / ((cl2 * height1 * k1) + (cl1 * height2 * k2) ** 2)

	# @staticmethod
	# def _dconddvk(k1,top1,bot1,sat1,k2,top2,bot2,sat2,area):
	# 	# todo: VARIABLE CV and DEWATER options
	# 	condsq = (1./((1./((area*k1)/(0.5*(top1-bot1)))) + (1./((area*k2)/(0.5*(top2-bot2))))))**2
	# 	return condsq / ((area * k1**2)/(0.5*(top1-bot1)))

	def lam_dAdk_h(self, gwf_name, gwf, lamb, dAdk, head_dict):
		ia = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "IA", gwf) - 1
		#IA is the number of connections, plus 1 (for self), for each node in grid. it is of size NNODES + 1
		ja = PerfMeas.get_ptr_from_gwf(gwf_name, "CON", "JA", gwf) - 1
		#JA is an array containing all cells for which there is a connection (including self) for each node. it is of size NJA
		iac = np.array([ia[i + 1] - ia[i] for i in range(len(ia) - 1)])
		ib = PerfMeas.get_ptr_from_gwf(gwf_name, "DIS", "IDOMAIN", gwf).reshape(-1)

		is_chd = False
		names = list(gwf.get_input_var_names())
		if '{0}/CHD_0/NODELIST'.format(gwf_name.upper()) in names:
			chd1 =  PerfMeas.get_ptr_from_gwf(gwf_name, "CHD_0", "NODELIST", gwf) - 1
			chd = np.array(chd1)
			is_chd = True
		if '{0}/CHD_1/NODELIST'.format(gwf_name.upper()) in names:
			chd2 = 	PerfMeas.get_ptr_from_gwf(gwf_name, "CHD_1", "NODELIST", gwf) - 1
			chd = np.append(chd1,chd2)
			is_chd = True


		my_list = []
		for k in range(len(lamb)):
			sum1 = 0.0
			sum2 = 0.0
			for j in range(iac[k]):
			    sum1 += dAdk[ia[k] + j] * head_dict[ja[ia[k] + j]]

			sum1 = lamb[k] * sum1
			for j in list(range(iac[k]))[1:]:
				if is_chd:
					if ja[ia[k] + j] in chd:
						sum2 += 0.0
					elif ib[ja[ia[k] + j]]==0:
						sum2 += 0.0
					else:
						sum2 += lamb[ja[ia[k] + j]] * dAdk[ia[k] + j] * (head_dict[k] - head_dict[ja[ia[k] + j]])
				elif ib[ja[ia[k] + j]]==0:
					sum2 += 0.0
				else:
					sum2 += lamb[ja[ia[k] + j]] * dAdk[ia[k] + j] * (head_dict[k] - head_dict[ja[ia[k] + j]])

			sums = sum1 + sum2
			my_list.append(sums)

		return my_list

	@staticmethod
	def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
		d = - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)
		return d

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

	# @staticmethod
	# def get_input_var_names():
	# 	return gwf.get_input_var_names()



