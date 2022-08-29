import os
import shutil
import numpy as np
import pandas as pd
import modflowapi
import flopy


class PerfMeasRecord(object):
	def __init__(self,kper,kstp,nnode,k=None,i=None,j=None,weight=None,obsval=None):
		self._kper = int(kper)
		self._kstp = int(kstp)
		self._nnode = int(nnode)
		self._k = None
		if k is not None:
			self._k = int(k)
		self._i = None
		if i is not None:
			self._i = int(i)
		self._j = None
		if j is not None:
			self._j = int(j)
		self._weight = None
		if weight is not None:
			self._weight = float(weight)
		self._obsval = None
		if obsval is not None:
			self._obsval = float(obsval)
class PerfMeas(object):
	def __init__(self,name,type,entries):
		self._name = name.lower().strip()
		self._type = type.lower().strip()
		self._entries = entries

	def solve_adjoint(self,kperkstp,deltat_dict,amat_dict,head_dict,head_old_dict):
		pass



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

