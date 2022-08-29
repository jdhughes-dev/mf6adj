import os
import shutil
import numpy as np
import pandas as pd
import modflowapi
import flopy


class PerfMeasLocationRecord(object):
	def __init__(self,kper,kstp,nnode,k=None,i=None,j=None):
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


class PerfMeas(object):
	def __init__(self,name,type,entries):
		self._name = name.lower().strip()
		self._type = type.lower().strip()
		self._entries = entries

