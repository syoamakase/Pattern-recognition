#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

# ρ
ROW = 0.03

class WH():
	def __init__(self,data,c):
		self.weight = np.ones((c,len(data)),dtype=np.float32)
		self.__data   = data		


	def g_x(self,data,p):
		gxp = (self.weight[p]*data[p]).sum()

		return gxp
		
