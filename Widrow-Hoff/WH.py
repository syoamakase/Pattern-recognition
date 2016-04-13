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

	def steepest_descent(self,p):
		pass


class file_operator():
	def __init__(self,filename):
		data = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecol(0)

	def getData(self):
		pass


if __name__ == "__main__":
			
