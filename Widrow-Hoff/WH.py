#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

# ρ
ROW = 0.03

EPOCH = 15

class WH():
	def __init__(self,data,class_data):
		self.weight      = np.ones((len(data),len(data[0])),dtype=np.float32)
		self.data        = data
		self.class_data  = class_data

	# to calculate g(x) = wx
	def g_x(self,p):
		gxp = self.weight[p]*self.data[p]

		return gxp

	def steepest_descent(self,p):
		pass
	
	# sum_of_squares 
	def loss(self,input_data,superviser):
		loss = ((input_data - superviser).sum())/2

		return loss

	# differentiate loss
	# ∂J/∂w
	def differential_calculus(self,input_data,superviser):
		differentiate_loss = (g_x(input_data) - superviser)*input_data
		
		return differentiate_loss

	def error_correction(self,p,differentiate_loss,input_data):
		# w' = w - ρεx
		self.weight[p] = self.weight[p] - ROW*differentiate_loss

	def learning_loop(self,data,class_data):
		
		for loop in range(EPOCH):
			for i in range(len(data)):
				


class file_operator():
	def __init__(self,filename):
		self.__data        = np.loadtxt(filename,delimiter=",",usecol=(0,1,2))
		self.__class_data  = np.genfromtxt(filename,delimeter=",",dtype=np.float32,usecol=(3))

	def getData(self):
		return self.__data, self.__class_data

if __name__ == "__main__":
	fo = file_operator("class.data")
	data,class_data = fo.getData()
	wh = WH(data,len(data),class_data)
