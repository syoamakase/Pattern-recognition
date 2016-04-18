#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy

# ρ
ROW = 0.001 

EPOCH = 10 

class WH():
	def __init__(self,c,data,class_data):
		self.weight      = np.ones((c,len(data[0])),dtype=np.float32)
		self.data        = data
		self.class_data  = class_data

	# to calculate g(x) = wx
	def g_x(self,i,p,input_data):
		gxp = (self.weight[i]*input_data[p])

		return gxp

	# sum_of_squares 
	def loss(self,i,p,input_data,superviser):
		loss = (((self.g_x(i,p,input_data) - superviser[p])**2).sum())/2

		return loss

	# differentiate loss
	# ∂J/∂w
	def differential_calculus(self,i,p,input_data,superviser):
		differentiate_loss = (self.g_x(i,p,input_data) - superviser[p])*input_data
		
		return differentiate_loss

	def error_correction(self,i,p,differentiate_loss,input_data):
		# w' = w - ρεx
		self.weight[i] = self.weight[i] - (ROW*differentiate_loss[p])

	def learning_loop(self,data,class_data):
		
		for loop in range(EPOCH):
			print("epoch : {}".format(loop))
			loss_all = 0
			for p in range(len(data)):
				for i in range(len(self.weight)):
					loss_all += self.loss(i,p,data,class_data)
					differentiate_loss = self.differential_calculus(i,p,data,class_data)
					self.error_correction(i,p,differentiate_loss,data)
					print(self.g_x(i,p,data))
				#print("loss : {}".format(loss_all))
			#print(self.weight) 			


class file_operator():
	def __init__(self,filename):
		self.__data        = np.loadtxt(filename,delimiter=",",usecols=(0,1,2))
		self.__class_data  = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(3))

	def getData(self):
		return self.__data, self.__class_data

if __name__ == "__main__":
	fo = file_operator("class.data")
	data,class_data = fo.getData()
	wh = WH(2,data,class_data)
	wh.learning_loop(data,class_data)
