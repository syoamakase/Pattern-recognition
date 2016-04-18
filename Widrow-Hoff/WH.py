#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# ρ
ROW = 0.001 

# max learning epoch
EPOCH = 100 

class WH():
	# c means weight length
	def __init__(self,c,data,class_data):
		self.weight      = np.ones((c,1),dtype=np.float32)
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

	# to start learning
	def learning_loop(self,data,class_data):
		
		for loop in range(EPOCH):
			print("epoch : {}".format(loop))
			loss_all = 0
			for p in range(len(data)):
				max_arg = 0
				tmp = float("inf")
				for i in range(len(self.weight)):
					loss_all += self.loss(i,p,data,class_data)
					differentiate_loss = self.differential_calculus(i,p,data,class_data)
				
					if tmp > self.g_x(i,p,data):
						arg_max = i+1
						tmp = self.g_x(i,p,data)
				if arg_max != class_data[p]:
					self.error_correction(class_data[p]-1,p,differentiate_loss,data)
					#print(self.g_x(i,p,data))
			#print(self.weight)
		for p in range(len(data)):
			print("input : {}".format(data[p]))
			for i in range(len(self.weight)):
				print("g_{}_{} : {}".format(i,p,self.g_x(i,p,data))) 			


class file_operator():
	def __init__(self,filename):
		self.__data        = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(0))
		self.__class_data  = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(1))

	def getData(self):
		return self.__data, self.__class_data

if __name__ == "__main__":
	fo = file_operator("class.data")
	data,class_data = fo.getData()
	wh = WH(3,data,class_data)
	wh.learning_loop(data,class_data)
