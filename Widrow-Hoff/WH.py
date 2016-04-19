#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

## ρ
ROW = 0.2 

## max learning epoch
EPOCH = 1

class WH():
	## c means weight length
	def __init__(self,c,data_length):
		self.weight      = np.random.rand(c,data_length)

	## to calculate g(x) = wx
	def g_x(self,i,p,input_data):
		gxp = (self.weight[i]*input_data[p]).sum()

		return gxp

	## sum_of_squares 
	def loss(self,p,input_data,superviser):
		
		loss = (((self.g_x(i,p,input_data) - superviser[p])**2).sum())/2

		return loss

	## differentiate loss
	## ∂J/∂w
	def differential_calculus(self,p,input_data,class_data):
		superviser = np.zeros((len(self.weight)),dtype=np.int32)
		differentiate_loss = np.zeros((len(self.weight),len(self.weight[0])),dtype=np.float32)
		superviser[class_data[p]-1] = 1 
		print(input_data[p])
		for i in range(len(self.weight)):
			differentiate_loss[i] = ((self.g_x(i,p,input_data) - superviser[i])*input_data[p])
	
		return differentiate_loss

	def error_correction(self,p,differentiate_loss,input_data):
		## w' = w - ρεx
		for i in range(len(self.weight)):
			self.weight[i] = self.weight[i] - (ROW*differentiate_loss[i])


	# to start learning
	def learning_loop(self,data,class_data):
		
		for loop in range(EPOCH):
			print("epoch : {}".format(loop))
			loss_all = 0
			## data loop
			for p in range(len(data)):
				## g(x) loop
				differentiate_loss = self.differential_calculus(p,data,class_data)
				self.error_correction(p,differentiate_loss,data)
		
		for d in range(len(data)): 	
			print("data: {} class : {} ".format(d+1,class_data[d]))
			print(self.g_x(0,d,data))
			print(self.g_x(1,d,data))
			print(self.g_x(2,d,data))	
		print(self.weight)


class file_operator():
	def __init__(self,filename):
		data        = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(0))
		self.__data = np.c_[np.ones(len(data),dtype=np.float32),data]
		self.__class_data  = np.genfromtxt(filename,delimiter=",",dtype=np.int32,usecols=(1))


	def getData(self):
		return self.__data, self.__class_data

if __name__ == "__main__":
	fo = file_operator("class.data")
	data,class_data = fo.getData()
	print len(data[0])
	wh = WH(3,len(data[0]))
	wh.learning_loop(data,class_data)
