#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import lines

row = 0.03
#Discriminant function rule
class DF():
	#define k-NN(default is 1-NN)
	def __init__(self,kNN,data,class_data):
		self.kNN           = kNN
		self.data          = data
		self.class_data    = class_data
		self.weight        = np.ones((2),dtype=np.float32)
		with open('data.json','w') as f:
			pass


	#calclate distance
	## g(x) = g_1(x) - g_2(x)
	def g_x(self,input_data):
		##g(x) = wx + w_0
		##if g(x) > 0 then x is class 1
		##if g(x) < 0 then x is class 2
		judge =  (self.weight*input_data).sum()
		
		return judge
	
	def error_correction(self,error_id,input_data):
		global row
		if error_id == 1:
			self.weight += row*input_data
		elif error_id == 2:
			self.weight -= row*input_data

	def error_judgement(self,judge,class_data):
		if judge <= 0 and class_data == 1:
			return 1
		elif judge >= 0 and class_data ==2:
			return 2
		else:
			return 0

	def learning_loop(self,data,class_data):
		prev_weight = copy.deepcopy(self.weight)
		graph = graph_generator(1,self.weight)
		for loop in range(15):
			for i in range(len(data)):
				judge =  self.g_x(data[i])
				error_id = self.error_judgement(judge,class_data[i])
				self.error_correction(error_id,data[i])
				if np.all(self.weight == prev_weight) == False:
					#print "judge : {} class : {}".format(judge,class_data[i])
					print "weight: {}".format(self.weight)
					prev_weight = copy.deepcopy(self.weight)
					graph.graph_gen(self.weight)

class file_operator():
	def __init__(self,filename):
		data = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(0))
		self.__data = np.c_[np.ones(len(data),dtype=np.float32),data]
		self.__class_data = np.genfromtxt(filename,delimiter=",",dtype=np.int32,usecols=(1))
				
	def getData(self):
		return self.__data , self.__class_data

class graph_generator():
		def __init__(self,sleep,data):
			self.__sleep = sleep
			self.fig, self.ax = plt.subplots(1,1)
			self.zero = np.zeros(2)
			plt.xlim(0,0.5)
			plt.ylim(0,1.2)
			self.lines, = self.ax.plot([0,data[0]],[0,data[1]])
			self.ax.plot([0,1],[0,2])
			self.ax.plot([0,1],[0,5])

		def graph_gen(self,data):
			self.lines.set_data([0,data[0]],[0,data[1]])
			plt.pause(self.__sleep)

if __name__== "__main__":
	print "dataload"
	fope = file_operator("2_class.data")
	data,class_data = fope.getData()
	df = DF(1,data,class_data)
	df.learning_loop(data,class_data)
