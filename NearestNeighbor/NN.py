#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import csv

class NN():
	def __init__(self,kNN,data,class_data):
		self.kNN        = kNN
		self.data       = data
		self.class_data = class_data
		
	def calc_D(self,raw_data,input_data):
		minus    = raw_data - input_data 
		print minus
		distance = np.linalg.norm(minus)

		return distance

	def ranking(self,input_data,class_name):
		print "class: {}".format(class_name)
		distance = np.zeros((len(data),),dtype=np.float32)
		for i in range(len(self.data)):
			distance[i] = self.calc_D(self.data[i],input_data)
		
		self.voting(distance)	

	def voting(self,distance):
		for rank in xrange(0,self.kNN):
			arg_num = np.argmin(distance)
			print "rank: {} arg: {} class: {} distance: {}".format(rank+1,arg_num,self.class_data[arg_num],distance[arg_num])
			distance[arg_num] = float("inf")

class file_operator():
	def __init__(self,filename):
		self.data = np.loadtxt("iris.data",delimiter=",",usecols=(0,1,2,3))
		self.class_data = np.genfromtxt("iris.data",delimiter=",",dtype=None,usecols=(4))		

	def getData(self):
		return self.data,self.class_data


if __name__== "__main__":
	print "dataload"
	fope       = file_operator("iris.data")
	data,class_data = fope.getData()
	nN = NN(51,data,class_data) 
	nN.ranking(data[0],class_data[0])
