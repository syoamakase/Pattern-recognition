#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt

# to define ρ in w' = w + ρ*x
ROW = 0.03
# How many times the learning do?
EPOCH = 15

CLASS_1_MISS = 1
CLASS_2_MISS = 2
CORRECT      = 0

#Discriminant function rule
class DF():
    #define k-NN(default is 1-NN)
    def __init__(self,data,class_data):
        self.data          = data
        self.class_data    = class_data
        #initialize weight
        self.weight        = np.array((1,1),dtype=np.float32)

    #calclate distance
    ## g(x) = g_1(x) - g_2(x)
    def g_x(self,input_data):
        ##g(x) = wx + w0
        ##if g(x) > 0 then x is class 1
        ##if g(x) < 0 then x is class 2
        judge =  (self.weight*input_data).sum()
        
        return judge
    
    def error_correction(self,error_id,input_data):
        global ROW
        if error_id == CLASS_1_MISS:
            self.weight += ROW*input_data
        elif error_id == CLASS_2_MISS:
            self.weight -= ROW*input_data

    def error_judgement(self,judge,class_data):
        if judge <= 0 and class_data == 1:
            return CLASS_1_MISS
        elif judge >= 0 and class_data == 2:
            return CLASS_2_MISS
        else:
            return CORRECT

    def learning_loop(self,data,class_data):
        prev_weight = copy.deepcopy(self.weight)
        graph = graph_generator(1,self.weight)
        for loop in range(EPOCH):
            for i in range(len(data)):
                judge =  self.g_x(data[i])
                error_id = self.error_judgement(judge,class_data[i])
                self.error_correction(error_id,data[i])
                if np.all(self.weight == prev_weight) == False:
                    #print("judge : {} class : {}".format(judge,class_data[i]))
                    print("weight: {}".format(self.weight))
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
    print("dataload")
    fope = file_operator("2_class.data")
    data,class_data = fope.getData()
    print("learning start")
    df = DF(data,class_data)
    df.learning_loop(data,class_data)
