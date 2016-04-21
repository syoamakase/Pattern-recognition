#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

## ρ
ROW = 0.2 

## max learning epoch
EPOCH = 25

class WH():
    ## c means weight length
    def __init__(self,c,data_length):
        self.weight      = np.random.rand(c,data_length)

    ## to calculate g(x) = wx
    def g_x(self,i,p,input_data):
        gxp = (self.weight[i]*input_data[p]).sum()

        return gxp

    ## sum_of_squares 
    def loss(self,p,input_data,class_data):
        superviser = np.zeros((len(self.weight)),dtype=np.int32)
        loss       = np.zeros((len(self.weight),len(self.weight[0])),dtype=np.float32)
        superviser[class_data[p]-1] = 1
        for i in range(len(self.weight)):
            loss[i] = (((self.g_x(i,p,input_data) - superviser[i])**2))/2

        return loss

    ## differentiate loss = ∂J/∂w
    def differential_calculus(self,p,input_data,class_data):
        superviser = np.zeros((len(self.weight)),dtype=np.int32)
        differentiate_loss = np.zeros((len(self.weight),len(self.weight[0])),dtype=np.float32)
        superviser[class_data[p]-1] = 1 
        for i in range(len(self.weight)):
            differentiate_loss[i] = ((self.g_x(i,p,input_data) - superviser[i])*input_data[p])
    
        return differentiate_loss
    
    def weight_update(self,p,differentiate_loss,judge,input_data,class_data):
        ## w' = w - ρεx
        if judge != (class_data[p]-1):
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] - (ROW*differentiate_loss[i])
    

    def error_judgement(self,p,input_data):
        max_arg = 0
        gx_list = np.zeros((len(self.weight),),dtype=np.float32)
        for i in range(len(self.weight)):
            gx_list[i] = self.g_x(i,p,input_data)

        return gx_list.argmax()
    
    def print_result(self,input_data,class_data):
        print("result")
        for p in range(len(input_data)):
            gx_max = np.zeros((len(self.weight)),dtype=np.float32)
            print("data {} :  class : {}".format(input_data[p],class_data[p]))
            for i in range(len(self.weight)):
                print("g_x_{} : {}".format(i,self.g_x(i,p,input_data)))
                gx_max[i] = self.g_x(i,p,input_data)
            print("class is {}".format(gx_max.argmax()+1))
            print("-------------------------------------")

    # to start learning
    def learning_loop(self,data,class_data):
        
        for loop in range(EPOCH):
            print("epoch : {}".format(loop))
            loss_all = 0
            ## data loop
            for p in range(len(data)):
                ## g(x) loop
                differentiate_loss = self.differential_calculus(p,data,class_data)
                judge = self.error_judgement(p,data)
                self.weight_update(p,differentiate_loss,judge,data,class_data)
        self.print_result(data,class_data)
        

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
    wh = WH(3,len(data[0]))
    wh.learning_loop(data,class_data)
