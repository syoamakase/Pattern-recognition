#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

## to define network architecture
# 
Networks = {
    'num_of_layer'  : 3,
    'input_layer'   : 2,
    'hidden_layer1' : 4,
    'output_layer'  : 3
}

## ρ
ROW = 0.2

## max learning epoch
EPOCH = 25

## load file name
FILENAME="class.data"

class MyError(Exception):
    def __init__(self,value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class BP():
    def __init__(self,net):
        weight = []
        self.num_of_layer = net['num_of_layer']
        try :
            if self.num_of_layer != len(net)-1:
                raise MyError('invalid layer structure')
        except MyError as e:
            print(type(e))
            print(e)

        for h in range(self.num_of_layer-1):
            print h==self.num_of_layer-2
            if h == 0:
                w = np.random.rand(net['hidden_layer{}'.format(h+1)],net['input_layer'])
                weight.append(w)

            elif (h==self.num_of_layer-2):
                w = np.random.rand(net['output_layer'],net['hidden_layer{}'.format(h)])
                weight.append(w)

            else:
                w = np.random.rand(net['hidden_layer{}'.format(h+1)],net['hidden_layer{}'.format(h)])
                weight.append(w)

        self.weight = np.array(weight)

    # output unit j for input i
    def h_j_p(self,l,input_data):
        hjp = np.zeros((len(selg.weight[l]),),dtype=np.float32)
        for j in range(len(self.weight[l])):
            for i in range(len(self.weight[l-1])):
                hjp[j] = (self.weight[l-1][j]*input_data).sum()
        
        return hjp

    # actiocating function - sigmoid
    def sigmoid(self,gjp):
        return 1/(1+np.exp(-1*gjp))

    # activationg function - tanhx
    def tanh(self,hjp):
        return (np.exp(gjp)-np.exp(-1*gjp))/(np.exp(gjp)+np.exp(-1*gjp))

    # add nonlinear function to hjp 
    def g_j_p(self,l,input_data):
        # Now it uses sigmoid

        return self.sigmoid(self.h_j_p(l,input_data))
        #return self.tanh(self.h_j_p(j,p,input_data))

    # Jp
    def squared_error(self,p,input_data,class_data):
        pass

    def weight_update(self,l,p,judge,class_data):
        pass

    def error_judgement(self,data):
        return data.argmax()

    def learning_loop(self,input_data,class_data):
        for epoch in range(EPOCH):
            for p in range(len(input_data)):
                data = input_data[p]
                for l in range(self.num_of_layer):
                    data = self.g_j_p(l,data)
                judge = error_judgement(data)


class file_operator():
    def __init__(self,filename):
        data      = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(0))
        self.data = np.c_[np.ones(len(data),dtype=np.float32),data]
        self.class_data = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(1))


if __name__ == "__main__":
    fo = file_operator(FILENAME)
    data = fo.data
    class_data = fo.class_data
    bp = BP(Networks)
    bp.learning_loop(data,class_data)
