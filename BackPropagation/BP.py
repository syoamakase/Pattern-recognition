#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# to define network architecture
# 
NetWorks = {
    'num_of_layer'  : 3,
    'input_layer'   : 2,
    'hidden1_layer' : 4,
    'output_layer'  : 3
}

FILENAME="class.data"

class BP():
    def __init__(self,Net):
        pass

    # output unit j for input i
    def h_j_p(self,j,p,input_data):
        pass

    # actiocating function - sigmoid
    def sigmoid(self,gjp):
        return 1/(1+np.exp(-1*gjp))

    # activationg function - tanhx
    def tanh(self,hjp):
        return (np.exp(gjp)-np.exp(-1*gjp))/(np.exp(gjp)+np.exp(-1*gjp))

    # add nonlinear function to hjp 
    def g_j_p(self,j,p,input_data):
        # Now it uses sigmoid

        #return self.tanh(self.h_j_p(j,p,input_data))
        return self.sigmoid(self.h_j_p(j,p,input_data))

    # Jp
    def squared_error(self,p,input_data,class_data):
        pass

    def weight_update(self):
        pass

    def learning_loop(self,input_data,class_data):
        pass

class file_operator():
    def __init__(self,filename):
        data      = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(0))
        self.data = np.c_[np.ones(len(data),dtype=np.float32),data]
        seld.class_data = np.genfromtxt(filename,delimiter=",",dtype=np.float32,usecols=(1))


if __name__ == "__main__":
    fo = file_operator(FILENAME)
    data = fo.data
    class_data = fo.class_data
    bp = BP(Networks)
    bp.learning_loop(data,class_data)
