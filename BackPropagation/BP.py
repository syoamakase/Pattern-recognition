#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# to define network architecture
# 
NetWorks = [3,2,3,3]

class BP():
    def __init__(self,Net):
        pass
    
    # output unit j for input i
    def h_j_p(self,j,p,input_data):
        pass

    # actiocating function - sigmoid
    def sigmoid(self,gjp):
        return 1/(a+np.exp(-1*gjp))

    # activationg function - tanhx
    def tanh(self,gjp):
        return (np.exp(gjp)-np.exp(-1*gjp))/(np.exp(gjp)+np.exp(-1*gjp))

    # add nonlinear function to hjp 
    def g_j_p(self,j,p,input_data):
        pass

    # Jp
    def squared_error(self,p,input_data,class_data):
        pass

    def weight_update(self):
        pass

    def learning_loop(self,input_data,class_data):
        pass

class file_operator():
    def __init__(self,filename):
        pass


if __name__ == "__main__":
    pass
