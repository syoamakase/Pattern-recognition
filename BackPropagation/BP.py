#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# to define network condtrution ?
NetWorks = [ 0,0,0,0]

class BP():
    def __init__(self,Net):
        pass
        
    def h_j_p(self,j,p,input_data):
        pass

    def sigmoid(self,gjp):
        return 1/(a+np.exp(-1*gjp))

    def tanh(self,gjp):
        return (np.exp(gjp)-np.exp(-1*gjp))/(np.exp(gjp)+np.exp(-1*gjp))

    def g_j_p(self,j,p,input_data):
        

class file_operator():
    def __init__(self,filename):
        pass


if __name__ == "__main__":
    pass
