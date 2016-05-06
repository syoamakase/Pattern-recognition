#!/usr/bin/env python
#-*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

def graph_tanh():
    x=np.arange(-10,10,0.1)
    y = np.tanh(x)
    plt.plot(x,y)
    plt.show()

def graph_sigmoid():
    x=np.arange(-10,10,0.1)
    y = (1/(1+ np.exp(-x)))
    plt.plot(x,y)
    plt.show()

def graph_relu():
    x=np.arange(-10,10,0.1)
    y = np.fmax(np.zeros((20//0.1+1)),x)
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    graph_tanh()
    graph_sigmoid()
    graph_relu()
