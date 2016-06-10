#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


# nearest neighbor rule(NN rule)
class NN():

    def __init__(self, kNN, data, class_data):
        self.__kNN = kNN
        self.__data = data
        self.__class_data = class_data

    # calclate distance
    def calc_D(self, raw_data, input_data):
        minus = raw_data - input_data
        distance = np.linalg.norm(minus)

        return distance

    # ranking data
    # Note: define minimum distance as top
    def ranking(self, input_data, class_name):
        print("class: {}".format(class_name))
        distance = np.zeros((len(self.__data), ), dtype=np.float32)
        for i in range(len(self.__data)):
            distance[i] = self.calc_D(self.__data[i], input_data)

        self.voting(distance)

    # to use k-NN (k>1)
    def voting(self, distance):
        for rank in range(0, self.__kNN):
            arg_num = np.argmin(distance)
            print("rank: {} arg: {} class: {} distance: {}".format(
                rank + 1, arg_num, self.__class_data[arg_num], distance[arg_num]))
            distance[arg_num] = float("inf")


# to open file and get data
class file_operator():

    def __init__(self, filename):
        self.__data = np.loadtxt(filename, delimiter=",", usecols=(0, 1, 2, 3))
        self.__class_data = np.genfromtxt(filename, delimiter=",", dtype=None, usecols=(4))

    def getData(self):
        return self.__data, self.__class_data


if __name__ == "__main__":
    print("dataload")
    fope = file_operator("iris.data")
    data, class_data = fope.getData()
    nN = NN(51, data, class_data)
    nN.ranking(data[0], class_data[0])
