#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


# to minimize each distance
class Normalization():

    def __init__(self):
        pass

    def __call__(self, data=None):
        d = len(data)
        var_all = np.std(data, axis=0)
        var_prod = np.prod(var_all)
        a = (var_prod ** (1.0 / d)) / var_all
        transformation_matrix = np.diag(a)
        result = np.zeros((d, len(data[0])))
        print("\ntransformation matrix:\n{}".format(transformation_matrix))
        for j in range(d):
            tmp = np.dot(transformation_matrix, data[j])
            result[j] = tmp
        return result

if __name__ == "__main__":
    norm = Normalization()
    data = np.array([[1, 2], [2, 4], [2, 6]])
    print("raw data:\n{}".format(data))
    result = norm(data)
    print("\nnormalizarion data:\n{}".format(result))
