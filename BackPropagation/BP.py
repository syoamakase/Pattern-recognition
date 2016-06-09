#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# to define network architecture
Networks = [
    2,
    4,
    3
]

# ρ
ROW = 0.02

# max learning epoch
EPOCH = 300

# load file name
FILENAME = "class.data"


class MyError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class BP():

    def __init__(self, net, nonlinear="sigmoid"):
        weight = []
        self.net = net
        self.num_of_layer = len(net)
        self.nonlinear = nonlinear
        try:
            if self.num_of_layer != len(net) - 1:
                raise MyError('invalid layer structure')
        except MyError as e:
            print(type(e))
            print(e)

        for h in range(self.num_of_layer - 1):
            w = np.random.rand(net[h + 1], net[h])
            weight.append(w)

        self.weight = np.array(weight)

    # output unit j for input i
    def h_j_p(self, l, input_data):
        hjp = np.zeros((len(self.weight[l]),), dtype=np.float32)
        for j in range(len(hjp)):
            hjp[j] = (self.weight[l][j] * input_data).sum()

        return hjp

    # activate function - sigmoid
    def sigmoid(self, gjp):
        return 1 / (1 + np.exp(-1 * gjp))

    # differentiate activate function(sigmoid)
    def sigmoid_dash(self, gjp):
        return gjp * (1 - gjp)

    # activationg function - tanhx
    def tanh(self, gjp):
        return (np.exp(gjp) - np.exp(-1 * gjp)) / (np.exp(gjp) + np.exp(-1 * gjp))

    # differentiate activate function(tanh)
    def tanh_dash(self, gjp):
        return 1 - (gjp ** 2)

    # add nonlinear function to hjp
    def g_j_p(self, l, input_data):
        # Now it uses sigmoid
        if self.nonlinear == "tanh":
            return self.tanh(self.h_j_p(l, input_data))
        else:
            return self.sigmoid(self.h_j_p(l, input_data))

    # Jp
    def squared_error(self, p, data, class_data):
        superviser = np.zeros(self.net[self.num_of_layer - 1], dtype=np.int32)
        superviser_arg = int(class_data[p] - 1)
        superviser[superviser_arg] = 1

        squared_error = (((data - superviser) ** 2).sum()) / 2

        return squared_error

    def print_result(self, input_data, class_data):
        for p in range(len(input_data)):
            data = input_data[p]
            gjp_list = []
            gjp_list.append(self.sigmoid(data))
            for l in range(0, len(self.weight)):
                data = self.g_j_p(l, data)
                gjp_list.append(data)
            judge = data.argmax()
            print("input data : {},\noutput : {}".format(input_data[p], data))
            print("\n \033[31m class : {}, prediction : {}\033[0m".format(
                int(class_data[p]), judge + 1))
            print("------------------------------")

    # recursive function
    def back_propagation(self, l, p, class_data, gjp, epsilon_kp=None):
        # deal with output layer
        if l == self.num_of_layer - 1:
            superviser = np.zeros(self.net[self.num_of_layer - 1], dtype=np.int32)
            superviser_arg = int(class_data[p] - 1)
            superviser[superviser_arg] = 1
            epsilon_jp = (gjp[l] - superviser) * self.sigmoid_dash(gjp[l])
            for j in range(self.net[self.num_of_layer - 1]):
                self.weight[l-1][j] = self.weight[l-1][j] - ROW * epsilon_jp[j] * gjp[l-1]
            
            # recursive call
            self.back_propagation(l-1, p, class_data, gjp, epsilon_jp)

        # finish
        elif l <= 0:
            return

        # deal with hidden layer or input layer
        else:
            dJp_dgjp = (epsilon_kp * self.weight[l].T).sum()
            epsilon_jp = dJp_dgjp * self.sigmoid_dash(gjp[l])

            for j in range(len(gjp[l])):
                self.weight[l-1][j] = self.weight[l-1][j] - ROW * epsilon_jp[j]*gjp[l-1]

            # recursive call
            self.back_propagation(l-1, p, class_data, gjp, epsilon_jp)

    # to start learning
    def learning_loop(self, input_data, class_data):
        for epoch in range(EPOCH):
            print('epoch : {}'.format(epoch))
            loss = 0
            for p in range(len(input_data)):
                data = input_data[p]
                gjp_list = []
                gjp_list.append(self.sigmoid(data))
                for l in range(0, len(self.weight)):
                    data = self.g_j_p(l, data)
                    gjp_list.append(data)
                gjp = np.array(gjp_list)
                judge = data.argmax()
                loss += self.squared_error(p, data, class_data)
                if (judge + 1) != class_data[p]:
                    self.back_propagation(self.num_of_layer-1, p, class_data, gjp)
            print('loss : {}'.format(loss))
        self.print_result(input_data, class_data)


class file_operator():

    def __init__(self, filename):
        data = np.genfromtxt(filename, delimiter=",", dtype=np.float32, usecols=(0))
        self.data = np.c_[np.ones(len(data), dtype=np.float32), data]
        self.class_data = np.genfromtxt(filename, delimiter=",", dtype=np.float32, usecols=(1))


if __name__ == "__main__":
    fo = file_operator(FILENAME)
    data = fo.data
    class_data = fo.class_data
    bp = BP(Networks)
    bp.learning_loop(data, class_data)
