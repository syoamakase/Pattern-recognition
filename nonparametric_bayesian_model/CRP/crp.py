#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def crp(alpha=10,N=1000):
    N_gen = np.random.rand(N)
    table = np.array([1], np.int32)
    table_num = np.ones(N)
    for i in range(2,N):
        new_table = alpha / (i+alpha)
        if N_gen[i] < new_table:
            # テーブル追加
            table = np.append(table, 1)
        else:
            norm_i_gen = N_gen[i] * i
            for j in range(len(table)):
                if norm_i_gen < table[j]:
                    table[j] += 1
                    break

                norm_i_gen -= table[j]
        table_num[i] = len(table)

    return table, table_num

def main():
    N = 1000
    table_10, table_num_10 = crp(alpha=10,N=N)
    table_2, table_num_2 = crp(alpha=2,N=N)
    print("Table length(alpha=10): {}".format(len(table_10)))
    print("Table(alpha=10) : {}".format(table_10))
    print("Table length(alpha=2): {}".format(len(table_2)))
    print("Table(alpha=2) : {}".format(table_2))
    plt.plot(table_num_10)
    plt.plot(table_num_2)
    plt.title("The number of tables")
    plt.xlabel("N")
    plt.ylabel("tables")
    plt.show()

if __name__ == "__main__":
    main()
