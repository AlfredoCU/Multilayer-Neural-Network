#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:28:52 2020

@author: alfredocu
"""

import numpy as np
import matplotlib.pyplot as plt
from MLP import *

def MLP_binary_classification_2d(X, Y, net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0, i] == 0:
            plt.plot(X[0, i], X[1, i], ".r")
        else:
           plt.plot(X[0, i], X[1, i], ".b")

    xmin, ymin = np.min(X[0, :]) - 0.5, np.min(X[1, :]) - 0.5
    xmax, ymax = np.max(X[0, :]) + 0.5, np.max(X[1, :]) + 0.5

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    data = [xx.ravel(), yy.ravel()]

    zz = net.predict(data)
    zz = zz.reshape(xx.shape)

    plt.contourf(xx, yy, zz, alpha = 0.8, cmap = plt.cm.RdBu)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    # plt.savefig("XOR-T.eps", format="eps")


# X datos.  
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# Y Valores deseados.
# Y = np.array([[0, 0, 0, 1]]) # COMPUERTA AND.
# Y = np.array([[0, 1, 1, 1]]) # COMPUERTA OR.
Y = np.array([[0, 1, 1, 0]]) # COMPUERTA XOR.

net = MLP((2, 100, 1)) # 2, 20, 1
print(net.predict(X))
MLP_binary_classification_2d(X, Y, net)

net.train(X, Y)
print(net.predict(X))
MLP_binary_classification_2d(X, Y, net)


'''

[[0.66119006 0.98658776 0.35930478 0.88535606]]
[[0.00165625 0.05349692 0.0524121  0.93738117]]

[[0.18792276 0.24010548 0.06637753 0.11290209]]
[[0.03443598 0.97651714 0.97705118 0.99990602]]

[[0.0775528  0.19352464 0.01831429 0.01307068]]
[[0.05421726 0.97354233 0.97270146 0.05676224]]

'''