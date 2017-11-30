""" This is an example driver script to test your implementation """

import numpy as np
import matplotlib.pyplot as plt

#
from layers import *
from test import *
from ann import *
from data import *
#
np.random.seed(499)
#

X_train, Y_train, X_test, Y_test = read_data(filename='../set1.dat')
train_count = int(X_train.shape[0] * 0.8)

X_train, Y_train, X_valid, Y_valid = X_train[ :train_count], Y_train[:train_count],\
                                     X_train[train_count: ], Y_train[train_count: ]

ann = ANN([4], 2)
ann.train_validate(X_train, Y_train, X_valid, Y_valid, maxEpochs=1000, learning_rate=1e-2)

preds = ann.predict(X_test)
print preds[:10]
print Y_test[:10]