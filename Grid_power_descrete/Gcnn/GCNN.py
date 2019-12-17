# -*- coding: utf-8 -*-
import os
os.environ['KERAS_BACKEND'] = 'theano'
import sys
sys.path.append('./GraphCNN-Origin/code')
from graph_convolution import GraphConv
import theano
theano.config.exception_verbosity='high'
import time
import math
import numpy  as np
import pandas as pd
import statistics

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers   import adam, RMSprop
from keras.regularizers import l2, l1
from keras.initializers import TruncatedNormal

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing   import normalize
from sklearn.model_selection import train_test_split

class Policy():
    def __init__(self):

        self.sigma = 81
        self.num_neighbors = 6
        self.graph_mat = self.correlation(np.random.random((11,6)),self.num_neighbors)
        self.epoch = 500
        self.epochs = np.arange(self.epoch)
        self.results_train = np.ones(self.epoch) * 100
        self.results_test  = np.ones(self.epoch) * 100
        self.batch_size = 1
        self.num_hidden_1 = 121
        self.num_hidden_2 =121
        self.filters_1  = 8
        self.filters_2  = 12
        self.model=self.Model()

    def Model(self):
        model = Sequential()
        model.add(GraphConv(filters=self.filters_1, neighbors_ix_mat =self.graph_mat, num_neighbors=self.num_neighbors, input_shape=(11,1)))
        model.add(Activation('relu'))
        model.add(GraphConv(filters=self.filters_2, neighbors_ix_mat = self.graph_mat, num_neighbors=self.num_neighbors)) 
        # model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Flatten())
        model.add(Dense(2))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train(self,state,next_state):

        for i in epochs:
            pred_test = self.model.predict(state, batch_size=1).flatten()
            pred_n_test  = self.model.predict(next_state, batch_size=1).flatten()

            RMSE = np.sqrt(mean_squared_error(pred_n_test, pred_test))

    def test(self,state):
        pred_test  = self.model.predict(state, batch_size=1).flatten()
        return pred_test

    def model_save(self):
        self.model.save('weights_GCNN.h5')

    def model_load(self):
        self.model.load_weights('weights_GCNN.h5')

    def correlation(self,data,num_neighbors):
        corr_mat  = np.array(normalize(np.abs(np.corrcoef(data.transpose())), norm='l1', axis=1),dtype='float64')
        graph_mat = np.argsort(corr_mat,1)[:,-num_neighbors:]
        return graph_mat

    def gaussiankernel(self,data,num_neighbors, sigma):
        X_trainT = data.T
        row  = data.shape[0]
        kernel_mat = np.zeros(row * row).reshape(row, row)

        for i in range(row):
            for j in range(row):
                kernel_mat[i, j] = math.exp( - (np.linalg.norm(X_trainT[i] - X_trainT[j]) ** 2) / (2 * sigma ** 2))
        kernel_mat = np.array(normalize(kernel_mat, norm='l1', axis=1))
        graph_mat  = np.argsort(kernel_mat, 1)[:,-num_neighbors:]
        return graph_mat

agent=Policy()
use_state=np.random.random((1,11))
use_state=np.reshape(use_state,(1,11,1))
print(use_state)
print(agent.test(use_state))
