# -*- coding: utf-8 -*-
import random
from graph_cnn_layer import GraphCNN as GraphConv
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing   import normalize
from sklearn.model_selection import train_test_split

class Policy():
    def __init__(self,input,output,test=False):
        self.test=test
        self.input=input
        self.action_size=output
        graph_mat=np.ones(self.input)
        graph_mat=np.reshape(graph_mat,(self.input,1))
        self.graph_mat = K.constant(graph_mat)
        self.epoch = 1
        self.epochs = np.arange(self.epoch)
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate=0.001
        self.batch_size = 1
        self.num_hidden_1 = 64
        self.num_hidden_2 =64
        self.filters_1  = 7
        self.batch_size=1
        self.action=None
        self.model=self.Model()
        self.pre_memo= deque(maxlen=200)

    def Model(self):
        model = Sequential()
        model.add(GraphConv(14,num_filters=self.filters_1, graph_conv_filters =self.graph_mat,input_shape=(self.input,)))
        model.add(Activation('relu'))
        model.add(GraphConv(56,num_filters=self.filters_1, graph_conv_filters =self.graph_mat))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size,activation='softmax'))
        # model.summary()
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def memo(self,state, action, reward, next_state, done):
        self.pre_memo.append((state, action, reward, next_state, done))

    def learn_act(self,state,reward,next_state,done,global_reward,memory):
        memory.pre_store(state,self.action,reward,next_state,done)
        if done:
            for state, action, reward, next_state, done in memory.pre_store:
                rewards=reward+global_reward
                memory.store(state,self.action,rewards,next_state,done)
            memory.pre_store=deque(maxlen=200)
        if memory.t_memory > self.batch_size:
            minibatch = memory.sample(self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + 0.95 *
                            np.amax(self.model.predict(next_state)))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                target_f=np.reshape(target_f,(1,2))
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def choose_action(self,state):
        if np.random.rand() <= self.epsilon and not self.test:
            self.action=random.randrange(self.action_size)
        else:
            pred_test  = self.model.predict(state, batch_size=1).flatten()
            self.action=np.argmax(pred_test)
        return  self.action

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
# agent=Policy(7,2)
# print(agent)