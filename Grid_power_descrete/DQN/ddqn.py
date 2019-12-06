# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size, action_size,test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.test=test
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.pre_memo= deque(maxlen=24)

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def memo(self,state, action, reward, next_state, done):
        self.pre_memo.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and not self.test:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = self.model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    # a = self.model.predict(next_state)[0]
                    t = self.target_model.predict(next_state)[0]
                    target[0][action] = reward + self.gamma * np.amax(t)
                    # target[0][action] = reward + self.gamma * t[np.argmax(a)]
                self.model.fit(state, target, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Policy:
    def __init__(self,state_dim,action_dim,name,test=False):
        self.state_dim=state_dim
        self.name=name
        self.action_dim=action_dim
        self.batch_size=34
        self.agent=DQNAgent(self.state_dim,self.action_dim,test)

    def choose_action(self,state):
        self.action = self.agent.act(state)
        return self.action

    def learn_act(self,state,reward,next_state,done,g_reward):
        self.agent.memo(state,self.action,reward,next_state,done)
        if done:
            for state, action, reward, next_state, done in self.agent.pre_memo:
                rewards=reward+g_reward
                self.agent.remember(state,self.action,rewards,next_state,done)
            self.agent.pre_memo=deque(maxlen=24)
        self.agent.replay(self.batch_size)

        
    def save_model(self):
        path="./ddqn_model_save/"+self.name
        if not os.path.exists(path):
            os.makedirs(path)
        self.agent.save(path+".h5")


    def test_model(self,state):
        path="./ddqn_model_save/"+self.name
        self.agent.load(path+".h5")