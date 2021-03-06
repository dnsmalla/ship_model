# -*- coding: utf-8 -*-
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size,test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.test=test
        self.memory = deque(maxlen=2000)
        self.test=test
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.pre_memo= deque(maxlen=24)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

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
                target = reward
                if not done:
                    target = (reward + self.gamma *
                            np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
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

    def learn_act(self,state,reward,next_state,done,global_reward):
        self.agent.memo(state,self.action,reward,next_state,done)
        if done:
            for state, action, reward, next_state, done in self.agent.pre_memo:
                rewards=reward+global_reward
                self.agent.remember(state,self.action,rewards,next_state,done)
            self.agent.pre_memo=deque(maxlen=24)
        self.agent.replay(self.batch_size)


    def save_model(self):
        path="./dqn_model_save/"
        if not os.path.exists(path):
            os.makedirs(path)
        path="./dqn_model_save/"+self.name
        self.agent.save(path+".h5")


    def test_model(self):
        path="./dqn_model_save/"+self.name
        self.agent.load(path+".h5")
