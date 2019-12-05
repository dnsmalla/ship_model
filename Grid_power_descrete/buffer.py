import os
import shutil
import numpy as np
import tensorflow as tf
from collections import deque

class Memory:
    def __init__(self, capacity, dims):
        self.cap=capacity
        self.capacity = np.zeros((capacity, dims))
        self.memo=deque(maxlen=96)
        self.memory_counter  = 0
        self.t_memory=0

    def pre_store(self,s,r,a,s_,mask):
        self.memo.append((s,r,a,s_,mask))

    def store(self, s,r,a,s_,mask):
        if self.memory_counter==self.cap:
            self.memory_counter = 0
        transition = np.hstack((s,r,a,s_,mask))
        index = self.memory_counter % self.cap
        self.capacity[index, :] = transition
        self.memory_counter += 1
        if self.t_memory<=self.cap-1:
            self.t_memory+=1
        else:
            self.t_memory=self.cap-1

    def sample(self, n):
        if len(self.capacity) > n:
            indices = np.random.choice(self.t_memory, size=n)
        return self.capacity[indices, :]