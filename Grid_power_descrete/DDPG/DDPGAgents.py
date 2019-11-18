import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import numpy as np
from collections import deque
from DDPG.actor import Actor
from DDPG.critic import Critic

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)][0] 
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        #assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

class Policy:
    def __init__(self,state_dim,action_dim,action_bound=1):
        tf.compat.v1.reset_default_graph()
        self.sess=tf.Session()
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.actor = Actor(self.sess,state_dim, action_dim, action_bound, LR_A, REPLACEMENT)
        self.critic = Critic(self.sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, self.actor.a, self.actor.a_)
        self.actor.add_grad_to_graph(self.critic.a_grads)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
        self.saver=tf.compat.v1.train.Saver()

    def choose_action(self,state):
        # tf.reset_default_graph()
        # state=np.reshape(state,[1,5])
        self.action = self.actor.choose_action(state)
        noise=0
        return self.action+noise


    def learn_act(self,state,reward,next_state):

        self.M.store_transition(state,self.action,reward,next_state)
        #print("self.name", state, self.action, reward, next_state )
        if self.M.pointer> BATCH_SIZE:
            b_M = self.M.sample(BATCH_SIZE)
            b_s = b_M[:, :self.state_dim]
            b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
            b_r = b_M[:, -self.state_dim - 1: -self.state_dim]
            b_s_ = b_M[:, -self.state_dim:]
            a_=self.actor.make_a(b_s_)
            self.critic.learn(b_s, b_a, b_r, b_s_,a_)
            self.actor.learn(b_s)

    def save_model(self):
        path="./model_save/"+self.name
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path+"/model.ckpt")


    def test_model(self,state):
        path="./model_save/"+self.name+"/model.ckpt"
        self.saver.restore(self.sess, path)
        self.action = self.actor.choose_action(state)
        return self.action
