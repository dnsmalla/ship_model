import os
import shutil
import numpy as np
import tensorflow as tf
from collections import deque
np.random.seed(1)

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

class Policy(object):
    def __init__(self, state_size, action_size,name,test=False):
        tf.reset_default_graph()
        self.n_features = state_size
        self.n_actions = action_size
        self.name=name
        self.test=test
        self.memory_size =5000
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99998
        self.lr = 0.0001
        self.batch=98
        self.output_graph=False
        self.learn_step_counter=0
        self.replace_target_iter=48
        self._build_net()
        self.r_count=0
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.Memory = Memory(self.memory_size, self.n_features * 2 + 9)
        self.saver=tf.train.Saver()

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s1 = tf.placeholder(tf.float32, [None, 7], name='s1')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0, 0.1), tf.constant_initializer(0.01)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 30,  kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            # tf.layers.dropout(e1,rate=0.25,noise_shape=None,seed=None,training=False,name=None)
            self.q_el = tf.layers.dense(e1, 20, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='e2')
            self.q_gat=tf.layers.dense(self.s1, 20, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='e3')
            self.q_gat_1=tf.layers.dense(self.q_gat, 10,tf.nn.tanh, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='e4')
            self.q_evl=tf.layers.dense(tf.concat([self.q_el, self.q_gat_1], axis=1),5,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='e6')
            self.q_eval=tf.layers.dense(self.q_evl,self.n_actions,tf.nn.softmax,
                                        kernel_initializer=w_initializer,bias_initializer=b_initializer, name='e7')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 30,  kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.t_el = tf.layers.dense(t1,20, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')
            self.t_gat=tf.layers.dense(self.s1, 20, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t3')
            self.t_gat_1=tf.layers.dense(self.t_gat, 10,tf.nn.tanh, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t4')
            self.q_ne=tf.layers.dense(tf.concat([self.t_el, self.t_gat_1], axis=1),5,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t6')
            self.q_next=tf.layers.dense(self.q_ne,self.n_actions,tf.nn.softmax,
                                            kernel_initializer=w_initializer,bias_initializer=b_initializer, name='t7')


        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            #self.loss = tf.clip_by_value(self.loss, -10, 10)
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        act_obs=observation[0]
        # if act_obs[0] > act_obs[1]:
        #     act_obs[0] = -1
        # if act_obs[1]<=1000 :
        #     act_obs[1] = -1
        # if act_obs[0] < 0 :
        #     act_obs[2] = -1
        act_obs=[act_obs]
        self.action=None
        if np.random.rand()>= self.epsilon or not self.test :
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation,self.s1:act_obs})
            action = np.argmax(actions_value)
        else:
            action=np.random.randint(2,size=1)[0]
        self.action=action
        return self.action

    def learn_act(self,observation,reward,next_state,done,global_reward):
        act_obs=observation[0]
        # if act_obs[0]>act_obs[1]:
        #     act_obs[0]=-1
        # if act_obs[0]<=1000 :
        #     act_obs[1]=-1
        # if act_obs[0] < 0 :
        #     act_obs[2] = -1
        act_obs=[act_obs]
        self.Memory.pre_store(observation[0],reward,self.action,next_state[0],act_obs[0])
        if done :
            pre_memory=self.Memory.memo
            self.Memory.memo=deque(maxlen=96)
            for state,re,action,next_state,mask in pre_memory:
                rewards=re+global_reward
                self.Memory.store(state,action,rewards,next_state,mask)

        if self.batch<self.Memory.t_memory:
            batch_memory=self.Memory.sample(self.batch)
            _, cost = self.sess.run(
                [self._train_op, self.loss],
                feed_dict={
                    self.s: batch_memory[:, 0:7],
                    self.s1:batch_memory[:, 16:23],
                    self.a: batch_memory[:, 7],
                    self.r: batch_memory[:, 8],
                    self.s_: batch_memory[:, 9:16],
                })
            self.cost_his.append(cost)
            if self.epsilon > self.epsilon_min :
                self.epsilon *= self.epsilon_decay
            self.learn_step_counter+=1
            if self.learn_step_counter> self.replace_target_iter :
                self.sess.run(self.target_replace_op)
                self.learn_step_counter=0

    def save_model(self):
        name = self.name
        print(name)
        path = "./GDqn_model_save/" + name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
        self.saver.save(self.sess, path + "/model.ckpt")

    def test_model(self):
        name = self.name
        path = "./GDqn_model_save/"
        self.saver.restore(self.sess, path + name + "/model.ckpt")
