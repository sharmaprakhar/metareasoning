#import gym
import sys
import math
import random
import itertools
import collections
import numpy as np
from utils import *
class fnApprox:
    """
    Implements methods to approximate an
    action value function using fourier basis function
    TO DO: Create a super class with inheritable functions to 
    extend functionality to non-linear(Non CNN) fnApprox classes
    """
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.psi_shape = pow(params_dict['order']+1, 2)
        self.init_weights()
        self.curr_weights = self.weights_init
        #this current_Q just serves as a copy of the Q function inside the fnAprox
        self.current_Q = [0]*len(params_dict['actions'])

    def init_weights(self):
        self.weights_init = {}
        for a in self.params_dict['actions']:
            self.weights_init.update({a:np.zeros((1, self.psi_shape))})

    def reset_weights(self):
        self.init_weights()
        self.curr_weights = self.weights_init

    def update_Q(self, psi_s):
        for each_action in self.params_dict['actions']:
            self.current_Q[each_action] = ((self.curr_weights[each_action]).dot(self.psi_s))
        return self.current_Q

    def update_params(self, a, a_prime, psi, psi_prime, r):
        delta = r + self.params_dict['gamma'] * self.curr_weights[a_prime].dot(psi_prime) - self.curr_weights[a].dot(psi)
        self.curr_weights[a] += self.params_dict['alpha'] * delta * psi.T

    def calculate_fourier(self, s):
        """
        Calculates a new state representation based on 
        order of the fourier basis and state input
        """
        self.psi_s=[]
        ns = self.norm_state(s)
        for c in itertools.product(range(self.params_dict['order']+1),repeat=2):
            self.psi_s.append(np.cos((math.pi * (np.asarray(c)).dot(ns))))
        self.psi_s = np.asarray(self.psi_s).reshape(len(self.psi_s),1)
        return self.psi_s

    def norm_state(self, s):
        """
        As solution qualities are in [0,1), no need to normalize q
        Taking max t as a general t=200 to be safe
        """
        norm_s = s[0]
        norm_t = (s[1]-0)/200
        assert norm_t < 1
        assert norm_s < 1
        return [norm_s, norm_t]



class agent:
    def __init__(self, params_dict=None):
        """
        Initialize the agent with internal parameters
        TO DO: provide args options in main to initialize
        agent with specific policies/Qvalue functions
        """
        self.params = params_dict
        self.fn = fnApprox(self.params) 
        # Q for the agent initialized here
        self.curr_Q = [0]*len(self.params['actions'])
        # update_Q has been pushed to fnApprox

    def next_action(self):
        p = random.uniform(0,1)
        if p<=self.params['epsilon']:
            a = np.argmax(self.curr_Q)
        else:
            a = random.randint(0,1)
        return a 

    def mini_method(self, env):
        """
        sets the initial Q for each episode
        """
        s = env.reset()
        self.psi = self.fn.calculate_fourier(s)
        self.curr_Q = self.fn.update_Q(self.psi)

    def run_sarsa(self, env):
        # print('Running sarsa on:', env.___name___)
        # RAS_mean = np.zeros((self.params['trials'], self.params['episodes']))
        # self.mini_method(env)
        # for t in range(self.params['trials']):
        #     self.fn.reset_weights()
        #     for e in range(self.params['episodes']):
        #         self.mini_method(env)
        #         a = self.next_action()
        #         r_cum = 0
        #         while True:
        #             s_prime, r, done = env.step(a)
        #             r_cum += r
        #             if done:
        #                 break
        #             psi_prime = self.fn.calculate_fourier(s_prime)
        #             self.curr_Q = self.fn.update_Q(psi_prime)
        #             a_prime = self.next_action()
        #             self.fn.update_params(a, a_prime, self.psi, psi_prime, r)
        #             s = s_prime
        #             a = a_prime
        #             self.psi = psi_prime
        #         RAS_mean[t][e] = r_cum
        # plot_mean(RAS_mean)
        return

    def run_Q(self, env):
        print('Running Q-learning on:', env.___name___)
        RAS_mean = np.zeros((self.params['trials'], self.params['episodes']))
        self.mini_method(env)
        for t in range(self.params['trials']):
            self.fn.reset_weights()
            for e in range(self.params['episodes']):
                s = env.reset()
                r_cum = 0
                
                r_track = 0
                r_track_list = []

                while True:
                    self.psi = self.fn.calculate_fourier(s)
                    self.curr_Q = self.fn.update_Q(self.psi)
                    a = self.next_action()
                    # print('taken action -', a)
                    s_prime, r, done = env.step(a)
                    r_cum += r
                    
                    r_track += r
                    if a==0 and e==199:
                        r_track_list.append(r_track)
                        r_track = 0

                    if done:
                        break
                    psi_prime = self.fn.calculate_fourier(s_prime)
                    self.curr_Q = self.fn.update_Q(psi_prime)
                    a_prime = self.next_action()
                    self.fn.update_params(a, a_prime, self.psi, psi_prime, r)
                    s = s_prime
                RAS_mean[t][e] = r_cum
        maxU_list = env.optim_point()
        # assert len(r_track_list)==len(maxU_list)
        for i in range(49):
            print('{} : {}'.format(r_track_list[i], maxU_list[i]))
        
        # plot_mean(RAS_mean)
        return

