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
    This class implements methods to approximate an
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

    def update_Q(self, psi_s):
        for each_action in self.params_dict['actions']:
            self.current_Q[each_action] = ((self.curr_weights[each_action]).dot(self.psi_s))
        return self.current_Q

    def update_params(self, a, a_prime, psi, psi_prime, r):
        # print('a prime', a_prime)
        # print('current weights for a prime', self.curr_weights[a_prime])
        # print('current weights for a prime shape', self.curr_weights[a_prime].shape)
        # print('psi prime', psi_prime)
        # print('psi prime shape', psi_prime.shape)
        delta = r + self.params_dict['gamma'] * self.curr_weights[a_prime].dot(psi_prime) - self.curr_weights[a].dot(psi)
        self.curr_weights[a] += self.params_dict['alpha'] * delta * psi.T

    def calculate_fourier(self, s):
        """
        Calculates a new state representation based on 
        order of the fourier basis and current state 
        """
        self.psi_s=[]
        ns = self.norm_state(s) #write norm state
        for c in itertools.product(range(self.params_dict['order']+1),repeat=2):
            self.psi_s.append(np.cos((math.pi * (np.asarray(c)).dot(ns))))
        self.psi_s = np.asarray(self.psi_s).reshape(len(self.psi_s),1)
        return self.psi_s

    def norm_state(self, s):
        """
        As solution qualities are in [0,1), no need to normalize q
        How do I normalize t -- ? from timesteps !
        """
        return [0.5, 1]



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
        if p>=self.params['epsilon']:
            a = np.argmax(self.curr_Q)
        else:
            a = random.randint(0,1)
        return a 

    def mini_method(self, env):
        """
        sets the initial Q for each episode
        """
        # get initial s
        s = env.reset()
        # get fourier basis state representation
        self.psi = self.fn.calculate_fourier(s)
        #update Q for that state for each action
        self.curr_Q = self.fn.update_Q(self.psi)

    def run_sarsa(self, env):
        """
        The initialization of weights for each
        action is pushed to fnApprox
        """
        print('Running sarsa on:', env.___name___)
        max_reward = 0
        RAS_mean = np.zeros((self.params['trials'], self.params['episodes']))
        self.mini_method(env)
        for t in range(self.params['trials']):
            for e in range(self.params['episodes']):
                # initial s already set while setiing curr_Q above in mini_method
                self.mini_method(env)
                a = self.next_action()
                r_cum = 0
                for ti in range(self.params['timesteps']):
                    s_prime, r, done = env.step(a)
                    r_cum += r
                    if r_cum>max_reward:
                        max_reward = r_cum
                    if done:
                        break
                    psi_prime = self.fn.calculate_fourier(s_prime)
                    self.curr_Q = self.fn.update_Q(psi_prime)
                    a_prime = self.next_action()
                    # update weights for each a using TD update rule
                    self.fn.update_params(a, a_prime, self.psi, psi_prime, r)
                    # previous s/a/psi = next s/a/psi and repeat
                    s = s_prime
                    a = a_prime
                    self.psi = psi_prime
                # The current algo runs till either
                # data exhaustion or till stopped by RL agent
                RAS_mean[t][e] = r_cum
        print(max_reward)
        #plot_mean(RAS_mean)
        return

    def run_Q(self):
        print('Running Q-learning on:', env.___name___)
        max_reward = 0
        RAS_mean = np.zeros((self.params['trials'], self.params['episodes']))
        self.mini_method(env)
        for t in range(self.params['trials']):
            for e in range(self.params['episodes']):
                # initial s already set while setiing curr_Q above in mini_method
                self.mini_method(env)
                r_cum = 0
                for ti in range(self.params['timesteps']):
                    # self.mini_method(env) - write this function !! - calculate fourier not this
                    a = self.next_action()
                    s_prime, r, done = env.step(a)
                    r_cum += r
                    if r_cum>max_reward:
                        max_reward = r_cum
                    if done:
                        break
                    psi_prime = self.fn.calculate_fourier(s_prime)
                    self.curr_Q = self.fn.update_Q(psi_prime)
                    a_prime = self.next_action()
                    # update weights for each a using TD update rule
                    self.fn.update_params(a, a_prime, self.psi, psi_prime, r)
                    # previous s/a/psi = next s/a/psi and repeat
                    s = s_prime
                    self.psi = psi_prime
                # The current algo runs till either
                # data exhaustion or till stopped by RL agent
                RAS_mean[t][e] = r_cum
        print(max_reward)
        plot_mean(RAS_mean)
        return

