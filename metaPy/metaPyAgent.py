import gym
import sys
import math
import random
import itertools
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class fnApprox:
    def __init__(self, actions, states=None):
        self.weights_init = {}
        for a in 
        self.ac = num_actions
        self.st = num_states
        self.curr_weights = self.weights_init

    def reset_weights(self):
        self.weights = self.weights_init

    def calculate_fourier(self):
        """
        Calculates a new state representation based on 
        order of the fourier basis and current state 
        """
        self.psi_s=[]
        ns = self.norm_state() #write norm state
        for c in itertools.product(range(self.order+1),repeat=2):
            self.psi_s.append(np.cos((math.pi * (np.asarray(c)).dot(ns))))
        self.psi_s = np.asarray(self.psi_s).reshape(len(self.psi_s),1)
        #return self.psi_s

    def norm_state(self):
        """
        Tailored to each environment
        Could also be included with the env 
        which is the dataset pulled from JSON
        """
        pass



class agent:
    def __init__(self, params_dict=None):
        self.params = params_dict
        self.fn = fnApprox(self.params['actions'], self.params['states']) #default - fourier 
        #init_Q has been pushed to each individual algorithm

    def update_Q(self, placeholder_inputs):
        #write the TD update rule here
        pass

    def choose_best_action(self, updated_Q, policy_type):
        #policy could either be softmax or epsilon greedy
        pass

    def init_Q():
        

    def next_action(self):
        p = random.uniform(0,1)
        if p>=self.params['epsilon']:
            a = np.argmax(self.Q)  
        else:
            a = random.randint(0,1)
        return a 

    def run_sarsa(self, env):
        print('Running sarsa on:', env)
        self.fn.calculate_fourier()
        self.

        # Note the initialization of weights for each
        # action is pushed to fnApprox
        for t in in range(self.params['trials']):
            for e in range(episodes):
                s = env.reset()
                a = self.next_action()



        return

    def run_Q(self):
        print('will run Q')
        return

