import itertools
import operator
import random

import matplotlib.pyplot as plt
import numpy as np

# import utils_orig
from utils import *
from utils_orig import *


QUALITY_CLASS_COUNT = 100
QUALITY_CLASSES = range(QUALITY_CLASS_COUNT)
QUALITY_CLASS_BOUNDS = np.linspace(0, 1, QUALITY_CLASS_COUNT)

TIME_CLASS_COUNT = 200
TIME_CLASSES = range(TIME_CLASS_COUNT)
INSTANCE_COUNT = 5000

class tabQ:
    def __init__(self, params_dict=None):
        self.params = params_dict
        prod = list(itertools.product(QUALITY_CLASSES, TIME_CLASSES))
        self.initial_q = {s: [0,0] for s in prod}
        self.d = digitize
        self.curr_Q = self.initial_q

    def next_action(self, state):
        p = random.uniform(0,1)
        if p<=self.params['epsilon']:
            a = np.argmax(self.curr_Q[state])
        else:
            a = random.randint(0,1)
        return a 

    def run_tabQ(self, env):
        print('Running tabular Q-learning on:', env.___name___)
        RAS_mean = np.zeros((self.params['trials'], self.params['episodes']))
        for t in range(self.params['trials']):
            for e in range(self.params['episodes']):
                _s = env.reset()
                s = (self.d(_s[0], QUALITY_CLASS_BOUNDS), _s[1])
                r_cum = 0
                while True:
                    a = self.next_action(s)
                    _s_prime, r, done = env.step(a)
                    s_prime = (self.d(_s_prime[0], QUALITY_CLASS_BOUNDS), _s_prime[1])
                    r_cum += r
                    if done:
                        break
                    max_next_Q = max(self.curr_Q[s_prime])
                    self.curr_Q[s][a] += self.params['epsilon']*(r + (self.params['gamma'] * max_next_Q) - self.curr_Q[s][a])
                    a_prime = self.next_action(s_prime)
                    s = s_prime
                RAS_mean[t][e] = r_cum
        maxU_list = env.optim_point()
        plot_mean(RAS_mean)
        return