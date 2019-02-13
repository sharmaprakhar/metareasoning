import itertools
import math

import numpy as np


# TODO Should I rename this class?
# TODO Verify with Prakhar that I didn't break anything
class FunctionApproximation:
    def __init__(self, param, env):
        self.param = param
        self.env = env
        self.weights = self.get_initial_weights()
        self.action_value_function = self.get_initial_action_value_function()

    def get_initial_weights(self):
        shape = (1, pow(self.param['order'] + 1, 2))
        return {action: np.zeros(shape) for action in self.env.ACTIONS}

    def get_initial_action_value_function(self):
        return [0] * len(self.env.ACTIONS)

    def update_weights(self, action, next_action, psi, next_psi, reward):
        delta = reward + self.param['gamma'] * self.weights[next_action].dot(next_psi) - self.weights[action].dot(psi)
        self.weights[action] += self.param['alpha'] * delta * psi.T

    def calculate_fourier(self, state):
        normalized_state = self.get_normalized_state(state)

        psi = []
        for c in itertools.product(range(self.param['order'] + 1), repeat=2):
            psi.append(np.cos(math.pi * np.asarray(c).dot(normalized_state)))
        psi = np.asarray(psi).reshape(len(psi), 1)

        return psi

    def update_action_value_function(self, psi):
        return [self.weights[action].dot(psi) for action in self.env.ACTIONS]

    def get_normalized_state(self, state):
        normalized_quality = state[0] / self.env.QUALITY_CLASS_COUNT
        normalized_time = state[1] / self.env.TIME_CLASS_COUNT
        return [normalized_quality, normalized_time]
