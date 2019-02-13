from __future__ import division

import random

import numpy as np

from function_approximation import FunctionApproximation


# TODO Discuss with Prakhar each agent
# TODO Maybe make self.env.actions accessed through a getter
# TODO Clean up some more
class Agent:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.function_approximation = FunctionApproximation(params, env)
        self.action_value_function = self.function_approximation.get_initial_action_value_function()

    def get_action(self):
        if random.random() > self.params["epsilon"]:
            return np.argmax(self.action_value_function)
        return random.choice(self.env.ACTIONS)

    def run_q_learning(self, statistics):
        for _ in range(self.params['episodes']):
            state = self.env.reset()

            while True:
                psi = self.function_approximation.calculate_fourier(state)
                self.action_value_function = self.function_approximation.update_action_value_function(psi)

                action = self.get_action()

                next_state, reward, is_episode_done = self.env.step(action)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["smoothed_errors"].append(np.average(statistics["errors"][:-20]))
                    statistics["stopping_points"].append(next_state[1])

                    break

                psi_prime = self.function_approximation.calculate_fourier(next_state)
                self.action_value_function = self.function_approximation.update_action_value_function(psi_prime)

                next_action = self.get_action()
                self.function_approximation.update_weights(action, next_action, psi, psi_prime, reward)
                state = next_state

    def run_sarsa(self, statistics):
        for _ in range(self.params['episodes']):
            state = self.env.reset()

            psi = self.function_approximation.calculate_fourier(state)
            self.action_value_function = self.function_approximation.update_action_value_function(psi)

            action = self.get_action()

            while True:
                next_state, reward, is_episode_done = self.env.step(action)

                next_psi = self.function_approximation.calculate_fourier(state)
                self.action_value_function = self.function_approximation.update_action_value_function(next_psi)

                next_action = self.get_action()

                self.function_approximation.update_weights(action, next_action, psi, next_psi, reward)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["smoothed_errors"].append(np.average(statistics["errors"][:-20]))
                    statistics["stopping_points"].append(next_state[1])

                    break

                state = next_state
                action = next_action
                psi = next_psi

                self.params['epsilon'] *= self.params['decay']
