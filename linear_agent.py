from __future__ import division

import random

import numpy as np

from function_approximation import FunctionApproximation


class Agent:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.function_approximation = FunctionApproximation(params, env)
        self.action_value_function = self.function_approximation.get_initial_action_value_function()

    def transfer(self, params, env, action_value_function):
        self.params = params
        self.env = env
        self.function_approximation = FunctionApproximation(params, env, action_value_function)

    def get_optimal_action(self, state):
        return np.argmax(self.action_value_function[state])

    def get_action(self, state):
        if random.random() > self.params["epsilon"]:
            return self.get_optimal_action(state)
        return random.choice(self.env.ACTIONS)

    def run_q_learning(self, statistics):
        print("Running linear Q-learning with the parameters {}".format(self.params))

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            psi = self.function_approximation.calculate_fourier_approximation(state)

            while True:
                self.action_value_function = self.function_approximation.update_action_value_function(state, psi)
                action = self.get_action(state)

                next_state, reward, is_episode_done = self.env.step(action)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["smoothed_errors"].append(np.average(statistics["errors"][-50:]))
                    statistics["stopping_points"].append(next_state[1])
                    statistics["smoothed_stopping_points"].append(np.average(statistics["stopping_points"][-50:]))

                    self.params["epsilon"] *= self.params["decay"]

                    break

                next_action = self.get_optimal_action(next_state)
                next_psi = self.function_approximation.calculate_fourier_approximation(next_state)
                self.function_approximation.update_weights(action, next_action, psi, next_psi, reward)

                state = next_state
                psi = next_psi

    def run_sarsa(self, statistics):
        print("Running linear SARSA with the parameters {}".format(self.params))

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            psi = self.function_approximation.calculate_fourier_approximation(state)

            self.action_value_function = self.function_approximation.update_action_value_function(state, psi)
            action = self.get_action(state)

            while True:
                next_state, reward, is_episode_done = self.env.step(action)
                next_psi = self.function_approximation.calculate_fourier_approximation(next_state)

                self.action_value_function = self.function_approximation.update_action_value_function(next_state, next_psi)
                next_action = self.get_action(next_state)

                self.function_approximation.update_weights(action, next_action, psi, next_psi, reward)

                if is_episode_done:
                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["smoothed_errors"].append(np.average(statistics["errors"][-50:]))
                    statistics["stopping_points"].append(next_state[1])
                    statistics["smoothed_stopping_points"].append(np.average(statistics["stopping_points"][-50:]))

                    self.params["epsilon"] *= self.params["decay"]

                    break

                state = next_state
                psi = next_psi
                action = next_action
