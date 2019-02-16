from __future__ import division

import random

import numpy as np


class Agent:
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.action_value_function = {state: [random.random(), random.random()] for state in env.get_states()}

    def transfer(self, params, env):
        self.params = params
        self.env = env

    def get_optimal_action(self, state):
        return np.argmax(self.action_value_function[state])

    def get_action(self, state):
        if random.random() > self.params["epsilon"]:
            return self.get_optimal_action(state)
        return random.choice(self.env.ACTIONS)

    def run_q_learning(self, statistics):
        print("Running tabular Q-learning with the parameters {}".format(self.params))

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            action = self.get_action(state)

            while True:
                next_state, reward, is_episode_done = self.env.step(action)

                current_value = self.action_value_function[state][action]
                estimated_value = reward + self.params["gamma"] * max(self.action_value_function[next_state])
                value_error = estimated_value - current_value
                self.action_value_function[state][action] += self.params["alpha"] * value_error

                next_action = self.get_action(state)

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
                action = next_action

    def run_sarsa(self, statistics):
        print("Running tabular SARSA with the parameters {}".format(self.params))

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            action = self.get_action(state)

            while True:
                next_state, reward, is_episode_done = self.env.step(action)

                current_value = self.action_value_function[state][action]
                estimated_value = reward + self.params["gamma"] * self.action_value_function[next_state][action]
                value_error = estimated_value - current_value
                self.action_value_function[state][action] += self.params["alpha"] * value_error

                next_action = self.get_action(state)

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
                action = next_action
