import itertools

import numpy as np

import utils

# TODO Clean up some more
class Environment:
    STOP_ACTION = 0
    CONTINUE_ACTION = 1
    ACTIONS = [STOP_ACTION, CONTINUE_ACTION]

    ALPHA = 200
    BETA = 0.3

    QUALITY_CLASS_COUNT = 200
    TIME_CLASS_COUNT = 200

    QUALITY_CLASSES = range(QUALITY_CLASS_COUNT)
    TIME_CLASSES = range(TIME_CLASS_COUNT)

    def __init__(self, problem_file):
        self.dataset = utils.get_dataset(problem_file)
        self.instance_id = 0
        self.state_id = 0

    def get_states(self):
        return list(itertools.product(self.QUALITY_CLASSES, self.TIME_CLASSES))

    def get_reward(self):
        previous_quality, previous_time = self.get_previous_state()
        current_quality, current_time = self.get_current_state()

        previous_utility = utils.get_time_dependent_utility(previous_quality, previous_time, self.ALPHA, self.BETA)
        current_utility = utils.get_time_dependent_utility(current_quality, current_time, self.ALPHA, self.BETA)

        return current_utility - previous_utility

    def get_utility(self):
        raw_state = self.dataset[self.instance_id][self.state_id]
        quality, time = self.get_normalized_state(raw_state)
        return utils.get_time_dependent_utility(quality, time, self.ALPHA, self.BETA)

    def get_optimal_utility(self):
        max_utility = float("-inf")

        for raw_state in self.dataset[self.instance_id]:
            quality, time = self.get_normalized_state(raw_state)
            utility = utils.get_time_dependent_utility(quality, time, self.ALPHA, self.BETA)

            if utility > max_utility:
                max_utility = utility

        return max_utility

    def get_normalized_state(self, raw_state):
        raw_quality, raw_time = raw_state
        bounds = np.linspace(0, 1, self.QUALITY_CLASS_COUNT)
        return utils.digitize(raw_quality, bounds), raw_time

    def get_previous_state(self):
        raw_state = self.dataset[self.instance_id][self.state_id - 1]
        return self.get_normalized_state(raw_state)

    def get_current_state(self):
        raw_state = self.dataset[self.instance_id][self.state_id]
        return self.get_normalized_state(raw_state)

    def is_last_instance(self):
        return self.instance_id == len(self.dataset) - 1

    def is_episode_done(self):
        return self.state_id == len(self.dataset[self.instance_id]) - 1

    def reset(self):
        self.instance_id = 0 if self.is_last_instance() else self.instance_id + 1
        self.state_id = 0
        return self.get_current_state()

    def step(self, action):
        if action == self.STOP_ACTION or self.is_episode_done():
            return self.get_current_state(), 0, True

        self.state_id += 1
        return self.get_current_state(), self.get_reward(), False


def main():
    print("Testing the environment...")
    env = Environment("problems/test-problem.json")

    print("Running episode 1...")
    print(env.reset())
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.STOP_ACTION))

    print("Running episode 2...")
    print(env.reset())
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.STOP_ACTION))

    print("Running episode 3...")
    print(env.reset())
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.CONTINUE_ACTION))
    print(env.step(env.CONTINUE_ACTION))


if __name__ == "__main__":
    main()