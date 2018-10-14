import numpy as np
import json
import math
import utils
import random

performance_file = '../simulations/50-tsp-0.1s.json'
INSTANCE_COUNT = 5000
TERMINAL = 1
NON_TERMINAL = 0


class gen_dataset:
    """
    returns a dict of the json dataset
    """
    def __init__(self, performance_file=None, dir= None):
        instances = utils.get_instances(performance_file)
        self.dataset = []
        for key,v in instances.items():
            self.dataset.append(instances[key]['estimated_qualities'])
        self.append_time()
        self.instance_count = len(self.appended_dataset)        

    def append_time(self):
        self.appended_dataset = []
        for each_inst_est_q in self.dataset:
            t = 0
            new_list = [[q] for q in each_inst_est_q]
            for e in new_list:
                e.append(t)
                t += 1
            self.appended_dataset.append(new_list)

class env:
    def __init__(self, performance_file):
        """
        use instance to control which data agent is trained on
        """
        self.d = gen_dataset(performance_file)
        self.___name___ = 'TSP instance ' + str(performance_file)

    def len_trajectory(self):
        return len(self.d.appended_dataset[self.instance_identifier])

    def is_overflow(self):
        if self.current_quality_identifier == self.len_trajectory()-1:
            return True
        return False

    def s(self):
        return self.d.appended_dataset[self.instance_identifier][self.current_quality_identifier]

    def ep_end(self):
        if self.instance_identifier == (self.d.instance_count-1):
            return True
        return False

    def reset(self):
        self.instance_identifier = 0
        self.current_quality_identifier = 0
        self.trajectory_length = self.len_trajectory()
        return self.s()

    def step(self, action):
        def calculate_reward():
            alpha = 200.0
            beta = 0.25
            prev_s = self.d.appended_dataset[self.instance_identifier][self.current_quality_identifier-1]
            curr_s = self.d.appended_dataset[self.instance_identifier][self.current_quality_identifier]
            U_prev = alpha*prev_s[0] - math.exp(beta*prev_s[1])
            U_curr = alpha*curr_s[0] - math.exp(beta*curr_s[1])
            return U_curr-U_prev
        #action - CONTINUE
        if action == 1:
            if self.ep_end():
                return self.reset(), 0, TERMINAL
            if self.is_overflow():
                self.instance_identifier += 1
                self.current_quality_identifier = 0
                reward = 0 #choose better reward for overflow - Issue
                # print("ACTION CONTINUE(overflow) - instance identifier", self.instance_identifier)
                # print("ACTION CONTINUE(overflow) - quality identifier: {}".format(self.current_quality_identifier))
                return self.s(), reward, NON_TERMINAL
            else:
                self.current_quality_identifier += 1
                # print("\nACTION CONTINUE - instance identifier", self.instance_identifier)
                # print("ACTION CONTINUE - quality identifier: {}".format(self.current_quality_identifier))
                reward = calculate_reward()
                # print("reward", reward)
                return self.s(), reward, NON_TERMINAL
        
        #action - STOP
        else:
            if self.ep_end():
                return self.reset(), 0, TERMINAL
            self.instance_identifier += 1
            # print("\nACTION STOP - instance identifier", self.instance_identifier)
            self.current_quality_identifier = 0
            reward = 0 
            return self.s(), reward, NON_TERMINAL


# e = env(performance_file)
# print(e.d.appended_dataset[0])
# print('\n')
# print(e.d.appended_dataset[1])
# print('\n')
# print(e.d.appended_dataset[2])
