##############################################################################
# TSP solution profiles as RL environments
# Input: json files comtaining TSP instances
# import scheme: import en
# Author: Prakhar Sharma
##############################################################################
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
        """
        gets instances as a dict. self.dataset is a list that holds 
        each of the 'instances' found in the JSON file as a list. So 
        its a list of lists (2D array)
        """
        instances = utils.get_instances(performance_file)
        self.dataset = []
        for key,v in instances.items():
            self.dataset.append(instances[key]['estimated_qualities'])
        self.append_time()
        self.instance_count = len(self.appended_dataset)        

    def append_time(self):
        """
        each 'instance' found inside the JSON file is converted into
        a list of lists that holds [q,t]
        """
        self.appended_dataset = []
        for each_inst_est_q in self.dataset:
            t = 0
            new_list = [[q] for q in each_inst_est_q]
            for e in new_list:
                e.append(t)
                t += 1
            self.appended_dataset.append(new_list)

class env:
        """
        self.d - holds tha dataset (can be called as self.d.appended_dataset)
        """
    def __init__(self, performance_file):
        self.d = gen_dataset(performance_file)
        self.___name___ = 'TSP instance ' + str(performance_file)

    def len_trajectory(self):
        """
        Note: the length of each instance found in the JSON file is not constant.
        This function returns the length of the particular instance
        """
        return len(self.d.appended_dataset[self.instance_identifier])

    def is_overflow(self):
        """
        current quality identifier tracks each [q,t] element of 'each instance found
        in the JSON file' in self.appended_dataset above
        """
        if self.current_quality_identifier == self.len_trajectory()-1:
            return True
        return False

    def optim_point(self):
        """
        This function returns the optimal point for each instance found in the JSON file.
        It calculates the utility of each point in each instance found in the JSON file and returns 
        the max
        """
        alpha = 200.0
        # beta = 0.25
        beta = 0.03
        maxU_list = []
        for inst in self.d.appended_dataset:
            maxU = 0
            for qt in inst:
                U = alpha*qt[0] - math.exp(beta*qt[1])
                if U>maxU:
                    maxU=U
            maxU_list.append(maxU)
        return maxU_list

    def s(self):
        """
        This function returns the next [q,t]

        self.instance_identifier represents which 'instance' among all the 
        instances found inside the JSON file is agent currently seeing

        self.current_quality_identifier represents which [q,t] pair of the particular 
        instance found inside the JSON file (given by self.instance_identifier) si the agent seeing.

        Together self.instance_identifier self.current_quality_identifier define a particular [q,t] pair
        """
        return self.d.appended_dataset[self.instance_identifier][self.current_quality_identifier]

    def ep_end(self):
        if self.instance_identifier == (self.d.instance_count-1):
            return True
        return False

    def reset(self):
        """
        reset always gives out the [q_0,t_0] element of the first instance found inside the JSON file

        self.trajectory_length keeps track of how long the current instance is (that is how many [q,t] pairs
        exist within that instance)
        """
        self.instance_identifier = 0
        self.current_quality_identifier = 0
        self.trajectory_length = self.len_trajectory()
        return self.s()

    def step(self, action):
        def calculate_reward():
            """
            reward calculator: utility(previous state)-utility(current state). The usual 
            formula for utility is employed.
            """
            alpha = 200.0
            beta = 0.03
            prev_s = self.d.appended_dataset[self.instance_identifier][self.current_quality_identifier-1]
            curr_s = self.d.appended_dataset[self.instance_identifier][self.current_quality_identifier]
            U_prev = alpha*prev_s[0] - math.exp(beta*prev_s[1])
            U_curr = alpha*curr_s[0] - math.exp(beta*curr_s[1])
            return U_curr-U_prev
        #action - CONTINUE
        if action == 1:
            """
            if it is the last instance (self.instance_identifier == (self.d.instance_count-1)): end the episode and send a TERMINATE signal
            """
            if self.ep_end():
                return self.reset(), 0, TERMINAL
            if self.is_overflow():
                """
                all the [q,t] pairs of this instance have been processed, start with the next instance. self.ep_end() above checks if it is the last 
                instance (in which case episode ends in the if statement above)
                """
                self.instance_identifier += 1
                self.current_quality_identifier = 0
                reward = 0 #choose better reward for overflow - Issue
                # print("ACTION CONTINUE(overflow) - instance identifier", self.instance_identifier)
                # print("ACTION CONTINUE(overflow) - quality identifier: {}".format(self.current_quality_identifier))
                return self.s(), reward, NON_TERMINAL
            else:
                """
                if its not the end of the episode and its not end of the instance, just return the next [q,t]
                """
                self.current_quality_identifier += 1
                # print("\nACTION CONTINUE - instance identifier", self.instance_identifier)
                # print("ACTION CONTINUE - quality identifier: {}".format(self.current_quality_identifier))
                reward = calculate_reward()
                return self.s(), reward, NON_TERMINAL
        
        #action - STOP
        else:
            """
            if the incoming action is STOP, first check if its the last instance, if yes, end the episode. 
            Otherwise, just return the [q_0,t_0] of the next instance

            instance is incremented by self.instance_identifier
            self.current_quality_identifier=0 points to [q_0,t_0]

            """
            if self.ep_end():
                return self.reset(), 0, TERMINAL
            self.instance_identifier += 1
            self.current_quality_identifier = 0
            reward = 0 
            return self.s(), reward, NON_TERMINAL



######### DESCRIPTION ##############
"""
The dataset (d.appended_dataset) looks like:

[  [  [q0_0,t0_0],[q0_1,t0_1], ...  [q0_n,t0_n]  ]  , [ [q1_0,t1_0],[q1_1,t1_1], ...  [q1_m,t1_m]  ]  ]

where [q0_1,t0_1] points to self.instance_identifier=0, self.current_quality_identifier=1 (0th instance, 1st solution quality. Indexing starts from 0)

In the 50 TSP json file, instance_count=50 and len_trajectory is different for different instance

"""


# e = env(performance_file)
# print(e.d.appended_dataset[0])
# print('\n')
# print(e.d.appended_dataset[1])
# print('\n')
# print(e.d.appended_dataset[2])
