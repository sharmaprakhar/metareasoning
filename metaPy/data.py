import numpy as np
import json

performance_file = '../simulations/50-tsp-0.1s.json'

class dataset():
    """
    returns a dict of the json dataset
    TODO: augment time information with each datum
    """
    def __init__(self, performance_file=None, dir= None):
        with open(performance_file) as f:
            performance_data = json.load(f)
        self.sol_dict = {}
        instance_num = 0
        for inst,q in performance_data.items():
            temp_list = []
            for k,v in q.items():
                temp_list.append(v)
            temp_arr = np.asarray(temp_list)
            self.sol_dict[instance_num] = temp_arr
            instance_num += 1

d = dataset(performance_file)