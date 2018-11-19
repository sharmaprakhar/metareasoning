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
            #temp_arr = np.asarray(temp_list)
            self.sol_dict[instance_num] = temp_list
            instance_num += 1

        self.append_time()

    def append_time(self):
        self.states = {}
        for k,v in self.sol_dict.items():
            somekey = k
            u = [y for l in v for y in l]
            t = [tp for tp in range(len(v[0]))]
            self.states[k] = list(zip(u,t))
            
        



#d = dataset(performance_file)
#print(type(d.sol_dict[0][0]))

class environment:
    def __init__(self, performance_file):
        """
        use instance to control which data agent is trained on
        """
        self.d = dataset(performance_file)
        self.instance = 0

    def reset(self):
        return self.d.states[self.instance][0]

e = environment(performance_file)
print(e.reset())