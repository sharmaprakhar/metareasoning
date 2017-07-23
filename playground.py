import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process, Manager
import time

SOLUTION_QUALITY_CLASS_COUNT = 50
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(0, 100, SOLUTION_QUALITY_CLASS_COUNT + 1)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

ALPHA_CLASS_COUNT = 4
ALPHA_CLASS_BOUNDS = np.linspace(0.0001, 0.0005, ALPHA_CLASS_COUNT + 1)
ALPHA_CLASSES = range(ALPHA_CLASS_COUNT)

TIME_PERIOD = 0.1
LEARNING_RATE = 0.1

ITERATIONS = 5
TIME_STEPS = range(ITERATIONS)

PROBLEMS = 10000
SIZE = 100
BIAS = 25
VARIANCE = 10


def get_initial_action_value_function():
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, TIME_STEPS)) 
    return {(state, action): 0 for action in ALPHA_CLASSES for state in states}


def get_initial_policy():
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, TIME_STEPS)) 
    return {state: random.choice(ALPHA_CLASSES) for state in states}


def get_greedy_action(action_value_function, state):
    return max(action_value_function[state], key=action_value_function[state].get)


def get_policy(action_value_function):
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, TIME_STEPS)) 
    return {state: get_greedy_action(action_value_function, state) for state in states}


def get_action_value_function(problems):
    action_value_function = get_initial_action_value_function()

    for _ in range(problems):
        examples, labels = get_problem(SIZE, BIAS, VARIANCE)

        policy = get_policy(action_value_function)

        d = Manager().dict()
        state = d['state'] = (0, 0)
        action = d['action'] = policy[state]
        current_value = d['current_value'] = 0
        previous_value = d['previous_value'] = None

        process = Process(target=get_weights, args=(d,))
        process.start()
        time.sleep(TIME_PERIOD)

        while process.is_alive():
            reward = d['current_value'] - d['previous_value']
            action_value_function[state, action] = action_value_function[state, action] + LEARNING_RATE * (reward + max(reward) - action_value_function[state, action])

            
            d['action'] = policy[d['state']]
            time.sleep(TIME_PERIOD)

    return action_value_function



def get_weights(x, y, weights, m, iterations):
    transposed_x = x.transpose()

    for i in range(0, iterations):
        hypothesis = np.dot(x, weights)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)

        gradient = np.dot(transposed_x, loss) / m
        weights = weights - GLOBAL_ALPHA * gradient

        print("Iteration %d | Cost = %f | alpha = %f " % (i, cost, GLOBAL_ALPHA))

    return weights


def get_problem(size, bias, variance):
    examples = np.zeros(shape=(size, 2))
    labels = np.zeros(shape=size)

    for i in range(0, size):
        examples[i][0] = 1
        examples[i][1] = i
        labels[i] = (i + bias) + random.uniform(0, 1) * variance

    return examples, labels


def main():
    manager = Manager()

    d = manager.dict()
    d['action'] = '1'
    d['value'] = 2
    d['state'] = 0

    p1 = Process(target=f, args=(d,))
    p2 = Process(target=f, args=(d,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == "__main__":
    main()