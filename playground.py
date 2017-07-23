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

PROBLEMS = 10000
SLEEP_INTERVAL = 0.1
LEARNING_RATE = 0.1

SIZE = 100
BIAS = 25
VARIANCE = 10

NUM_ITERATIONS = 5
ITERATIONS = range(NUM_ITERATIONS)


def get_initial_action_value_function():
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, ITERATIONS)) 
    return {(state, action): 0 for action in ALPHA_CLASSES for state in states}


def get_greedy_action(action_value_function, state):
    return max(action_value_function[state], key=action_value_function[state].get)


def get_policy(action_value_function):
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, ITERATIONS)) 
    return {state: get_greedy_action(action_value_function, state) for state in states}


# TODO Should this work on classes or actual values?
def get_utility(quality, time):
    return 100 * quality - 0.1 * time


def get_action_values(action_value_function, state):
    return [action_value_function[state][action] for action in action_value_function[state]]


def get_action_value_function():
    action_value_function = get_initial_action_value_function()

    for _ in range(PROBLEMS):
        examples, labels = get_problem(SIZE, BIAS, VARIANCE)
        policy = get_policy(action_value_function)

        previous_quality = 0
        start_time = time.time()

        memory = Manager().dict()
        quality = memory['quality'] = previous_quality
        time = memory['time'] = start_time

        # TODO Calculate the state properly
        state = (quality, time)

        action = memory['action'] = policy[state]

        process = Process(target=get_weights, args=(examples, labels, memory))
        process.start()
        time.sleep(SLEEP_INTERVAL)

        while process.is_alive():
            next_quality = memory['quality']
            next_time = memory['time']

            next_state = (next_quality, next_time)
            
            reward = get_utility(next_quality, next_time) - get_utility(quality, time)
            action_values = get_action_values(action_value_function, next_state)
            action_value_function[state][action] = action_value_function[state][action] + LEARNING_RATE * (reward + max(action_values) - action_value_function[state][action])

            # TODO Is this right?
            quality = next_quality
            time = next_time

            memory['action'] = policy[state]
            time.sleep(SLEEP_INTERVAL)

    return action_value_function



def get_weights(examples, labels, memory):
    num_examples, num_features = np.shape(examples)
    weights = np.ones(num_features)    
    transposed_examples = examples.transpose()

    for i in range(0, NUM_ITERATIONS):
        hypothesis = np.dot(examples, weights)
        loss = hypothesis - labels
        cost = np.sum(loss ** 2) / (2 * num_examples)

        memory['quality'] = cost
        memory['time'] = time.time()
        alpha = memory['action']

        gradient = np.dot(transposed_examples, loss) / num_examples
        weights = weights - alpha * gradient

        print("Iteration %d | Cost = %f | alpha = %f " % (i, cost, alpha))
        
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
    print(get_action_value_function())


if __name__ == "__main__":
    main()
