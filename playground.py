import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process, Manager
import time
import utils
import operator

SOLUTION_QUALITY_CLASS_COUNT = 100
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(100, 0, SOLUTION_QUALITY_CLASS_COUNT + 1)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

ALPHA_CLASS_COUNT = 5
ALPHA_CLASSES = np.linspace(0.0001, 0.0006, ALPHA_CLASS_COUNT + 1)

PROBLEMS = 100
SLEEP_INTERVAL = 0.01
LEARNING_RATE = 0.2

SIZE = 100
BIAS = 25
VARIANCE = 10

NUM_ITERATIONS = 50000
ITERATIONS = range(NUM_ITERATIONS)


def get_initial_action_value_function():
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, ITERATIONS)) 
    return {state: {action: 0 for action in ALPHA_CLASSES} for state in states}


def get_policy(action_value_function):
    states = list(itertools.product(SOLUTION_QUALITY_CLASSES, ITERATIONS))
    return {state: max(action_value_function[state].items(), key=operator.itemgetter(1))[0] for state in states}


def digitize(item, bins):
    if bins[0] < item:
        return 0

    for i, _ in enumerate(bins):
        if i + 1 < len(bins):
            if bins[i] >= item > bins[i + 1]:
                return i
             
    return len(bins) - 1


def get_utility(quality, time):
    return 100 * quality - 0.1 * time


def get_action_values(action_value_function, state):
    return [action_value_function[state][action] for action in action_value_function[state]]


def get_action_value_function():
    action_value_function = get_initial_action_value_function()

    for _ in range(PROBLEMS):
        examples, labels = get_problem(SIZE, BIAS, VARIANCE)
        print("Initialized problem: {size = %d, bias = %f, variance = %f}..." % (SIZE, BIAS, VARIANCE))

        previous_quality = 0
        start_time = time.time()

        memory = Manager().dict()
        q = memory['quality'] = previous_quality
        t = memory['time'] = start_time
        print("Initialized current conditions: (q = %f, t = %f)..." % (q, t))

        q_class = 0
        t_class = 0
        state = (q_class, t_class)
        print("Initialized current state: (q_class = %f, t_class = %f)..." % (state[0], state[1]))

        policy = get_policy(action_value_function)
        action = memory['action'] = policy[state] if random.random() > 0.1 else random.choice(ALPHA_CLASSES)
        print("Initialized current action: (action = %f)..." % action)

        print("Starting process...")
        process = Process(target=get_weights, args=(examples, labels, memory))
        process.start()

        print("Sleeping...")
        time.sleep(SLEEP_INTERVAL)

        while process.is_alive():
            print("Monitoring...")

            next_q = memory['quality']
            next_t = memory['time'] - start_time
            print("Set next conditions: (next_q = %f, next_t = %f)" % (next_q, next_t))

            next_q_class = digitize(next_q, SOLUTION_QUALITY_CLASS_BOUNDS)
            next_t_class = round(next_t / SLEEP_INTERVAL)
            next_state = (next_q_class, next_t_class)
            print("Set next state: (next_q_class = %f, next_t_class = %f)" % (next_state[0], next_state[1]))

            reward = get_utility(next_q, next_t) - get_utility(q, t)
            action_values = get_action_values(action_value_function, next_state)
            action_value_function[state][action] = action_value_function[state][action] + LEARNING_RATE * (reward + max(action_values) - action_value_function[state][action])

            q = next_q
            t = next_t
            q_class = digitize(next_q, SOLUTION_QUALITY_CLASS_BOUNDS)
            t_class = round(next_t / SLEEP_INTERVAL)
            state = (q_class, t_class)
            print("Set current state: (q_class = %f, t_class = %f)" % (state[0], state[1]))

            action = memory['action'] = policy[next_state] if random.random() > 0.1 else random.choice(ALPHA_CLASSES)
            print("Set current action: (action = %f)..." % action)

            print("Sleeping...")
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

        alpha = memory['action']
        memory['quality'] = cost
        memory['time'] = time.time()

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
    get_action_value_function()


if __name__ == "__main__":
    main()
