import itertools
import operator
import random
import time
from multiprocessing import Manager, Process

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import linear_model, svm

import tsp
import utils

QUALITY_CLASS_COUNT = 100
QUALITY_CLASSES = range(QUALITY_CLASS_COUNT)
QUALITY_CLASS_BOUNDS = np.linspace(0, 1, QUALITY_CLASS_COUNT)
TIME_CLASSES = range(50)
ACTIONS = ['STOP', 'CONTINUE']
SLEEP_INTERVAL = 0.5
EPISODES = 200
LEARNING_RATE = 0.01
EPSILON = 0.1

def k_opt_tsp_solver(states, start_state, iterations, memory):
    tour = tsp.get_initial_random_tour(states, start_state)
    cities = tsp.get_swappable_cities(tour)
    distance = tsp.get_tour_distance(tour)

    for _ in range(iterations):
        has_changed = False

        best_tour = tour
        best_distance = distance
        for first_key, second_key in itertools.combinations(cities, 2):
            current_tour = tsp.get_mutated_tour(tour, first_key, second_key)
            current_distance = tsp.get_tour_distance(current_tour)

            if current_distance < best_distance:
                best_tour = current_tour
                best_distance = current_distance

                has_changed = True

        tour = best_tour
        distance = best_distance

        memory['q'] = distance
        memory['t'] = time.time()

        if not has_changed:
            break

    return tour


def get_S():
    return list(itertools.product(QUALITY_CLASSES, TIME_CLASSES))


def get_initial_Q_function():
    return {s: {'STOP': random.random(), 'CONTINUE': random.random()} for s in get_S()}


def get_pi(Q):
    return {s: max(Q[s].items(), key=operator.itemgetter(1))[0] for s in get_S()}


def U(q, t):
    return 100 * q - np.exp(0.25 * t)


def get_Q_values(Q, s):
    return [Q[s][a] for a in Q[s]]


def get_Q_function(episodes, learning_rate, epsilon):
    print('{"episodes": %f, "learning_rate": %f, "epsilon": %f}' % (episodes, learning_rate, epsilon))

    actual_values = []
    optimal_values = []

    Q = get_initial_Q_function()
    pi = get_pi(Q)

    for episode in range(episodes):
        is_terminated = False
        observed_values = []

        start_q = 0
        start_t = time.time()

        memory = Manager().dict()
        memory['q'] = start_q
        memory['t'] = start_t

        q_class = 0
        t_class = 0
        s = (q_class, t_class)
        a = 'CONTINUE'

        states = tsp.get_instance(50, 0, 10000, 1)
        start_state = list(states)[0]

        heuristic = tsp.get_mst_distance(start_state, states)

        process = Process(target=k_opt_tsp_solver, args=(states, start_state, 1000, memory))
        process.start()

        time.sleep(SLEEP_INTERVAL)

        while process.is_alive():
            next_q = heuristic / (memory['q'] + 0.00001)
            next_t = memory['t']

            next_q_class = utils.digitize(next_q, QUALITY_CLASS_BOUNDS)
            next_t_class = round((next_t - start_t) / SLEEP_INTERVAL)
            next_s = (next_q_class, next_t_class)

            current_value = U(next_q_class, next_t_class)
            r = current_value - U(q_class, t_class)
            next_Q_values = get_Q_values(Q, next_s)
            Q[s][a] += learning_rate * (r + max(next_Q_values) - Q[s][a])
            pi[s] = max(Q[s].items(), key=operator.itemgetter(1))[0]

            a = pi[next_s] if random.random() > epsilon else random.choice(ACTIONS)

            q_class = next_q_class
            t_class = next_t_class
            s = (q_class, t_class)

            if a is 'STOP' and not is_terminated:
                actual_values.append(current_value)
                is_terminated = True

            observed_values.append(current_value)

            time.sleep(SLEEP_INTERVAL)

        print(observed_values)
        optimal_values.append(max(observed_values))

        print('{"episode": %d, "actual_value": %f, "optimal_value": %f}' % (episode, actual_values[-1], optimal_values[-1]))

    f = open('experiment-%d-%f-%f.png' % (episodes, learning_rate, epsilon), 'w')
    for value in actual_values:
        f.write('%d\n' % value)
    f.close()

    return Q


def main():
    epsilons = [1, 0]

    for epsilon in epsilons:
        get_Q_function(EPISODES, epsilon, EPSILON)

    # with open('experiment-200-0.010000-0.000000.png') as f:
    #     lines = f.read().splitlines()

    # lines = [float(line) for line in lines]

    # values = []
    # for i in range(len(lines)):
    #     start = 0 if i - 100 <= 0 else i - 100
    #     values.append(np.average(lines[start:i]))

    # plt.figure()
    # plt.title('Learning Curve')
    # plt.xlabel('Episodes')
    # plt.ylabel('Value')
    # plt.plot(range(len(values)), values)
    # plt.savefig('test.png')
    # plt.show()


if __name__ == "__main__":
    main()
