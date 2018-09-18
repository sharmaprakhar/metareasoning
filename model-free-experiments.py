import itertools
import operator
import random

import matplotlib.pyplot as plt
import numpy as np

import utils

QUALITY_CLASS_COUNT = 100
QUALITY_CLASSES = range(QUALITY_CLASS_COUNT)
QUALITY_CLASS_BOUNDS = np.linspace(0, 1, QUALITY_CLASS_COUNT)

TIME_CLASS_COUNT = 200
TIME_CLASSES = range(TIME_CLASS_COUNT)

ACTIONS = ['STOP', 'CONTINUE']

LEARNING_RATE = 0.1
EPSILON = 0.3

INSTANCE_COUNT = 5000

def get_S():
    return list(itertools.product(QUALITY_CLASSES, TIME_CLASSES))


def get_initial_Q_value_function():
    return {s: {'STOP': random.random(), 'CONTINUE': random.random()} for s in get_S()}


def get_pi(Q):
    return {s: max(Q[s].items(), key=operator.itemgetter(1))[0] for s in get_S()}


def U(q, t):
    return 200 * q - np.exp(0.25 * t)


def get_Q_values(Q, s):
    return [Q[s][a] for a in Q[s]]


def get_Q_value_function(instances, alpha, epsilon, default_Q_value_function=None):
    print('{"learning_rate": %f, "epsilon": %f}' % (alpha, epsilon))

    statistics = {
        'actual_stopping_values': [],
        'smoothed_actual_stopping_values': []
    }

    Q = get_initial_Q_value_function() if default_Q_value_function is None else default_Q_value_function
    pi = get_pi(Q)

    for instance in instances:
        is_terminated = False

        q_class = 0
        t_class = 0
        s = (q_class, t_class)
        a = 'CONTINUE'

        for q in instance:
            next_q_class = utils.digitize(q, QUALITY_CLASS_BOUNDS)
            next_t_class = t_class + 1
            next_s = (next_q_class, next_t_class)

            next_value = U(next_q_class, next_t_class)
            r = next_value - U(q_class, t_class)
            next_Q_values = get_Q_values(Q, next_s)
            Q[s][a] += alpha * (r + max(next_Q_values) - Q[s][a])
            pi[s] = max(Q[s].items(), key=operator.itemgetter(1))[0]

            q_class = next_q_class
            t_class = next_t_class
            s = (q_class, t_class)
            a = pi[s] if random.random() > epsilon else random.choice(ACTIONS)

            if a is 'STOP' and not is_terminated:
                statistics['actual_stopping_values'].append(next_value)
                statistics['smoothed_actual_stopping_values'].append(np.average(statistics['actual_stopping_values'][:-100]))

                is_terminated = True

                break

    return Q, statistics


def main():
    instances = utils.get_instances('simulations/50-tsp-0.1s.json')

    dataset = []
    for _ in range(INSTANCE_COUNT):
        key = random.choice(list(instances.keys()))
        dataset.append(instances[key]['estimated_qualities'])

    Q_value_function, training_statistics = get_Q_value_function(dataset, LEARNING_RATE, EPSILON)
    # _, test_statistics = get_Q_value_function(dataset[:1000], 0, 0, default_Q_value_function=Q_value_function)

    plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.rcParams['grid.linestyle'] = "-"
    plt.grid(True)

    plt.xlabel('Trials')
    plt.ylabel('Utility')

    plt.plot(range(len(training_statistics['smoothed_actual_stopping_values'])), training_statistics['smoothed_actual_stopping_values'], color='green', linewidth=2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
