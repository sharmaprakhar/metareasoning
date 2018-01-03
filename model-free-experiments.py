import itertools
import operator
import random

import numpy as np

import utils

QUALITY_CLASS_COUNT = 100
QUALITY_CLASSES = range(QUALITY_CLASS_COUNT)
QUALITY_CLASS_BOUNDS = np.linspace(0, 1, QUALITY_CLASS_COUNT)

ACTIONS = ['STOP', 'CONTINUE']

LEARNING_RATE = 0.01
EPSILON = 0.2

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


def get_Q_function(instances, alpha, epsilon):
    Q = get_initial_Q_function()
    pi = get_pi(Q)

    for instance in instances:
        print('Episode:', instance)

        q_class = 0
        t_class = 0
        s = (q_class, t_class)
        a = 'CONTINUE'

        for q in instances[instance]['estimated_qualities']:
            next_q_class = utils.digitize(q, QUALITY_CLASS_BOUNDS)
            next_t_class = t_class + 1
            next_s = (next_q_class, next_t_class)

            r = U(next_q_class, next_t_class) - U(q_class, t_class)
            next_Q_values = get_Q_values(Q, next_s)
            Q[s][a] += alpha * (r + max(next_Q_values) - Q[s][a])
            pi[s] = max(Q[s].items(), key=operator.itemgetter(1))[0]

            q_class = next_q_class
            t_class = next_t_class
            s = (q_class, t_class)
            a = pi[s] if random.random() > epsilon else random.choice(ACTIONS)

            if a is 'STOP':
                break

    return Q


def main():
    pass


if __name__ == '__main__':
    main()
