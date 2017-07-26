import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process, Manager
import time
import utils
import operator
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
from keras import losses


# SOLUTION_QUALITY_CLASS_COUNT = 56
# SOLUTION_QUALITY_CLASS_BOUNDS = list(np.linspace(100, 20, 17)) + list(np.linspace(19.5, 0, 40))
SOLUTION_QUALITY_CLASS_COUNT = 200
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(100, 0, SOLUTION_QUALITY_CLASS_COUNT + 1)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

ALPHA_CLASS_COUNT = 5
ALPHA_CLASSES = np.linspace(0.0001, 0.0006, ALPHA_CLASS_COUNT + 1)

TIME_CLASSES = range(150)

EPISODES = 10
SLEEP_INTERVAL = 0.5
LEARNING_RATE = 0.1

SIZE = 100
BIAS = 25
VARIANCE = 10

NUM_ITERATIONS = 50000


def get_S():
    return list(itertools.product(SOLUTION_QUALITY_CLASSES, TIME_CLASSES))


# def get_initial_Q_function():
#     return {state: {action: 0 for action in ALPHA_CLASSES} for state in get_S()}


def get_initial_Q_function():
    model = Sequential()
    model.add(Dense(8, input_shape=(3,), activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_absolute_error')
    return model


# def get_pi(Q):
#     return {s: max(Q[s].items(), key=operator.itemgetter(1))[0] for s in get_S()}


def get_pi(Q):
    return {(q, t): max({a: Q.predict(np.array([q, t, a]).reshape(1, 3))[0][0] for a in ALPHA_CLASSES}.items(), key=operator.itemgetter(1))[0] for q, t in get_S()}


def digitize(item, bins):
    if bins[0] < item:
        return 0

    for i, _ in enumerate(bins):
        if i + 1 < len(bins):
            if bins[i] >= item > bins[i + 1]:
                return i

    return len(bins) - 1


def U(q, t):
    return 1000 * (1 / q) - 0.05 * t


# def get_Q_values(Q, s):
#     return [Q[s][a] for a in Q[s]]


def get_Q_values(Q, s):
    q, t = s
    return [Q.predict(np.array([q, t, a]).reshape(1, 3))[0][0] for a in ALPHA_CLASSES]


def get_Q_function():
    q_class_history = []
    problem_history = []

    Q = get_initial_Q_function()
    pi = get_pi(Q)

    for problem in range(EPISODES):
        print("Episode %d" % problem)

        X, y = get_training_set(SIZE, BIAS, VARIANCE)

        previous_q = 150
        start_t = time.time()

        memory = Manager().dict()
        q = memory['q'] = previous_q
        t = memory['t'] = start_t

        q_class = 0
        t_class = 0
        s = (q_class, t_class)
        a = memory['a'] = pi[s] if random.random() > 0.1 else random.choice(ALPHA_CLASSES)

        process = Process(target=get_weights, args=(X, y, memory))
        process.start()
        time.sleep(SLEEP_INTERVAL)

        while process.is_alive():
            next_q = memory['q']
            next_t = memory['t']

            next_q_class = digitize(next_q, SOLUTION_QUALITY_CLASS_BOUNDS)
            next_t_class = round((next_t - start_t) / SLEEP_INTERVAL)
            next_s = (next_q_class, next_t_class)

            r = U(next_q, next_t) - U(q, t)
            Q_values = get_Q_values(Q, next_s)

            x = np.array([round(q_class / SOLUTION_QUALITY_CLASS_COUNT, 2), round(t_class / len(TIME_CLASSES), 2), a]).reshape(1, 3)
            y = np.array([r + max(Q_values)]).reshape(1, 1)

            print(r)
            print(Q_values)

            Q.train_on_batch(x, y)
            pi[s] = max({a: Q.predict(np.array([round(q_class / 200, 2), round(t_class / len(TIME_CLASSES), 2), a]).reshape(1, 3))[0][0] for a in ALPHA_CLASSES}.items(), key=operator.itemgetter(1))[0]
            # Q[s][a] = Q[s][a] + LEARNING_RATE * (r + max(Q_values) - Q[s][a])
            # pi[s] = max(Q[s].items(), key=operator.itemgetter(1))[0]

            q = next_q
            t = next_t
            q_class = next_q_class
            t_class = next_t_class
            s = (q_class, t_class)

            a = memory['a'] = pi[next_s] if random.random() > 0.1 else random.choice(ALPHA_CLASSES)

            time.sleep(SLEEP_INTERVAL)

        Q.reset_states()
        q_class_history.append(q_class)
        problem_history.append(problem)

    plt.scatter(problem_history, q_class_history)
    plt.show()

    return Q


def get_weights(examples, labels, memory):
    num_examples, num_features = np.shape(examples)
    weights = np.ones(num_features)
    transposed_examples = examples.transpose()

    for i in range(0, NUM_ITERATIONS):
        hypothesis = np.dot(examples, weights)
        loss = hypothesis - labels
        cost = np.sum(loss ** 2) / (2 * num_examples)

        alpha = memory['a']
        memory['q'] = cost
        memory['t'] = time.time()

        gradient = np.dot(transposed_examples, loss) / num_examples
        weights = weights - alpha * gradient

    return weights


def get_training_set(size, bias, variance):
    examples = np.zeros(shape=(size, 2))
    labels = np.zeros(shape=size)

    for i in range(0, size):
        examples[i][0] = 1
        examples[i][1] = i
        labels[i] = (i + bias) + random.uniform(0, 1) * variance

    return examples, labels


def main():
    Q = get_Q_function()
    pi = get_pi(Q)
    print(pi)

if __name__ == "__main__":
    main()
