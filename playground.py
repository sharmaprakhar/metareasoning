import numpy as np
import random
import matplotlib.pyplot as plt

SOLUTION_QUALITY_CLASS_COUNT = 200
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(0, 110, SOLUTION_QUALITY_CLASS_COUNT + 1)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

ALPHA_CLASS_COUNT = 100
ALPHA_CLASS_BOUNDS = np.linspace(0.00001, 0.0005, ALPHA_CLASS_COUNT + 1)
ALPHA_CLASSES = range(ALPHA_CLASS_COUNT)

ITERATIONS = 50000
SAMPLES = 100000

SIZE = 100
BIAS = 25
VARIANCE = 10


def get_performance_profile(samples):
    for _ in range(SAMPLES):
        data = get_data(SIZE, BIAS, VARIANCE)
        m, n = np.shape(x)
        weights = get_weights


def get_weights(x, y, weights, alpha, m, iterations):
    xT = x.transpose()

    for i in range(0, iterations):
        hypothesis = np.dot(x, weights)
        loss = hypothesis - y

        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost = %f | alpha = %f " % (i, cost, alpha))

        gradient = np.dot(xT, loss) / m

        weights = weights - alpha * gradient

    return weights


def get_data(size, bias, variance):
    x = np.zeros(shape=(size, 2))
    y = np.zeros(shape=size)

    for i in range(0, size):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance

    return x, y

x, y = get_data(100, 25, 10)
