import json
import math

import numpy as np

def get_instances(filename):
    with open(filename) as file:
        return json.load(file)


def digitize(item, bins):
    for i, _ in enumerate(bins):
        if i + 1 < len(bins):
            if bins[i] <= item < bins[i + 1]:
                return i
    return len(bins) - 1


def get_dataset(problem_file, increment):
    instances = get_instances(problem_file)

    dataset = []
    for instance in instances.values():
        entries = list(enumerate(instance["estimated_qualities"]))
        dataset.append([(quality, time) for time, quality in entries[0:len(entries):increment]])

    return dataset


def get_time_dependent_utility(quality, time, alpha, beta):
    return alpha * quality - math.exp(beta * time)


def get_convergence_data(data, threshold, period):
    threshold_iterations = 0

    for i in range(len(data) - 1):
        difference = abs(data[i] - data[i - 1])

        if difference <= threshold:
            threshold_iterations += 1
            if threshold_iterations >= period:
                return {"episode": i, "error": data[i]}
        else:
            threshold_iterations = 0


def get_smoothed_values(data, window_size):
    smoothed_values = []

    for i in range(1, len(data)):
        truncated_data = data[:i]
        windowed_data = truncated_data[-window_size:]
        smoothed_values.append(np.average(windowed_data))

    return smoothed_values
