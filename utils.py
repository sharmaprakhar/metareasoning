import json
import math

import matplotlib.pyplot as plt
import numpy as np


def plot_data(y_data, yerr=None, x_data=None):
    if not x_data:
        x_axis = range(len(y_data))

    plt.errorbar(x_axis, y_data, yerr=yerr, ecolor="r")
    plt.title("Performance")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.show()


def get_instances(filename):
    with open(filename) as file:
        return json.load(file)


def plot_mean(arr):
    mean = arr.mean(axis=0, keepdims=True)
    mean = mean.reshape(mean.shape[1])

    std = np.std(arr, axis=0, keepdims=True)
    std = std.reshape(std.shape[1])

    plot_data(mean, std)


def init_interface(num_features, num_actions, l=0.999):
    num_theta = num_features * num_actions

    theta = np.zeros((num_features * num_actions, 1))
    A = np.zeros((num_features + num_theta, num_features + num_theta))
    b = np.zeros((num_features + num_theta, 1))
    z_stat = np.zeros((num_features + num_theta, 1))

    return theta, l, A, b, z_stat


def get_action_probabilities(features, num_actions, theta):
    action_probabilities = np.zeros((num_actions))

    num_features = len(features)

    for i in range(num_actions):
        start_index = num_features * i
        end_index = num_features * i + num_features
        action_probabilities[i] = features.T.dot(theta[start_index:end_index])

    action_probabilities_exp = np.exp(action_probabilities)
    action_probabilities_sum = np.sum(action_probabilities_exp)
    action_probabilities = action_probabilities_exp / action_probabilities_sum

    return action_probabilities


def dlnpi(features, theta, num_actions, action, num_features):
    action += 1

    action_probabilities = get_action_probabilities(features, num_actions, theta)

    result = np.zeros((1, (num_actions * num_features)))
    for i in range(num_actions):
        if i == action:
            result[0, num_features*action : (num_features * action + num_features)] = features.T * (1 - action_probabilities[action])
        else:
            result[0, num_features * i : (num_features * i + num_features)] = -1 * features.T * action_probabilities[i]

    return result


def get_action(action_probabilities, actions):
    return np.random.choice(actions, p=action_probabilities)


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
