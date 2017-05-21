from __future__ import division

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import tsp
import tsp_solver
from utils import get_instance_name, get_solution_quality_map, get_naive_solution_qualities, get_line_components, get_solution_qualities, get_max_length, get_solution_quality_averages, get_trimmed_solution_qualities
from scipy.special import expit
import math

TIME_COST = 1
INTRINSIC_VALUE_MULTIPLIER = 200


def get_time_cost(time):
    return np.multiply(-TIME_COST, time)


def get_estimated_utility(x, a, b, c):
    # return a * expit(x + b) + c
    # return x / (a * np.sqrt(1 + np.power(x, 2)) + b)
    # return a * np.tanh(x + b) + c
    # return a * np.arctan(x + b) + c
    return a * np.arctan(x + b) + c


def save_performance_profiles(results_filename, instances_filename, directory):
    online_pp_losses = []
    average_pp_losses = []

    with open(instances_filename) as f:
        lines = f.readlines()

        for line in lines:
            instance_filename, _ = get_line_components(line)

            instance_name = get_instance_name(instance_filename)
            plot_filename = directory + '/' + instance_name + '.png'

            online_pp_loss, average_pp_loss = save_performance_profile(results_filename, instance_filename, plot_filename)

            online_pp_losses.append(online_pp_loss)
            average_pp_losses.append(average_pp_loss)

    print("Online Performance Profile Mean Loss: %f" % np.average(online_pp_losses))
    print("Average Performance Profile Mean Loss: %f" % np.average(average_pp_losses))


def save_performance_profile(results_filename, instance_filename, plot_filename):
    plt.figure()
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Solution Quality')

    map = get_solution_quality_map(results_filename)

    solution_quality_groups = [np.multiply(solution_qualities, INTRINSIC_VALUE_MULTIPLIER) for solution_qualities in map.values()]
    max_length = get_max_length(solution_quality_groups)
    trimmed_solution_quality_groups = get_trimmed_solution_qualities(solution_quality_groups, max_length)
    solution_quality_averages = get_solution_quality_averages(trimmed_solution_quality_groups)

    solution_qualities = np.multiply(map[instance_filename], INTRINSIC_VALUE_MULTIPLIER)
    length = len(solution_qualities)
    time = range(length)

    plt.plot(time, solution_quality_averages[:length], color='b', zorder=2)

    sample_start = 10
    sample_end = 50

    initial_gray = 0.9
    terminal_gray = 0
    difference = initial_gray - terminal_gray
    decrement = difference / (sample_end - sample_start)
    current_color = initial_gray

    time_cost = get_time_cost(time)
    plt.plot(time, time_cost, color='y', zorder=4)

    true_comprehensive_values = solution_qualities + time_cost
    plt.plot(time, true_comprehensive_values, color='g', zorder=5)

    true_best_time = list(true_comprehensive_values).index(max(true_comprehensive_values))
    plt.scatter([true_best_time], true_comprehensive_values[true_best_time], color='m', zorder=10, s=25)

    plt.scatter(-1, 33, color='m', s=25)
    plt.text(0, 30, "%0.2f - Best Reward" % true_comprehensive_values[true_best_time], zorder=10)

    for sample_limit in range(sample_start, sample_end):
        try:
            parameters, _ = curve_fit(get_estimated_utility, time[:sample_limit], solution_qualities[:sample_limit])
            estimates = get_estimated_utility(time, parameters[0], parameters[1], parameters[2])
            plt.plot(time, estimates, color=str(current_color), zorder=1)
            current_color -= decrement

            estimated_comprehensive_values = estimates + time_cost
            estimated_best_time = list(estimated_comprehensive_values).index(max(estimated_comprehensive_values))

            # Stopping conditions
            if estimated_best_time <= sample_limit:
                plt.scatter([estimated_best_time], true_comprehensive_values[estimated_best_time], color='c', zorder=7, s=25)

                plt.scatter(-1, 23, color='c', s=25)
                plt.text(0, 20, "%0.2f - Best Reward w/ Online Performance Profile" % true_comprehensive_values[estimated_best_time], zorder=10)

                break
            else:
                plt.scatter([estimated_best_time], true_comprehensive_values[estimated_best_time], color=str(current_color), zorder=5, s=25)
        except Exception as e:
            pass

    plt.scatter(time, solution_qualities, color='r', s=25, zorder=3)

    average_comprehensive_values = solution_quality_averages + get_time_cost(range(len(solution_quality_averages)))
    average_best_time = list(average_comprehensive_values).index(max(average_comprehensive_values))
    plt.scatter([average_best_time], true_comprehensive_values[average_best_time], color='y', zorder=6, s=25)
    plt.scatter(-1, 13, color='y', s=25)
    plt.text(0, 10, "%0.2f - Best Reward w/ Average Performance Profile" % true_comprehensive_values[average_best_time], zorder=10)

    online_pp_loss = true_comprehensive_values[true_best_time] - true_comprehensive_values[estimated_best_time]
    average_pp_loss = true_comprehensive_values[true_best_time] - true_comprehensive_values[average_best_time]

    plt.text(0, -10, "%0.2f - Online Performance Profile Loss" % online_pp_loss, zorder=10)
    plt.text(0, -20, "%0.2f - Average Performance Profile Loss" % average_pp_loss, zorder=10)

    plt.savefig(plot_filename)

    return online_pp_loss, average_pp_loss


def display_solution_qualities(instances_filename):
    solution_quality_map = {}

    with open(instances_filename) as f:
        lines = f.readlines()

        for line in lines:
            instances_filename, optimal_distance = get_line_components(line)

            cities = tsp.load_instance(instances_filename)
            start_city = list(cities)[0]
            statistics = {'time': [], 'distances': []}
            tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

            solution_quality_map[instances_filename] = get_naive_solution_qualities(statistics['distances'], optimal_distance)

    print(json.dumps(solution_quality_map))


def display_performance_profiles(instances_filename):
    with open(instances_filename) as f:
        lines = f.readlines()

        solution_quality_groups = [get_solution_qualities(line) for line in lines]
        max_length = get_max_length(solution_quality_groups)
        trimmed_solution_quality_groups = get_trimmed_solution_qualities(solution_quality_groups, max_length)
        solution_quality_averages = get_solution_quality_averages(trimmed_solution_quality_groups)

        plt.title('Performance Profile')
        plt.xlabel('Time')
        plt.ylabel('Solution Quality')

        plt.plot(range(len(solution_quality_averages)), solution_quality_averages, color='b')

        for solution_qualities in solution_quality_groups:
            plt.scatter(range(len(solution_qualities)), solution_qualities, color='r', s=0.1)

        plt.show()


def display_projections(instance_filename, optimal_distance, results_filename):
    with open(results_filename) as f:
        lines = f.readlines()

        solution_quality_groups = [get_solution_qualities(line) for line in lines]
        max_length = get_max_length(solution_quality_groups)
        trimmed_solution_quality_groups = get_trimmed_solution_qualities(solution_quality_groups, max_length)
        solution_quality_averages = get_solution_quality_averages(trimmed_solution_quality_groups)

        cities = tsp.load_instance(instance_filename)
        start_city = list(cities)[0]
        statistics = {'time': [], 'distances': []}
        tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

        # solution_qualities = get_standard_solution_qualities(statistics['distances'], optimal_distance)
        solution_qualities = get_naive_solution_qualities(statistics['distances'], optimal_distance)

        plt.title('Performance Profile')
        plt.xlabel('Time')
        plt.ylabel('Solution Quality')

        plt.scatter(statistics['time'], solution_qualities, color='r')
        plt.plot(range(len(solution_quality_averages)), solution_quality_averages, color='b')

        for sample_limit in range(4, 20):
            try:
                parameters, _ = curve_fit(get_estimated_utility, statistics['time'][:sample_limit], solution_qualities[:sample_limit])
                estimates = get_estimated_utility(statistics['time'], parameters[0], parameters[1], parameters[2], parameters[3])
                plt.plot(statistics['time'], estimates, color='r')
            except:
                print('Encountered exception')

        plt.show()


def main():
    save_performance_profiles('results/naive.json', 'instances/50-tsp/instances.csv', 'plots')
    # display_solution_qualities('instances/50-tsp/instances.csv')

if __name__ == '__main__':
    main()
