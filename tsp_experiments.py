from __future__ import division

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import tsp
import tsp_solver
from utils import get_instance_name, get_solution_quality_map, get_naive_solution_qualities, get_line_components, get_solution_qualities, get_max_length, get_solution_quality_averages, get_trimmed_solution_qualities


def model(x, a, c, d):
    return a * np.arctan(x + c) + d


def save_performance_profiles(results_filename, instances_filename, directory):
    with open(instances_filename) as f:
        lines = f.readlines()

        for line in lines:
            instance_filename, _ = get_line_components(line)

            instance_name = get_instance_name(instance_filename)
            plot_filename = directory + '/' + instance_name + '.png'

            save_performance_profile(results_filename, instance_filename, plot_filename)


def save_performance_profile(results_filename, instance_filename, plot_filename):
    plt.figure()
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Solution Quality')

    map = get_solution_quality_map(results_filename)

    solution_quality_groups = [solution_qualities for solution_qualities in map.values()]
    max_length = get_max_length(solution_quality_groups)
    trimmed_solution_quality_groups = get_trimmed_solution_qualities(solution_quality_groups, max_length)
    solution_quality_averages = get_solution_quality_averages(trimmed_solution_quality_groups)

    solution_qualities = map[instance_filename]
    plt.scatter(range(len(solution_qualities)), solution_qualities, color='r', s=0.2)

    plt.plot(range(len(solution_quality_averages)), solution_quality_averages, color='b')

    plt.savefig(plot_filename)


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

    data = json.dumps(solution_quality_map)
    print(data)


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
                parameters, _ = curve_fit(model, statistics['time'][:sample_limit], solution_qualities[:sample_limit])
                estimates = model(statistics['time'], parameters[0], parameters[1], parameters[2])
                plt.plot(statistics['time'], estimates, color='r')
            except:
                print('Encountered exception')

        plt.show()


def main():
    save_performance_profiles('results/naive.json', 'instances/50-tsp/instances.csv', 'plots')
    # display_solution_qualities('instances/50-tsp/instances.csv')

if __name__ == '__main__':
    main()
