from __future__ import division

import randomized_tour_improver
import tsp
import matplotlib.pyplot as plt
from utils import get_standard_solution_qualities, get_line_components, get_solution_qualities, get_max_length, get_solution_quality_averages, get_trimmed_solution_qualities


def display_performance_profile(filename, optimal_distance):
    print('File:', filename)

    cities = tsp.load_instance(filename)
    start_city = list(cities)[0]
    statistics = {'time': [], 'distances': []}
    randomized_tour_improver.k_opt_solve(cities, start_city, statistics)

    solution_qualities = get_standard_solution_qualities(statistics['distances'], optimal_distance)

    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Solution Quality')
    plt.scatter(statistics['time'], solution_qualities, color='r')
    plt.show()


def display_solution_qualities(filename):
    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            filename, optimal_distance = get_line_components(line)

            cities = tsp.load_instance(filename)
            start_city = list(cities)[0]
            statistics = {'time': [], 'distances': []}
            randomized_tour_improver.k_opt_solve(cities, start_city, statistics)

            solution_qualities = get_standard_solution_qualities(statistics['distances'], optimal_distance)

            print solution_qualities


def display_performance_profiles(filename):
    with open(filename) as f:
        lines = f.readlines()

        solution_quality_groups = [get_solution_qualities(line) for line in lines]
        max_length = get_max_length(solution_quality_groups)
        trimmed_solution_quality_groups = get_trimmed_solution_qualities(solution_quality_groups, max_length)
        solution_quality_averages = get_solution_quality_averages(trimmed_solution_quality_groups)

        plt.title('Performance Profile')
        plt.xlabel('Time')
        plt.ylabel('Solution Quality')

        for solution_qualities in solution_quality_groups:
            size = len(solution_qualities)
            plt.scatter(range(size), solution_qualities, color='r', s=0.1)

        plt.plot(range(max_length), solution_quality_averages, color='b')

        plt.show()


def main():
    # display_performance_profile('instances/50-tsp/instance-11.tsp', 11228)
    # display_solution_qualities('instances/50-tsp/instances.csv')
    display_performance_profiles('results/50-tsp-results.txt')

if __name__ == '__main__':
    main()
