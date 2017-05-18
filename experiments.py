from __future__ import division

import json

import matplotlib.pyplot as plt
import numpy as np

import anytime_astar_solver
import astar_solver
import n_puzzle
import n_puzzle_problem
from utils import Problem


def f(x, a, c, d):
    return a * np.arctan(x + c) + d


def get_model(a, c, d):
    return lambda x: f(x, a, c, d)


def parse_line(line):
    filename, raw_optimal_distance = line.split(',')

    stripped_optimal_distance = raw_optimal_distance.strip()
    truncated_optimal_distance = stripped_optimal_distance.split('.')[0]
    casted_optimal_distance = int(truncated_optimal_distance)

    return filename, casted_optimal_distance


def get_solution_qualities(distances, optimal_distance):
    return [1 - ((distance - optimal_distance) / optimal_distance) for distance in distances]


def get_components(line):
    raw_states, raw_start_state, raw_optimal_distance = line.split(';')

    states = [tuple(x) for x in json.loads(raw_states)]
    start_state = tuple(json.loads(raw_start_state))
    optimal_distance = json.loads(raw_optimal_distance)

    return states, start_state, optimal_distance


# def get_averages(solution_quality_groups):
#     return [sum(solution_qualities) / len(solution_qualities) for solution_qualities in zip(*solution_quality_groups)]
#
#
# def get_average_performance_profile(filename, iterations=10000, limit=10):
#     solution_quality_groups = []
#
#     with open(filename) as f:
#         lines = f.readlines()
#
#         for i, line in enumerate(lines):
#             if i == limit:
#                 break
#
#             filename, optimal_distance = parse_line(line)
#
#             print 'File:', filename
#
#             cities = old.tsp.load_instance(filename)
#             start_city = random.choice(list(cities))
#
#             statistics = {'time': [], 'distances': []}
#             old.randomized_tour_improver.naive_solve(cities, start_city, statistics, iterations=iterations, is_detailed=True)
#
#             solution_qualities = get_solution_qualities(statistics['distances'], optimal_distance)
#             solution_quality_groups.append(solution_qualities)
#
#     return range(iterations), get_averages(solution_quality_groups)


def main():
    experiment()


def experiment():
    # puzzle = np.matrix([[5, 8, 7],
    #                     [6, 0, 1],
    #                     [4, 3, 2]])

    # 20
    # puzzle = np.matrix([[3, 8, 5, 6],
    #                     [2, 0, 9, 11],
    #                     [13, 1, 4, 7],
    #                     [14, 10, 15, 12]])

    # 30
    # puzzle = np.matrix([[7, 11, 9, 8],
    #                     [6, 1, 2, 3],
    #                     [0, 10, 5, 12],
    #                     [4, 13, 14, 15]])

    puzzle = n_puzzle.get_random_puzzle(34)
    print(puzzle)

    problem = Problem(
        puzzle,
        n_puzzle_problem.is_goal,
        n_puzzle_problem.get_successors,
        n_puzzle_problem.get_cost,
        n_puzzle_problem.get_heuristic
    )

    astar_statistics = {'expanded_nodes': 0}
    optimal_cost = len(astar_solver.solve(problem, astar_statistics))
    print('A*')
    print('Optimal Cost:', optimal_cost)
    print('Expanded Nodes:', astar_statistics['expanded_nodes'])
    print()

    anytime_astar_statistics = {'time': [], 'costs': [], 'expanded_nodes': 0, 'max_open_list_size': -1, 'max_closed_set_size': -1}
    anytime_astar_solver.solve(problem, anytime_astar_statistics, weight=5)
    print('Anytime A*')
    print('Costs:', anytime_astar_statistics['costs'])
    print('Expanded Nodes:', anytime_astar_statistics['expanded_nodes'])
    print('Max Open List Size:', anytime_astar_statistics['max_open_list_size'])
    print('Max Closed Set Size:', anytime_astar_statistics['max_closed_set_size'])

    solution_qualities = [optimal_cost / cost for cost in anytime_astar_statistics['costs']]

    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Solution Quality')
    plt.scatter(anytime_astar_statistics['time'], solution_qualities, color='b')
    plt.show()


if __name__ == '__main__':
    main()
