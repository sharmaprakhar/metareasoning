import numpy as np
from utils import Problem, get_standard_solution_qualities
import n_puzzle_problem
import astar_solver
import anytime_astar_solver
import matplotlib.pyplot as plt


def display_performance_profile(problem):
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

    solution_qualities = get_standard_solution_qualities(anytime_astar_statistics['costs'], optimal_cost)

    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Solution Quality')
    plt.scatter(anytime_astar_statistics['time'], solution_qualities, color='b')
    plt.show()


def main():
    # puzzle = np.matrix([[5, 8, 7],
    #                     [6, 0, 1],
    #                     [4, 3, 2]])

    # puzzle = np.matrix([[3, 8, 5, 6],
    #                     [2, 0, 9, 11],
    #                     [13, 1, 4, 7],
    #                     [14, 10, 15, 12]])

    puzzle = np.matrix([[7, 11, 9, 8],
                        [6, 1, 2, 3],
                        [0, 10, 5, 12],
                        [4, 13, 14, 15]])

    problem = Problem(
        puzzle,
        n_puzzle_problem.is_goal,
        n_puzzle_problem.get_successors,
        n_puzzle_problem.get_cost,
        n_puzzle_problem.get_heuristic
    )

    display_performance_profile(problem)


if __name__ == '__main__':
    main()
