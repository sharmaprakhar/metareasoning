import numpy as np

import n_puzzle


def is_goal(state):
    return np.array_equal(state, n_puzzle.get_initial_puzzle())


def get_successors(state):
    successors = []

    for action in n_puzzle.ACTIONS:
        location = n_puzzle.get_blank_location(state)
        next_location = n_puzzle.get_next_blank_location(location, action)

        if n_puzzle.is_valid_blank_location(next_location):
            neighbor = n_puzzle.get_next_puzzle(state, action)
            successors.append({'action': action, 'state': neighbor})

    return successors


def get_cost(state, action, next_state):
    return 1


def get_heuristic(state):
    return n_puzzle.get_manhattan_distance(state)
