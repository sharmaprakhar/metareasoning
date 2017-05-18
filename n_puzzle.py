import random

import numpy as np
from scipy.spatial import distance

from utils import key

# TODO Remove the dependency on size
SIZE = 4
BLANK_SYMBOL = 0
ACTIONS = {
    'North': (-1, 0),
    'East': (0, -1),
    'South': (1, 0),
    'West': (0, 1)
}


def get_initial_puzzle():
    puzzle = np.zeros((SIZE, SIZE))

    for i in range(SIZE):
        for j in range(SIZE):
            puzzle[i, j] = (i * SIZE) + j + 1

    puzzle[SIZE - 1, SIZE - 1] = BLANK_SYMBOL

    return puzzle


def get_next_puzzle(puzzle, action):
    location = get_blank_location(puzzle)

    x, y = location
    next_x, next_y = get_next_blank_location(location, action)

    next_state = np.copy(puzzle)
    next_state[x, y], next_state[next_x, next_y] = next_state[next_x, next_y],  next_state[x, y]

    return next_state


def get_blank_location(puzzle):
    locations = np.where(puzzle == BLANK_SYMBOL)
    x = locations[0][0]
    y = locations[1][0]
    return x, y


def get_next_blank_location(location, action):
    x, y = location
    delta_x, delta_y = ACTIONS[action]
    new_x, new_y = x + delta_x, y + delta_y
    return new_x, new_y


def is_valid_blank_location(location):
    x, y = location
    return SIZE > x >= 0 and SIZE > y >= 0


def generate_random_puzzle(iterations=1000):
    puzzle = get_initial_puzzle()

    for _ in range(iterations):
        location = get_blank_location(puzzle)

        action = random.choice(list(ACTIONS.keys()))
        next_location = get_next_blank_location(location, action)

        while not is_valid_blank_location(next_location):
            action = random.choice(list(ACTIONS.keys()))
            next_location = get_next_blank_location(location, action)

        puzzle = get_next_puzzle(puzzle, action)

    return puzzle


def get_random_puzzle(minimum_difficulty, epsilon=0.1):
    puzzle = get_initial_puzzle()

    difficulty = get_manhattan_distance(puzzle)

    while difficulty < minimum_difficulty:
        action = random.choice(list(ACTIONS.keys()))

        location = get_blank_location(puzzle)
        next_location = get_next_blank_location(location, action)

        if is_valid_blank_location(next_location):
            next_puzzle = get_next_puzzle(puzzle, action)
            next_difficulty = get_manhattan_distance(next_puzzle)

            if next_difficulty > difficulty or random.random() < epsilon:
                puzzle = next_puzzle
                difficulty = next_difficulty

    return puzzle


GOAL_PUZZLE = get_initial_puzzle()
heuristic_cache = {}


def get_manhattan_distance(puzzle):
    puzzle_key = key(puzzle)

    if puzzle_key in heuristic_cache:
        return heuristic_cache[puzzle_key]

    manhattan_distance = 0

    for i in range(SIZE):
        for j in range(SIZE):
            value = GOAL_PUZZLE[i, j]

            if value == BLANK_SYMBOL:
                continue

            locations = np.where(puzzle == value)
            x, y = locations[0][0], locations[1][0]
            manhattan_distance += distance.cityblock((x, y), (i, j))

    heuristic_cache[puzzle_key] = manhattan_distance

    return manhattan_distance
