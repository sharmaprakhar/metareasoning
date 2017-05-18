import random

import numpy as np
import old.tsp


def get_initial_tour(states, start_state):
    adjustable_states = set(states) - set([start_state])
    return [start_state] + list(np.random.permutation(list(adjustable_states))) + [start_state]


def get_cities(tour):
    tour_length = len(tour)
    return range(tour_length - 2)


def get_mutated_tour(tour, first_key, second_key):
    mutated_tour = list(tour)
    mutated_tour[first_key], mutated_tour[second_key] = mutated_tour[second_key], mutated_tour[first_key]
    return mutated_tour


def naive_solve(states, start_state, statistics, iterations, is_detailed=False, epsilon=0.05):
    tour = get_initial_tour(states, start_state)
    distance = old.tsp.get_distance(tour)

    best_distance = distance

    cities = get_cities(tour)

    for time in range(iterations):
        first_key, second_key = random.sample(cities, 2)

        new_tour = get_mutated_tour(tour, first_key, second_key)
        new_distance = old.tsp.get_distance(new_tour)

        if new_distance < best_distance:
            best_distance = new_distance

        if new_distance < distance or random.random() < epsilon:
            tour = new_tour
            distance = new_distance

            if not is_detailed:
                statistics['time'].add(time)
                statistics['distances'].add(best_distance)

        if is_detailed:
            statistics['time'].add(time)
            statistics['distances'].add(best_distance)

    return tour


def solve(states, start_state, statistics, iterations=1000):
    tour = get_initial_tour(states, start_state)
    distance = old.tsp.get_distance(tour)

    cities = get_cities(tour)

    for time in range(iterations):
        first_key = random.choice(cities)

        adjusted_swappable_keys = set(cities) - set([first_key])
        for second_key in adjusted_swappable_keys:
            new_tour = get_mutated_tour(tour, first_key, second_key)
            new_distance = old.tsp.get_distance(new_tour)

            if new_distance < distance:
                tour = new_tour
                distance = new_distance

                statistics['time'].add(time)
                statistics['distances'].add(distance)

                break

    return tour
