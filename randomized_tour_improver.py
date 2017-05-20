import itertools
import random

import numpy as np

import tsp


def get_initial_random_tour(states, start_state):
    adjustable_states = set(states) - set([start_state])
    return [start_state] + list(np.random.permutation(list(adjustable_states))) + [start_state]


def get_nearest_neighbor_tour(cities, start_city):
    remaining_cities = set(cities) - set([start_city])

    tour = [start_city]
    current_city = start_city

    while remaining_cities:
        best_neighbor = None
        best_distance = float('inf')

        for neighbor in remaining_cities:
            distance = np.linalg.norm(np.subtract(current_city, neighbor))

            if distance < best_distance:
                best_neighbor = neighbor
                best_distance = distance

        tour += [best_neighbor]
        remaining_cities -= set([best_neighbor])
        current_city = best_neighbor

    tour += [start_city]

    return tour


def get_cities(tour):
    tour_length = len(tour)
    return range(tour_length)


def get_mutated_tour(tour, first_key, second_key):
    mutated_tour = list(tour)
    mutated_tour[first_key], mutated_tour[second_key] = mutated_tour[second_key], mutated_tour[first_key]
    return mutated_tour


def random_solve(states, start_state, statistics, iterations=1000, is_detailed=False):
    tour = get_nearest_neighbor_tour(states, start_state)
    distance = tsp.get_distance(tour)

    best_distance = distance
    cities = range(1, len(tour) - 1)

    for time in range(iterations):
        first_key, second_key = random.sample(cities, 2)

        new_tour = get_mutated_tour(tour, first_key, second_key)
        new_distance = tsp.get_distance(new_tour)

        if new_distance < best_distance:
            best_distance = new_distance

        if new_distance < distance:
            tour = new_tour
            distance = new_distance

            if not is_detailed:
                statistics['time'].append(time)
                statistics['distances'].append(best_distance)

        if is_detailed:
            statistics['time'].append(time)
            statistics['distances'].append(best_distance)

    return tour


def robust_random_solve(states, start_state, statistics, iterations=1000):
    tour = get_initial_random_tour(states, start_state)
    distance = tsp.get_distance(tour)

    cities = range(1, len(tour) - 1)

    for time in range(iterations):
        first_key = random.choice(cities)

        adjusted_swappable_keys = set(cities) - set([first_key])
        for second_key in adjusted_swappable_keys:
            new_tour = get_mutated_tour(tour, first_key, second_key)
            new_distance = tsp.get_distance(new_tour)

            if new_distance < distance:
                tour = new_tour
                distance = new_distance

                statistics['time'].append(time)
                statistics['distances'].append(distance)

                break

    return tour


def two_opt_solve(states, start_state, statistics, iterations=500):
    tour = get_initial_random_tour(states, start_state)
    distance = tsp.get_distance(tour)

    cities = range(1, len(tour) - 1)

    for time in range(iterations):
        has_changed = False

        for first_key, second_key in itertools.combinations(cities, 2):
            new_tour = get_mutated_tour(tour, first_key, second_key)
            new_distance = tsp.get_distance(new_tour)

            if new_distance < distance:
                tour = new_tour
                distance = new_distance

                statistics['time'].append(time)
                statistics['distances'].append(distance)

                has_changed = True

                break

        if not has_changed:
            break

    return tour


def k_opt_solve(states, start_state, statistics, iterations=100):
    tour = get_initial_random_tour(states, start_state)
    distance = tsp.get_distance(tour)

    cities = range(1, len(tour) - 1)

    for time in range(iterations):
        has_changed = False

        best_tour = tour
        best_distance = distance
        for first_key, second_key in itertools.combinations(cities, 2):
            current_tour = get_mutated_tour(tour, first_key, second_key)
            current_distance = tsp.get_distance(current_tour)

            if current_distance < best_distance:
                best_tour = current_tour
                best_distance = current_distance

                has_changed = True

        tour = best_tour
        distance = best_distance

        statistics['time'].append(time)
        statistics['distances'].append(distance)

        if not has_changed:
            break

    return tour
