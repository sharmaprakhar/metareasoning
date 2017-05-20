import itertools
import random

import tsp


def nearest_neighbor_solve(cities, start_city):
    tour = [start_city]
    current_city = start_city
    remaining_cities = set(cities) - set([start_city])

    while remaining_cities:
        best_neighbor = None
        best_distance = float('inf')

        for neighbor in remaining_cities:
            distance = tsp.get_distance(current_city, neighbor)

            if distance < best_distance:
                best_neighbor = neighbor
                best_distance = distance

        tour += [best_neighbor]
        current_city = best_neighbor
        remaining_cities -= set([best_neighbor])

    tour += [start_city]

    return tour


def random_solve(states, start_state, statistics, iterations, is_detailed=False):
    tour = tsp.get_initial_random_tour(states, start_state)
    cities = tsp.get_swappable_cities(tour)
    distance = tsp.get_tour_distance(tour)
    best_distance = distance

    for time in range(iterations):
        first_key, second_key = random.sample(cities, 2)

        new_tour = tsp.get_mutated_tour(tour, first_key, second_key)
        new_distance = tsp.get_tour_distance(new_tour)

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


def robust_random_solve(states, start_state, statistics, iterations):
    tour = tsp.get_initial_random_tour(states, start_state)
    cities = tsp.get_swappable_cities(tour)
    distance = tsp.get_tour_distance(tour)

    for time in range(iterations):
        first_key = random.choice(cities)

        adjusted_swappable_keys = set(cities) - set([first_key])
        for second_key in adjusted_swappable_keys:
            new_tour = tsp.get_mutated_tour(tour, first_key, second_key)
            new_distance = tsp.get_tour_distance(new_tour)

            if new_distance < distance:
                tour = new_tour
                distance = new_distance

                statistics['time'].append(time)
                statistics['distances'].append(distance)

                break

    return tour


def two_opt_solve(states, start_state, statistics, iterations):
    tour = tsp.get_initial_random_tour(states, start_state)
    cities = tsp.get_swappable_cities(tour)
    distance = tsp.get_tour_distance(tour)

    for time in range(iterations):
        has_changed = False

        for first_key, second_key in itertools.combinations(cities, 2):
            new_tour = tsp.get_mutated_tour(tour, first_key, second_key)
            new_distance = tsp.get_tour_distance(new_tour)

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


def k_opt_solve(states, start_state, statistics, iterations):
    tour = tsp.get_initial_random_tour(states, start_state)
    cities = tsp.get_swappable_cities(tour)
    distance = tsp.get_tour_distance(tour)

    for time in range(iterations):
        has_changed = False

        best_tour = tour
        best_distance = distance
        for first_key, second_key in itertools.combinations(cities, 2):
            current_tour = tsp.get_mutated_tour(tour, first_key, second_key)
            current_distance = tsp.get_tour_distance(current_tour)

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
