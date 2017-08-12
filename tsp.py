import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

import utils

FILE_TEMPLATE = '''NAME : %s
COMMENT : %s
TYPE : TSP
DIMENSION: %d
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
%s
EOF
'''
CITY_TEMPLATE = '%d %i %i %s'
COMMENT = 'No Comment'
CITY_PATTERN = '\d+ (\d+) (\d+)'
DELIMITER = '\n'


def get_initial_random_tour(states, start_state):
    adjustable_states = set(states) - set([start_state])
    return [start_state] + list(np.random.permutation(list(adjustable_states))) + [start_state]


def get_swappable_cities(tour):
    return range(1, len(tour) - 1)


def get_mutated_tour(tour, first_key, second_key):
    mutated_tour = list(tour)
    mutated_tour[first_key], mutated_tour[second_key] = mutated_tour[second_key], mutated_tour[first_key]
    return mutated_tour


def get_distance(first_city, second_city):
    return np.linalg.norm(np.subtract(first_city, second_city))


def get_tour_distance(tour):
    distance = 0

    for i in range(len(tour)):
        if i + 1 == len(tour):
            break

        distance += get_distance(tour[i], tour[i + 1])

    return distance


def get_graph(cities):
    graph = {}

    for start_city in cities:
        graph[start_city] = {}

        for end_city in cities:
            graph[start_city][end_city] = get_distance(start_city, end_city)

    return graph


def get_nearest_city_distance(start_city, cities):
    nearest_distance = float('inf')

    for city in cities:
        if start_city == city:
            continue

        current_city_distance = get_distance(start_city, city)
        if current_city_distance < nearest_distance:
            nearest_distance = current_city_distance

    return nearest_distance


def get_mst_distance(start_city, cities):
    subset = cities - set([start_city])
    graph = get_graph(subset)

    predecessors = {}
    key = {}
    queue = {}

    for vertex in graph:
        predecessors[vertex] = -1
        key[vertex] = sys.maxsize

    key[start_city] = 0

    for vertex in graph:
        queue[vertex] = key[vertex]

    while queue:
        city = utils.pop(queue)

        for vertex in graph[city]:
            if vertex in queue and graph[city][vertex] < key[vertex]:
                predecessors[vertex] = city
                key[vertex] = graph[city][vertex]
                queue[vertex] = graph[city][vertex]

    cost = 0
    for parent_city in predecessors:
        child_city = predecessors[parent_city]
        if child_city != -1:
            cost += get_distance(parent_city, child_city)

    return cost + 2 * get_nearest_city_distance(start_city, cities)


def get_instance(size, start_position, end_position, minimum_distance):
    choices = np.arange(start_position, end_position, minimum_distance)

    cities = set()
    while len(cities) < size:
        x = round(random.choice(choices), 3)
        y = round(random.choice(choices), 3)
        cities.add((x, y))

    return cities


def get_clustered_instance(size, start_position, end_position, minimum_step, centroid_count, centroid_radius, centroid_minimum_step):
    choices = np.arange(start_position, end_position, minimum_step)
    centroids = []
    normalizer = 0

    for _ in range(centroid_count):
        threshold = random.random()
        normalizer += threshold

        centroids.append({
            'weight': threshold,
            'x': round(random.choice(choices), 3),
            'y': round(random.choice(choices), 3)
        })

    for centroid in centroids:
        centroid['weight'] /= normalizer

    choices = np.arange(-centroid_radius, centroid_radius, centroid_minimum_step)
    cities = set()

    while len(cities) < size:
        threshold = random.random()

        selected_centroid = None
        for centroid in centroids:
            threshold -= centroid['weight']
            selected_centroid = centroid

            if threshold <= 0:
                break

        new_x = round(selected_centroid['x'] + random.choice(choices), 3)
        new_y = round(selected_centroid['y'] + random.choice(choices), 3)
        city = (new_x, new_y)

        cities.add(city)

    return cities


def save_instance(name, comment, cities):
    size = len(cities)

    node_coord_section = ''
    for i, city in enumerate(cities):
        x, y = city
        delimiter = DELIMITER if i < size - 1 else ''
        node_coord_section += CITY_TEMPLATE % (i + 1, x, y, delimiter)

    instance = FILE_TEMPLATE % (name, comment, size, node_coord_section)

    f = open(name, 'w')
    f.write(instance)
    f.close()


def load_instance(filename):
    cities = set()

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(CITY_PATTERN, line)
            if match:
                x = float(match.groups()[0])
                y = float(match.groups()[1])
                cities.add((x, y))

    start_city = list(cities)[0]

    return cities, start_city
