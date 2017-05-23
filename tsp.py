import random
import re

import numpy as np

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

SIZE = 50


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


def get_instance(size, start_position=0, end_position=1, minimum_distance=0.001):
    choices = np.arange(start_position, end_position, minimum_distance)

    cities = set()
    while len(cities) < size:
        x = round(random.choice(choices), 3)
        y = round(random.choice(choices), 3)
        cities.add((x, y))

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


def generate_instance_file(size, name):
    cities = get_instance(size, start_position=0, end_position=2000, minimum_distance=1)
    save_instance(name, COMMENT, cities)


def main():
    for i in range(SIZE):
        generate_instance_file(SIZE, 'instances/%d-tsp/instance-%d.tsp' % (SIZE, i))


if __name__ == '__main__':
    main()
