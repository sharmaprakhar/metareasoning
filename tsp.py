import random
import re
import n_puzzle
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
CITY_PATTERN = '\d+ (\d+.\d+) (\d+.\d+)'


def generate_instance(size, start_position=0, end_position=1, minimum_distance=0.001):
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
        id = i + 1
        x, y = city
        delimiter = '\n' if i < size - 1 else ''

        node_coord_section += CITY_TEMPLATE % (id, x, y, delimiter)

    instance = FILE_TEMPLATE % (name, comment, len(cities), node_coord_section)

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

    return cities


def get_distance(tour):
    distance = 0

    for i in range(len(tour)):
        if i + 1 == len(tour):
            break

        distance += np.linalg.norm(np.subtract(tour[i], tour[i + 1]))

    return distance


def main():
    cities = generate_instance(100, start_position=0, end_position=2000, minimum_distance=1)
    save_instance('instance-4.tsp', 'Comment', cities)


if __name__ == '__main__':
    main()
