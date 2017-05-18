import random

import numpy as np

import anytime_astar_solver
import astar_solver
import tsp
from utils import Problem


def pop(queue):
    minimum_value = float('inf')
    minimum_key = None

    for key in queue:
        if queue[key] < minimum_value:
            minimum_value = queue[key]
            minimum_key = key

    del queue[minimum_key]

    return minimum_key


def get_heuristic(current_node, start_city, cities):
    if is_goal(current_node.state, cities):
        return 0

    # TODO: Add the start city
    subset = cities - set(current_node.state)

    predecessors = {}
    key = {}
    queue = {}

    for state in subset:
        predecessors[state] = -1
        key[state] = float('inf')
        queue[state] = float('inf')

    current_city = current_node.state[-1]
    queue[current_city] = 0

    while queue:
        current_city = pop(queue)

        for successor in get_successors(current_node.state, subset):
            next_city = successor['state'][-1]

            # TODO: Fix this
            cost = np.linalg.norm(np.subtract(current_city, next_city))

            if next_city in queue and cost < key[next_city]:
                predecessors[next_city] = current_city
                key[next_city] = cost
                queue[next_city] = cost

    cost = 0
    for parent_city, child_city in predecessors.iteritems():
        if child_city != -1:
            # TODO: Fix this
            cost += np.linalg.norm(np.subtract(parent_city, child_city))

    return cost


def is_goal(state, cities):
    return len(cities) == len(state)


def get_successors(state, cities):
    return [{'action': city, 'state': list(state) + [city]} for city in cities - set(state)]


def get_cost(state, action, next_state):
    return np.linalg.norm(np.subtract(state[-1], next_state))


def get_tour_distance(start_city, actions):
    distance = get_cost(start_city, actions[0], actions[0])

    for i in range(len(actions)):
        if i + 1 == len(actions):
            break

        distance += get_cost(actions[i], actions[i + 1], actions[i + 1])

    return distance


def main():
    cities = tsp.generate_instance(8)
    start_city = random.choice(list(cities))

    print 'Cities:', cities
    print 'Start City:', start_city

    print 'Optimal Solution:', astar_solver.solve(Problem(
        [start_city],
        lambda state: is_goal(state, cities),
        lambda state: get_successors(state, cities),
        get_cost,
        lambda state: get_heuristic(state, start_city, cities)
    ))

    print 'Generating solutions...'
    anytime_astar_solver.solve(Problem(
        [start_city],
        lambda state: is_goal(state, cities),
        lambda state: get_successors(state, cities),
        get_cost,
        lambda state: get_heuristic(state, start_city, cities)
    ))


if __name__ == '__main__':
    main()
