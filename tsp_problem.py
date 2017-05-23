import tsp
from utils import pop


def is_goal(state, cities):
    return len(state) == len(cities)


def get_successors(state, cities):
    return [{'state': list(state) + [city], 'action': city} for city in cities - set(state)]


def get_cost(state, action, next_state):
    return tsp.get_distance(state[-1], next_state)


def get_heuristic(current_node, start_city, cities):
    if is_goal(current_node.state, cities):
        return 0

    # TODO Should we add the start city to this?
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

            cost = tsp.get_distance(current_city, next_city)

            if next_city in queue and cost < key[next_city]:
                predecessors[next_city] = current_city
                key[next_city] = cost
                queue[next_city] = cost

    cost = 0
    for parent_city, child_city in predecessors.iteritems():
        if child_city != -1:
            cost += tsp.get_distance(parent_city, child_city)

    return cost
