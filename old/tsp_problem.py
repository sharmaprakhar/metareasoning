import tsp


def is_goal(state, cities):
    return len(state) == len(cities)


def get_successors(state, cities):
    return [{'state': list(state) + [city], 'action': city} for city in cities - set(state)]


def get_cost(state, action, next_state):
    return tsp.get_distance(state[-1], next_state)


def get_heuristic(current_node, cities):
    return tsp.get_mst_distance(current_node.state[-1], cities)
