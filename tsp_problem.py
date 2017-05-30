import tsp
from utils import pop


def is_goal(state, cities):
    return len(state) == len(cities)


def get_successors(state, cities):
    return [{'state': list(state) + [city], 'action': city} for city in cities - set(state)]


def get_cost(state, action, next_state):
    return tsp.get_distance(state[-1], next_state)


def get_heuristic(current_node, cities):
    if is_goal(current_node.state, cities):
        return 0

    subset = cities

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

        for successor in get_successors(current_city, subset):
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


def get_graph(cities):
    graph = {}

    for start_city in cities:
        graph[start_city] = {}

        for end_city in cities:
            graph[start_city][end_city] = tsp.get_distance(start_city, end_city)

    return graph


def popmin(pqueue):
    lowest = 1000
    keylowest = None
    for key in pqueue:
        if pqueue[key] < lowest:
            lowest = pqueue[key]
            keylowest = key
    del pqueue[keylowest]
    return keylowest


def get_nearest_city_distance(start_city, cities):
    nearest_distance = float('inf')

    for city in cities:
        if start_city == city:
            continue

        current_city_distance = tsp.get_distance(start_city, city)
        if current_city_distance < nearest_distance:
            nearest_distance = current_city_distance

    return nearest_distance


def prim(start_city, cities):
    graph = get_graph(cities)

    pred = {}
    key = {}
    pqueue = {}

    for v in graph:
        pred[v] = -1
        key[v] = 1000

    key[start_city] = 0

    for v in graph:
        pqueue[v] = key[v]

    while pqueue:
        u = popmin(pqueue)

        for v in graph[u]:
            if v in pqueue and graph[u][v] < key[v]:
                pred[v] = u
                key[v] = graph[u][v]
                pqueue[v] = graph[u][v]

    cost = 0
    for first_city in pred.keys():
        second_city = pred[first_city]
        if second_city != -1:
            cost += tsp.get_distance(first_city, second_city)

    print start_city
    print cities
    return cost + 2 * get_nearest_city_distance(start_city, cities)

