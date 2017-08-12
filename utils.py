import json

import numpy as np

import computation


class Problem(object):
    def __init__(self, start_state, is_goal, get_successors, get_cost, get_heuristic):
        self.start_state = start_state
        self.is_goal = is_goal
        self.get_successors = get_successors
        self.get_cost = get_cost
        self.get_heuristic = get_heuristic


class Node(object):
    def __init__(self, state, parent=None, path_cost=0, depth=0, action=None):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.depth = depth
        self.action = action


class OpenList(object):
    def __init__(self):
        self.items = []
        self.cache = {}

    def add(self, node, value):
        self.items.append((value, node))
        self.items.sort(key=lambda item: item[0])

        node_key = get_key(node.state)
        self.cache[node_key] = node

    def remove(self):
        return self.items.pop(0)[1]

    def __len__(self):
        return len(self.items)

    def __contains__(self, node):
        node_key = get_key(node.state)
        return node_key in self.cache

    def __getitem__(self, node):
        node_key = get_key(node.state)
        return self.cache[node_key]

    def __delitem__(self, new_node):
        new_node_key = get_key(new_node.state)

        for i, (_, node) in enumerate(self.items):
            node_key = get_key(node.state)

            if node_key == new_node_key:
                self.items.pop(i)

        del self.cache[new_node_key]


def get_children_nodes(problem, parent):
    children_nodes = []

    for successor in problem.get_successors(parent.state):
        path_cost = parent.path_cost + problem.get_cost(parent.state, successor['action'], successor['state'])
        child_node = Node(successor['state'], parent, path_cost, parent.depth + 1, successor['action'])
        children_nodes.append(child_node)

    return children_nodes


def get_solution(node):
    if node.parent is None:
        return []
    return get_solution(node.parent) + [node.action]


def get_key(state):
    return str(state.tolist())


def pop(queue):
    minimum_value = float('inf')
    minimum_key = None

    for key in queue:
        if queue[key] < minimum_value:
            minimum_value = queue[key]
            minimum_key = key

    del queue[minimum_key]

    return minimum_key


def get_max_list_length(lists):
    return max(len(inner_list) for inner_list in lists)


def get_trimmed_lists(groups, max_length):
    trimmed_groups = []

    for solution_qualities in groups:
        trimmed_group = list(solution_qualities)

        while len(trimmed_group) < max_length:
            trimmed_group.append(trimmed_group[-1])

        trimmed_groups.append(trimmed_group)

    return trimmed_groups


def get_groups(instances, key):
    return [instance[key] for instance in instances.values()]


def get_intrinsic_value_groups(instances, multiplier, key):
    return [computation.get_intrinsic_values(instance[key], multiplier) for instance in instances.values()]


def get_instances(filename):
    with open(filename) as file:
        return json.load(file)


def get_column(lists, index):
    array = np.array(lists)
    return array[:, index]


def digitize(item, bins):
    for i, _ in enumerate(bins):
        if i + 1 < len(bins):
            if bins[i] <= item < bins[i + 1]:
                return i
    return len(bins) - 1


def get_bin_value(bin, bin_size):
    length = 1 / bin_size
    offset = length / 2
    return (bin / bin_size) + offset


def get_line_components(line):
    filename, raw_optimal_distance = line.split(',')

    stripped_optimal_distance = raw_optimal_distance.strip()
    truncated_optimal_distance = stripped_optimal_distance.split('.')[0]
    casted_optimal_distance = int(truncated_optimal_distance)

    return filename, casted_optimal_distance


def get_percent_error(accepted_value, approximate_value):
    return np.absolute(accepted_value - approximate_value) / accepted_value * 100


def get_average_intrinsic_values(instances, multiplier):
    intrinsic_value_groups = get_intrinsic_value_groups(instances, multiplier, 'qualities')
    max_length = get_max_list_length(intrinsic_value_groups)
    trimmed_intrinsic_value_groups = get_trimmed_lists(intrinsic_value_groups, max_length)
    return [sum(intrinsic_values) / len(intrinsic_values) for intrinsic_values in zip(*trimmed_intrinsic_value_groups)]
