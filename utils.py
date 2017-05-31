from __future__ import division

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


def get_standard_solution_qualities(costs, optimal_cost):
    return [1 - ((cost - optimal_cost) / optimal_cost) for cost in costs]


def get_naive_solution_qualities(costs, optimal_cost):
    return [optimal_cost / cost for cost in costs]


def get_solution_quality_groups(solution_quality_map, key):
    return [solution_qualities[key] for solution_qualities in solution_quality_map.values()]


def get_intrinsic_value_groups(solution_quality_map, multiplier, key):
    return [computation.get_intrinsic_values(solution_qualities[key], multiplier) for solution_qualities in solution_quality_map.values()]


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


def get_average_intrinsic_values(solution_quality_map, multiplier):
    intrinsic_value_groups = get_intrinsic_value_groups(solution_quality_map, multiplier, 'solution_qualities')
    max_length = get_max_list_length(intrinsic_value_groups)
    trimmed_intrinsic_value_groups = get_trimmed_lists(intrinsic_value_groups, max_length)
    return [sum(intrinsic_values) / len(intrinsic_values) for intrinsic_values in zip(*trimmed_intrinsic_value_groups)]


def get_instance_map(filename):
    with open(filename) as f:
        return json.load(f)


def get_line_components(line):
    filename, raw_optimal_distance = line.split(',')

    stripped_optimal_distance = raw_optimal_distance.strip()
    truncated_optimal_distance = stripped_optimal_distance.split('.')[0]
    casted_optimal_distance = int(truncated_optimal_distance)

    return filename, casted_optimal_distance


def get_projected_solution_qualities(x, a, b, c):
    return a * np.arctan(x + b) + c


def digitize(solution_quality, buckets):
    bucket_id = len(buckets) - 1

    for i in range(len(buckets)):
        if i + 1 == len(buckets):
            break

        range_start = buckets[i]
        range_end = buckets[i + 1]

        if range_start <= solution_quality < range_end:
            bucket_id = i
            break

    return bucket_id


def get_solution_quality(solution_quality_class, bucket_size):
    length = 1 / bucket_size
    offset = length / 2
    return (solution_quality_class / bucket_size) + offset


def get_optimal_fixed_allocation_time(performance_profile, bucket_size, intrinsic_value_multiplier, time_cost_multiplier):
    best_step = None
    best_value = float('-inf')

    time_limit = len(performance_profile.keys())
    steps = range(time_limit)
    solution_quality_classes = range(bucket_size)

    for step in steps:
        estimated_value = 0

        for solution_quality_class in solution_quality_classes:
            adjusted_performance_profile = performance_profile.get_adjusted_performance_profile(performance_profile, solution_quality_class)

            probability = adjusted_performance_profile[step][solution_quality_class]

            solution_quality = get_solution_quality(solution_quality_class, bucket_size)
            intrinsic_value = computation.get_intrinsic_values(solution_quality, intrinsic_value_multiplier)
            time_cost = computation.get_time_costs(step, time_cost_multiplier)
            comprehensive_value = computation.get_comprehensive_values(intrinsic_value, time_cost)

            estimated_value += probability * comprehensive_value

        if estimated_value > best_value:
            best_step = step
            best_value = estimated_value

    return best_step


def get_column(lists, index):
    array = np.array(lists)
    return array[:, index]


def get_percent_error(true_value, approximate_value):
    return np.absolute(true_value - approximate_value) / true_value * 100
