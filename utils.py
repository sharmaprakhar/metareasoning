from __future__ import division

import ast
import json

import numpy as np


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


def get_solution_quality_groups(solution_quality_map):
    return [solution_qualities for solution_qualities in solution_quality_map.values()]


def get_intrinsic_value_groups(solution_quality_map, multiplier):
    return [get_intrinsic_values(solution_qualities, multiplier) for solution_qualities in solution_quality_map.values()]


def get_max_length(groups):
    return max(len(group) for group in groups)


def get_trimmed_groups(groups, max_length):
    trimmed_groups = []

    for solution_qualities in groups:
        trimmed_group = list(solution_qualities)

        while len(trimmed_group) < max_length:
            trimmed_group.append(trimmed_group[-1])

        trimmed_groups.append(trimmed_group)

    return trimmed_groups


def get_intrinsic_value_averages(solution_quality_map, multiplier):
    intrinsic_value_groups = get_intrinsic_value_groups(solution_quality_map, multiplier)
    max_length = get_max_length(intrinsic_value_groups)
    trimmed_intrinsic_value_groups = get_trimmed_groups(intrinsic_value_groups, max_length)
    return [sum(intrinsic_values) / len(intrinsic_values) for intrinsic_values in zip(*trimmed_intrinsic_value_groups)]


def get_naive_performance_profile(solution_quality_map, buckets):
    performance_profile = {}

    solution_quality_groups = get_solution_quality_groups(solution_quality_map)
    max_length = get_max_length(solution_quality_groups)
    trimmed_solution_quality_groups = get_trimmed_groups(solution_quality_groups, max_length)
    solution_quality_matrix = np.array(trimmed_solution_quality_groups)

    for i in range(max_length):
        # TODO Replace with norm when you get everything else working
        performance_profile[i] = np.histogram(solution_quality_matrix[:, i], buckets)[0] / len(solution_quality_groups)

    return performance_profile


def get_adjusted_performance_profile(performance_profile, current_solution_quality):
    adjusted_performance_profile = {}

    for key in performance_profile:
        distribution = list(performance_profile[key])

        for i in range(current_solution_quality):
            distribution[i] = 0

        adjusted_performance_profile[key] = [float(i) / (sum(distribution) + 0.0000000001) for i in distribution]

    return adjusted_performance_profile


def get_dynamic_performance_profile(solution_quality_map, buckets):
    class_count = len(buckets) - 1
    performance_profile = {key: class_count * [0] for key in range(class_count)}

    solution_quality_groups = get_solution_quality_groups(solution_quality_map)

    # TODO should this refer to trimmed_solution_quality_groups?
    for solution_qualities in solution_quality_groups:
        time_length = len(solution_qualities)
        for t in range(time_length):
            if t + 1 < time_length:
                solution_quality_start = digitize(solution_qualities[t], buckets)
                solution_quality_target = digitize(solution_qualities[t + 1], buckets)
                performance_profile[solution_quality_start][solution_quality_target] += 1

    for i in range(class_count):
        performance_profile[i] = [float(j) / (sum(performance_profile[i]) + 0.0000000001) for j in performance_profile[i]]

    return performance_profile


def get_solution_quality_map(filename):
    with open(filename) as f:
        return json.load(f)


def get_solution_qualities(line):
    return ast.literal_eval(line)


def get_line_components(line):
    filename, raw_optimal_distance = line.split(',')

    stripped_optimal_distance = raw_optimal_distance.strip()
    truncated_optimal_distance = stripped_optimal_distance.split('.')[0]
    casted_optimal_distance = int(truncated_optimal_distance)

    return filename, casted_optimal_distance


def get_instance_name(filename):
    return filename.split('/')[2]


def get_estimated_intrinsic_value(x, a, b, c):
    return a * np.arctan(x + b) + c


def get_intrinsic_values(solution_qualities, multiplier):
    return np.multiply(multiplier, solution_qualities)


def get_time_costs(time, multiplier):
    return np.multiply(multiplier, time)


def get_comprehensive_values(instrinsic_value, time_cost):
    return instrinsic_value - time_cost


def get_optimal_stopping_point(comprehensive_values):
    return list(comprehensive_values).index(max(comprehensive_values))


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
