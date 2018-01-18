import json

import numpy as np

import computation

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


def get_transformed_instances(instances, f):
    transformed_instances = instances
    for instance in instances:
        transformed_instances[instance]['qualities'] = [f(q) for q in instances[instance]['qualities']]
        transformed_instances[instance]['estimated_qualities'] = [f(q) for q in instances[instance]['estimated_qualities']]
    return transformed_instances