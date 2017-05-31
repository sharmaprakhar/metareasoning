from __future__ import division

import numpy as np

import utils


def get_probabilistic_performance_profile(solution_quality_lists, buckets):
    normalizer = len(solution_quality_lists)
    time_limit = utils.get_max_list_length(solution_quality_lists)
    trimmed_lists = utils.get_trimmed_lists(solution_quality_lists, time_limit)
    return {t: np.histogram(utils.get_column(trimmed_lists, t), buckets)[0] / normalizer for t in range(time_limit)}


def get_normalized_probabilistic_performance_profile(performance_profile, solution_quality_class):
    normalized_performance_profile = {}

    for t in performance_profile:
        distribution = list(performance_profile[t])

        for i in range(solution_quality_class):
            distribution[i] = 0

        fudge = np.nextafter(0, 1)
        normalizer = sum(distribution) + fudge

        normalized_performance_profile[t] = [float(i) / normalizer for i in distribution]

    return normalized_performance_profile


def get_static_probabilistic_performance_profile(solution_quality_map, buckets):
    solution_quality_groups = utils.get_solution_quality_groups(solution_quality_map, 'estimated_solution_qualities')
    class_count = len(buckets) - 1

    performance_profile = {key: class_count * [0] for key in range(class_count)}

    for solution_qualities in solution_quality_groups:
        time_length = len(solution_qualities)
        for t in range(time_length):
            if t + 1 < time_length:
                solution_quality_start = utils.digitize(solution_qualities[t], buckets)
                solution_quality_target = utils.digitize(solution_qualities[t + 1], buckets)
                performance_profile[solution_quality_start][solution_quality_target] += 1

    fudge = np.nextafter(0, 1)
    for i in range(class_count):
        length = sum(performance_profile[i]) + fudge
        performance_profile[i] = [float(j) / length for j in performance_profile[i]]

    return performance_profile


def get_dynamic_performance_profile(solution_quality_map, buckets):
    solution_quality_groups = utils.get_solution_quality_groups(solution_quality_map, 'estimated_solution_qualities')
    max_length = utils.get_max_length(solution_quality_groups)
    trimmed_solution_quality_groups = utils.get_trimmed_groups(solution_quality_groups, max_length)

    class_count = len(buckets) - 1

    performance_profile = {key: {inner_key: class_count * [0] for inner_key in range(max_length)} for key in range(class_count)}

    for solution_qualities in trimmed_solution_quality_groups:
        for t in range(max_length):
            if t + 1 < max_length:
                solution_quality_start = utils.digitize(solution_qualities[t], buckets)
                solution_quality_target = utils.digitize(solution_qualities[t + 1], buckets)

                performance_profile[solution_quality_start][t][solution_quality_target] += 1

    fudge = np.nextafter(0, 1)
    for i in range(class_count):
        for j in range(max_length):
            length = sum(performance_profile[i][j]) + fudge
            performance_profile[i][j] = [float(k) / length for k in performance_profile[i][j]]

    return performance_profile


def get_dynamic_estimated_performance_profile(solution_quality_map, buckets):
    solution_quality_groups = utils.get_solution_quality_groups(solution_quality_map, 'solution_qualities')
    estimated_solution_quality_groups = utils.get_solution_quality_groups(solution_quality_map, 'estimated_solution_qualities')

    max_length = utils.get_max_list_length(solution_quality_groups)

    trimmed_solution_quality_groups = utils.get_trimmed_lists(solution_quality_groups, max_length)
    trimmed_estimated_solution_quality_groups = utils.get_trimmed_lists(estimated_solution_quality_groups, max_length)

    class_count = len(buckets) - 1

    performance_profile = {key: {inner_key: class_count * [0] for inner_key in range(max_length)} for key in range(class_count)}

    for i in range(len(trimmed_solution_quality_groups)):
        solution_qualities = trimmed_solution_quality_groups[i]
        estimated_solution_qualities = trimmed_estimated_solution_quality_groups[i]

        for t in range(max_length):
            if t + 1 < max_length:
                estimated_solution_quality_start = utils.digitize(estimated_solution_qualities[t], buckets)
                solution_quality_target = utils.digitize(solution_qualities[t + 1], buckets)

                performance_profile[estimated_solution_quality_start][t][solution_quality_target] += 1

    fudge = np.nextafter(0, 1)
    for i in range(class_count):
        for j in range(max_length):
            length = sum(performance_profile[i][j]) + fudge
            performance_profile[i][j] = [float(k) / length for k in performance_profile[i][j]]

    return performance_profile


def get_dynamic_estimation_map(solution_quality_map, buckets):
    solution_quality_groups = utils.get_solution_quality_groups(solution_quality_map, 'solution_qualities')
    estimated_solution_quality_groups = utils.get_solution_quality_groups(solution_quality_map, 'estimated_solution_qualities')

    max_length = utils.get_max_list_length(solution_quality_groups)

    trimmed_solution_quality_groups = utils.get_trimmed_lists(solution_quality_groups, max_length)
    trimmed_estimated_solution_quality_groups = utils.get_trimmed_lists(estimated_solution_quality_groups, max_length)

    class_count = len(buckets) - 1

    performance_profile = {key: {inner_key: class_count * [0] for inner_key in range(max_length)} for key in range(class_count)}

    for i in range(len(trimmed_solution_quality_groups)):
        solution_qualities = trimmed_solution_quality_groups[i]
        estimated_solution_qualities = trimmed_estimated_solution_quality_groups[i]

        for t in range(max_length):
            estimated_solution_quality_start = utils.digitize(estimated_solution_qualities[t], buckets)
            solution_quality_target = utils.digitize(solution_qualities[t], buckets)

            performance_profile[estimated_solution_quality_start][t][solution_quality_target] += 1

    fudge = np.nextafter(0, 1)
    for i in range(class_count):
        for j in range(max_length):
            length = sum(performance_profile[i][j]) + fudge
            performance_profile[i][j] = [float(k) / length for k in performance_profile[i][j]]

    return performance_profile
