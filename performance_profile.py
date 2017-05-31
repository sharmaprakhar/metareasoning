from __future__ import division

import numpy as np

import utils


def get_estimated_dynamic_performance_profile(solution_quality_map, buckets):
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


def get_estimated_dynamic_performance_map(solution_quality_map, buckets):
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
