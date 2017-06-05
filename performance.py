import numpy as np
import utils

TYPE_1 = lambda qualities, estimated_qualities, step: (estimated_qualities[step], qualities[step + 1])
TYPE_2 = lambda qualities, estimated_qualities, step: (estimated_qualities[step], estimated_qualities[step + 1])
TYPE_3 = lambda qualities, estimated_qualities, step: (estimated_qualities[step], qualities[step])


def get_initial_performance_profile(classes, count, steps):
    return {key: {inner_key: count * [0] for inner_key in steps} for key in classes}


def get_normalized_performance_profile(profile, classes, steps):
    fudge = np.nextafter(0, 1)

    for origin_class in classes:
        for step in steps:
            normalizer = sum(profile[origin_class][step]) + fudge
            profile[origin_class][step] = [probability / normalizer for probability in profile[origin_class][step]]

    return profile


def get_performance_profile(instances, config, selector):
    classes = config['solution_quality_classes']
    bounds = config['solution_quality_class_bounds']
    count = config['solution_quality_class_count']

    groups = utils.get_solution_quality_groups(instances, 'solution_qualities')
    estimated_groups = utils.get_solution_quality_groups(instances, 'estimated_solution_qualities')

    length = utils.get_max_list_length(groups)
    steps = range(length)

    trimmed_groups = utils.get_trimmed_lists(groups, length)
    trimmed_estimated_groups = utils.get_trimmed_lists(estimated_groups, length)

    profile = get_initial_performance_profile(classes, count, steps)

    for i, _ in enumerate(trimmed_groups):
        qualities = trimmed_groups[i]
        estimated_qualities = trimmed_estimated_groups[i]

        for step in steps:
            if step + 1 < length:
                origin_quality, target_quality = selector(qualities, estimated_qualities, step)

                origin_class = utils.digitize(origin_quality, bounds)
                target_class = utils.digitize(target_quality, bounds)

                profile[origin_class][step][target_class] += 1

    return get_normalized_performance_profile(profile, classes, steps)
