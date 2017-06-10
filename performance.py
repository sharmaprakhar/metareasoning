import numpy as np

import utils

TYPE_1 = lambda qualities, estimated_qualities, step: (estimated_qualities[step], qualities[step + 1])
TYPE_2 = lambda qualities, estimated_qualities, step: (estimated_qualities[step], estimated_qualities[step + 1])
TYPE_3 = lambda qualities, estimated_qualities, step: (estimated_qualities[step], qualities[step])


def get_standard_solution_qualities(costs, optimal_cost):
    return [1 - ((cost - optimal_cost) / optimal_cost) for cost in costs]


def get_naive_solution_qualities(costs, optimal_cost):
    return [optimal_cost / cost for cost in costs]


def get_initial_probabilistic_performance_profile(count, steps):
    return {step: count * [0] for step in steps}


def get_probabilistic_performance_profile(instances, config):
    groups = utils.get_groups(instances, 'solution_qualities')

    length = utils.get_max_list_length(groups)
    steps = range(length)

    trimmed_groups = utils.get_trimmed_lists(groups, length)

    profile = get_initial_probabilistic_performance_profile(config['solution_quality_class_count'], steps)

    for step in steps:
        for qualities in trimmed_groups:
            target_quality = qualities[step]

            # TODO Figure out why this is happening
            # target_quality = 0.999999 if target_quality > 1 else target_quality

            target_class = utils.digitize(target_quality, config['solution_quality_class_bounds'])
            profile[step][target_class] += 1

        normalizer = sum(profile[step])
        for target_class in config['solution_quality_classes']:
            profile[step][target_class] /= normalizer

    return profile


def get_initial_dynamic_performance_profile(classes, count, steps):
    return {origin_class: {step: count * [0] for step in steps} for origin_class in classes}


def get_normalized_performance_profile(profile, classes, steps):
    fudge = np.nextafter(0, 1)

    for origin_class in classes:
        for step in steps:
            normalizer = sum(profile[origin_class][step]) + fudge
            profile[origin_class][step] = [probability / normalizer for probability in profile[origin_class][step]]

    return profile


def get_dynamic_performance_profile(instances, config, selector):
    classes = config['solution_quality_classes']
    bounds = config['solution_quality_class_bounds']
    count = config['solution_quality_class_count']

    groups = utils.get_groups(instances, 'solution_qualities')
    estimated_groups = utils.get_groups(instances, 'estimated_solution_qualities')

    length = utils.get_max_list_length(groups)
    steps = range(length)

    trimmed_groups = utils.get_trimmed_lists(groups, length)
    trimmed_estimated_groups = utils.get_trimmed_lists(estimated_groups, length)

    profile = get_initial_dynamic_performance_profile(classes, count, steps)

    for i, _ in enumerate(trimmed_groups):
        qualities = trimmed_groups[i]
        estimated_qualities = trimmed_estimated_groups[i]

        for step in steps:
            if step + 1 < length:
                origin_quality, target_quality = selector(qualities, estimated_qualities, step)

                # # TODO Figure out why this is happening
                # origin_quality = 0.999999 if origin_quality > 1 else origin_quality
                # target_quality = 0.999999 if target_quality > 1 else target_quality

                origin_class = utils.digitize(origin_quality, bounds)
                target_class = utils.digitize(target_quality, bounds)

                profile[origin_class][step][target_class] += 1


    return get_normalized_performance_profile(profile, classes, steps)
