import numpy as np

import utils


def get_intrinsic_values(solution_qualities, configuration):
    return np.multiply(configuration, solution_qualities)


def get_time_costs(time, multiplier):
    return np.multiply(multiplier, time)


def get_comprehensive_values(instrinsic_value, time_cost):
    return instrinsic_value - time_cost


def get_mevc(estimated_solution_quality, step, performance_profile, performance_map, configuration):
    solution_quality_classes = range(configuration['solution_quality_class_length'])

    current_estimated_solution_quality_class = utils.digitize(estimated_solution_quality, configuration['solution_quality_classes'])

    expected_current_comprehensive_value = 0
    for solution_quality_class in solution_quality_classes:
        current_solution_quality = utils.get_solution_quality(solution_quality_class, configuration['solution_quality_class_length'])
        current_intrinsic_value = get_intrinsic_values(current_solution_quality, configuration['intrinsic_value_multiplier'])
        current_time_cost = get_time_costs(step, configuration['time_cost_multiplier'])
        current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

        expected_current_comprehensive_value += performance_map[current_estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

    expected_next_comprehensive_value = 0
    for solution_quality_class in solution_quality_classes:
        next_solution_quality = utils.get_solution_quality(solution_quality_class, configuration['solution_quality_class_length'])
        next_intrinsic_value = get_intrinsic_values(next_solution_quality, configuration['intrinsic_value_multiplier'])
        next_time_cost = get_time_costs(step + 1, configuration['time_cost_multiplier'])
        next_comprehensive_value = get_comprehensive_values(next_intrinsic_value, next_time_cost)

        expected_next_comprehensive_value += performance_profile[current_estimated_solution_quality_class][step][solution_quality_class] * next_comprehensive_value

    return expected_next_comprehensive_value - expected_current_comprehensive_value


def get_optimal_values(current_solution_quality, step, performance_profile_1, performance_profile_2, configuration):
    solution_quality_classes = range(configuration['solution_quality_class_length'])
    value = 0

    estimated_solution_quality_class = utils.digitize(current_solution_quality, configuration['solution_quality_classes'])

    best_action = ''

    while True:
        delta = 0

        stop_value = 0
        for solution_quality_class in solution_quality_classes:
            current_solution_quality = utils.get_solution_quality(solution_quality_class, configuration['solution_quality_class_length'])
            current_intrinsic_value = get_intrinsic_values(current_solution_quality, configuration['intrinsic_value_multiplier'])
            current_time_cost = get_time_costs(step, configuration['time_cost_multiplier'])
            current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

            stop_value += performance_profile_2[estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

        continue_value = 0
        for solution_quality_class in solution_quality_classes:
            current_solution_quality = utils.get_solution_quality(solution_quality_class, configuration['solution_quality_class_length'])
            current_intrinsic_value = get_intrinsic_values(current_solution_quality, configuration['intrinsic_value_multiplier'])
            current_time_cost = get_time_costs(step, configuration['time_cost_multiplier'])
            current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

            continue_value += performance_profile_1[estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

        new_value = max(stop_value, continue_value)

        delta = max(delta, abs(new_value - value))
        value = new_value

        if stop_value >= continue_value:
            best_action = 'stop'

        if stop_value < continue_value:
            best_action = 'continue'

        if delta < 0.001:
            return best_action
