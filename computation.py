import numpy as np

import utils

STOP_SYMBOL = 'STOP'
CONTINUE_SYMBOL = 'CONTINUE'


def get_intrinsic_values(solution_qualities, configuration):
    return np.multiply(configuration, solution_qualities)


def get_time_costs(time, multiplier):
    return np.exp(np.multiply(multiplier, time))


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


def get_optimal_action(current_solution_quality, step, performance_profile, performance_map, configuration):
    solution_quality_classes = range(configuration['solution_quality_class_length'])
    estimated_solution_quality_class = utils.digitize(current_solution_quality, configuration['solution_quality_classes'])

    value = 0

    while True:
        delta = 0

        stop_value = 0
        for solution_quality_class in solution_quality_classes:
            current_solution_quality = utils.get_solution_quality(solution_quality_class, configuration['solution_quality_class_length'])
            current_intrinsic_value = get_intrinsic_values(current_solution_quality, configuration['intrinsic_value_multiplier'])
            current_time_cost = get_time_costs(step, configuration['time_cost_multiplier'])
            current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

            stop_value += performance_map[estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

        continue_value = 0
        for solution_quality_class in solution_quality_classes:
            current_solution_quality = utils.get_solution_quality(solution_quality_class, configuration['solution_quality_class_length'])
            current_intrinsic_value = get_intrinsic_values(current_solution_quality, configuration['intrinsic_value_multiplier'])
            current_time_cost = get_time_costs(step, configuration['time_cost_multiplier'])
            current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

            continue_value += performance_profile[estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

        new_value = max(stop_value, continue_value)
        delta = max(delta, abs(new_value - value))
        value = new_value

        if delta < 0.001:
            return STOP_SYMBOL if stop_value >= continue_value else CONTINUE_SYMBOL
