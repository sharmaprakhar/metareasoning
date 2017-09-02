import copy

import numpy as np

import utils

STOP_SYMBOL = 'STOP'
CONTINUE_SYMBOL = 'CONTINUE'


def get_intrinsic_values(qualities, multiplier):
    return np.multiply(multiplier, qualities)


def get_time_costs(steps, multiplier):
    return np.exp(np.multiply(multiplier, steps))


def get_comprehensive_values(instrinsic_values, time_costs):
    return instrinsic_values - time_costs


def get_mevc(estimated_quality, step, profile_1, profile_3, config):
    origin_class = utils.digitize(estimated_quality, config['solution_quality_class_bounds'])

    current_expected_value = 0
    next_expected_value = 0

    for target_class in config['solution_quality_classes']:
        target_quality = utils.get_bin_value(target_class, config['solution_quality_class_count'])
        intrinsic_value = get_intrinsic_values(target_quality, config['intrinsic_value_multiplier'])

        current_time_cost = get_time_costs(step, config['time_cost_multiplier'])
        current_comprehensive_value = get_comprehensive_values(intrinsic_value, current_time_cost)
        current_expected_value += profile_3[origin_class][step][target_class] * current_comprehensive_value

        next_time_cost = get_time_costs(step + 1, config['time_cost_multiplier'])
        next_comprehensive_value = get_comprehensive_values(intrinsic_value, next_time_cost)
        next_expected_value += profile_1[origin_class][step][target_class] * next_comprehensive_value

    return next_expected_value - current_expected_value


def get_optimal_values(profile_2, profile_3, config, epsilon=0.2):
    limit = len(profile_3[0])

    values = {origin_class: limit * [0] for origin_class in config['solution_quality_classes']}

    while True:
        new_values = copy.deepcopy(values)

        delta = 0

        for origin_class in config['solution_quality_classes']:
            for step in range(limit):
                if step + 1 < limit:
                    stop_value = 0
                    continue_value = 0

                    for target_class in config['solution_quality_classes']:
                        target_quality = utils.get_bin_value(target_class, config['solution_quality_class_count'])

                        intrinsic_value = get_intrinsic_values(target_quality, config['intrinsic_value_multiplier'])
                        time_cost = get_time_costs(step, config['time_cost_multiplier'])
                        comprehensive_value = get_comprehensive_values(intrinsic_value, time_cost)
                        stop_value += profile_3[origin_class][step][target_class] * comprehensive_value

                        continue_value += profile_2[origin_class][step][target_class] * values[target_class][step + 1]

                    new_values[origin_class][step] = max(stop_value, continue_value)
                    delta = max(delta, abs(new_values[origin_class][step] - values[origin_class][step]))

        values = new_values
        
        print('Delta: %f' % delta)
        
        if delta < epsilon:
            return values


def get_optimal_action(quality, step, values, profile_2, profile_3, config):
    origin_class = utils.digitize(quality, config['solution_quality_class_bounds'])

    stop_value = 0
    continue_value = 0

    for target_class in config['solution_quality_classes']:
        target_quality = utils.get_bin_value(target_class, config['solution_quality_class_count'])

        intrinsic_value = get_intrinsic_values(target_quality, config['intrinsic_value_multiplier'])
        time_cost = get_time_costs(step, config['time_cost_multiplier'])
        comprehensive_value = get_comprehensive_values(intrinsic_value, time_cost)

        stop_value += profile_3[origin_class][step][target_class] * comprehensive_value
        continue_value += profile_2[origin_class][step][target_class] * values[target_class][step + 1]

    return STOP_SYMBOL if stop_value >= continue_value else CONTINUE_SYMBOL
