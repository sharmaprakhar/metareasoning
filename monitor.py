import copy

import numpy as np
from scipy.optimize import curve_fit

import computation

import utils


def get_optimal_stopping_point(comprehensive_values):
    return list(comprehensive_values).index(max(comprehensive_values))


# def get_fixed_stopping_point(intrinsic_values, limit, config):
#     steps = range(len(intrinsic_values))
#     time_costs = computation.get_time_costs(steps, config['time_cost_multiplier'])
#     comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)
#     stopping_point = get_optimal_stopping_point(comprehensive_values)
#     return stopping_point if stopping_point < limit else limit - 1


def get_fixed_stopping_point(steps, profile_4, config):
    best_values = []

    for step in steps:
        expected_value = 0

        for target_class in config['solution_quality_classes']:
            target_quality = utils.get_bin_value(target_class, config['solution_quality_class_count'])

            intrinsic_value = computation.get_intrinsic_values(target_quality, config['intrinsic_value_multiplier'])
            time_cost = computation.get_time_costs(step, config['time_cost_multiplier'])
            comprehensive_value = computation.get_comprehensive_values(intrinsic_value, time_cost)

            expected_value += profile_4[step][target_class] * comprehensive_value

        best_values.append(expected_value)

    return get_optimal_stopping_point(best_values)


def get_nonmyopic_stopping_point(qualities, steps, profile_2, profile_3, limit, config):
    values = computation.get_optimal_values(steps, profile_2, profile_3, config)

    for step in steps:
        if step + 1 == limit:
            return step

        action = computation.get_optimal_action(qualities[step], step, values, profile_2, profile_3, config)

        if action is computation.STOP_SYMBOL:
            return step


def get_myopic_stopping_point(qualities, steps, profile_1, profile_3, limit, config):
    for step in steps:
        if step + 1 == limit:
            return step

        mevc = computation.get_mevc(qualities[step], step, profile_1, profile_3, config)

        if mevc <= 0:
            return step


def get_projected_stopping_point(qualities, steps, limit, config):
    intrinsic_value_groups = []
    stopping_point = 0

    model = lambda x, a, b, c: a * np.arctan(x + b) + c

    for end in range(config['monitor_threshold'], limit):
        try:
            start = 0 if config['window'] is None else end - config['window']

            params, _ = curve_fit(model, steps[start:end], qualities[start:end])
            projections = model(steps, params[0], params[1], params[2])

            intrinsic_values = computation.get_intrinsic_values(projections, config['intrinsic_value_multiplier'])
            time_costs = computation.get_time_costs(steps, config['time_cost_multiplier'])
            comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)
            stopping_point = get_optimal_stopping_point(comprehensive_values)

            intrinsic_value_groups.append(intrinsic_values)

            if stopping_point < end - 1:
                return end - 1, intrinsic_value_groups
        except (RuntimeError, TypeError):
            pass

    return stopping_point, intrinsic_value_groups
