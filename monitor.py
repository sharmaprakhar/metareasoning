from scipy.optimize import curve_fit

import computation
import utils


def get_optimal_stopping_point(comprehensive_values):
    return list(comprehensive_values).index(max(comprehensive_values))


def get_fixed_stopping_point(average_intrinsic_values, time_limit, configuration):
    steps = range(len(average_intrinsic_values))
    time_costs = computation.get_time_costs(steps, configuration['time_cost_multiplier'])
    average_comprehensive_values = average_intrinsic_values - time_costs
    fixed_stopping_point = get_optimal_stopping_point(average_comprehensive_values)
    return fixed_stopping_point if fixed_stopping_point < time_limit else time_limit - 1


def get_nonmyopic_stopping_point(solution_qualities, steps, performance_profile, performance_map, time_limit, configuration):
    values = computation.get_optimal_values(steps, performance_profile, performance_map, configuration)
    
    for step in steps:
        if step + 1 == time_limit:
            return step
        
        action = computation.get_optimal_action(solution_qualities[step], step, values, performance_profile, performance_map, configuration)

        if action is computation.STOP_SYMBOL:
            return step


def get_myopic_stopping_point(solution_qualities, steps, performance_profile, performance_map, time_limit, configuration):
    for step in steps:
        if step + 1 == time_limit:
            return step

        mevc = computation.get_mevc(solution_qualities[step], step, performance_profile, performance_map, configuration)

        if mevc <= 0:
            return step


def get_projected_stopping_point(solution_qualities, steps, time_limit, configuration):
    projected_intrinsic_value_groups = []
    projected_best_time = 0

    time_costs = computation.get_time_costs(steps, configuration['time_cost_multiplier'])

    for sample_limit in range(configuration['monitor_threshold'], time_limit):
        try:
            start = 0 if configuration['window'] is None else sample_limit - configuration['window']

            parameters, _ = curve_fit(utils.get_projected_solution_qualities, steps[start:sample_limit], solution_qualities[start:sample_limit])

            projected_solution_qualities = utils.get_projected_solution_qualities(steps, parameters[0], parameters[1], parameters[2])
            projected_intrinsic_values = computation.get_intrinsic_values(projected_solution_qualities, configuration['intrinsic_value_multiplier'])
            projected_comprehensive_values = computation.get_comprehensive_values(projected_intrinsic_values, time_costs)
            projected_best_time = get_optimal_stopping_point(projected_comprehensive_values)

            projected_intrinsic_value_groups.append(projected_intrinsic_values)

            if projected_best_time < sample_limit - 1:
                projected_best_time = sample_limit - 1
                break
        except:
            pass

    return projected_best_time, projected_intrinsic_value_groups
