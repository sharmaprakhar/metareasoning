from scipy.optimize import curve_fit
import computation
import utils


def get_optimal_stopping_point(comprehensive_values):
    return list(comprehensive_values).index(max(comprehensive_values))


def get_fixed_stopping_point(intrinsic_values, limit, config):
    steps = range(len(intrinsic_values))
    time_costs = computation.get_time_costs(steps, config['time_cost_multiplier'])
    comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)
    stopping_point = get_optimal_stopping_point(comprehensive_values)
    return stopping_point if stopping_point < limit else limit - 1


def get_nonmyopic_stopping_point(qualities, steps, profile_2, profile_3, time_limit, config):
    values = computation.get_optimal_values(steps, profile_2, profile_3, config)

    for step in steps:
        if step + 1 == time_limit:
            return step

        action = computation.get_optimal_action(qualities[step], step, values, profile_2, profile_3, config)

        if action is computation.STOP_SYMBOL:
            return step


def get_myopic_stopping_point(qualities, steps, profile_1, profile_3, time_limit, config):
    for step in steps:
        if step + 1 == time_limit:
            return step

        mevc = computation.get_mevc(qualities[step], step, profile_1, profile_3, config)

        if mevc <= 0:
            return step


def get_projected_stopping_point(qualities, steps, time_limit, config):
    intrinsic_value_groups = []
    stopping_point = 0

    time_costs = computation.get_time_costs(steps, config['time_cost_multiplier'])

    for end in range(config['monitor_threshold'], time_limit):
        try:
            start = 0 if config['window'] is None else end - config['window']

            parameters, _ = curve_fit(utils.get_projected_solution_qualities, steps[start:end], qualities[start:end])

            projected_qualities = utils.get_projected_solution_qualities(steps, parameters[0], parameters[1], parameters[2])
            intrinsic_values = computation.get_intrinsic_values(projected_qualities, config['intrinsic_value_multiplier'])
            comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)
            stopping_point = get_optimal_stopping_point(comprehensive_values)

            intrinsic_value_groups.append(intrinsic_values)

            if stopping_point < end - 1:
                return end - 1, intrinsic_value_groups
        except RuntimeError:
            pass

    return stopping_point, intrinsic_value_groups
