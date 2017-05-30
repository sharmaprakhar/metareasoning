import computation
import utils
from scipy.optimize import curve_fit


def get_nonmyopic_best_time(steps, solution_qualities, performance_profile_1, performance_profile_2, solution_quality_classes, solution_quality_class_length, intrinsic_value_multiplier, time_cost_multiplier):
    for step in steps:
        action = computation.get_optimal_values(solution_qualities[step], step, performance_profile_1, performance_profile_2, solution_quality_classes, solution_quality_class_length, intrinsic_value_multiplier, time_cost_multiplier)

        if action is 'stop':
            return step


def get_myopic_best_time(steps, solution_qualities, performance_profile_1, performance_profile_2, solution_quality_classes, solution_quality_class_length, intrinsic_value_multiplier, time_cost_multiplier, time_limit):
    for step in steps:
        if step + 1 == time_limit:
            return step

        mevc = computation.get_mevc(solution_qualities[step], step, performance_profile_1, performance_profile_2, solution_quality_classes, solution_quality_class_length, intrinsic_value_multiplier, time_cost_multiplier)

        if mevc <= 0:
            return step


def get_projected_best_time(steps, solution_qualities, time_costs, monitor_threshold, time_limit, window, intrinsic_value_multiplier):
    projected_intrinsic_value_groups = []
    projected_best_time = 0

    for sample_limit in range(monitor_threshold, time_limit):
        try:
            start = sample_limit - window

            parameters, _ = curve_fit(utils.get_projected_solution_qualities, steps[start:sample_limit], solution_qualities[start:sample_limit])

            projected_solution_qualities = utils.get_projected_solution_qualities(steps, parameters[0], parameters[1], parameters[2])
            projected_intrinsic_values = computation.get_intrinsic_values(projected_solution_qualities, intrinsic_value_multiplier)
            projected_comprehensive_values = computation.get_comprehensive_values(projected_intrinsic_values, time_costs)
            projected_best_time = utils.get_optimal_stopping_point(projected_comprehensive_values)

            projected_intrinsic_value_groups.append(projected_intrinsic_values)

            if projected_best_time < sample_limit - 1:
                projected_best_time = sample_limit - 1
                break
        except:
            pass

    return projected_best_time, projected_intrinsic_value_groups