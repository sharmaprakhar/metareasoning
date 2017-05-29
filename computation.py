import numpy as np
import utils


def get_intrinsic_values(solution_qualities, multiplier):
    return np.multiply(multiplier, solution_qualities)


def get_time_costs(time, multiplier):
    return np.multiply(multiplier, time)


def get_comprehensive_values(instrinsic_value, time_cost):
    return instrinsic_value - time_cost


def get_mevc(estimated_solution_quality, step, performance_profile_1, performance_profile_2, buckets, bucket_size, intrinsic_value_multiplier, time_cost_mulitiplier):
    solution_quality_classes = range(bucket_size)

    current_estimated_solution_quality_class = utils.digitize(estimated_solution_quality, buckets)

    expected_current_comprehensive_value = 0
    for solution_quality_class in solution_quality_classes:
        current_solution_quality = utils.get_solution_quality(solution_quality_class, bucket_size)
        current_intrinsic_value = get_intrinsic_values(current_solution_quality, intrinsic_value_multiplier)
        current_time_cost = get_time_costs(step, time_cost_mulitiplier)
        current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

        expected_current_comprehensive_value += performance_profile_2[current_estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

    expected_next_comprehensive_value = 0
    for solution_quality_class in solution_quality_classes:
        next_solution_quality = utils.get_solution_quality(solution_quality_class, bucket_size)
        next_intrinsic_value = get_intrinsic_values(next_solution_quality, intrinsic_value_multiplier)
        next_time_cost = get_time_costs(step + 1, time_cost_mulitiplier)
        next_comprehensive_value = get_comprehensive_values(next_intrinsic_value, next_time_cost)

        expected_next_comprehensive_value += performance_profile_1[current_estimated_solution_quality_class][step][solution_quality_class] * next_comprehensive_value

    return expected_next_comprehensive_value - expected_current_comprehensive_value


# def get_mevc(solution_quality, step, performance_profile):
#     current_solution_quality_class = utils.digitize(solution_quality, BUCKETS)
#     current_solution_quality = utils.get_solution_quality(current_solution_quality_class, BUCKET_SIZE)
#     current_intrinsic_value = computation.get_intrinsic_values(current_solution_quality, INTRINSIC_VALUE_MULTIPLIER)
#     current_time_cost = computation.get_time_costs(step, TIME_COST_MULTIPLIER)
#     current_comprehensive_value = computation.get_comprehensive_values(current_intrinsic_value, current_time_cost)
#
#     solution_quality_classes = range(BUCKET_SIZE)
#     expected_next_comprehensive_value = 0
#
#     for next_solution_quality_class in solution_quality_classes:
#         next_solution_quality = utils.get_solution_quality(next_solution_quality_class, BUCKET_SIZE)
#         next_intrinsic_value = computation.get_intrinsic_values(next_solution_quality, INTRINSIC_VALUE_MULTIPLIER)
#         next_time_cost = computation.get_time_costs(step + 1, TIME_COST_MULTIPLIER)
#         next_comprehensive_value = computation.get_comprehensive_values(next_intrinsic_value, next_time_cost)
#
#         expected_next_comprehensive_value += performance_profile[current_solution_quality_class][step][next_solution_quality_class] * next_comprehensive_value
#
#     return expected_next_comprehensive_value - current_comprehensive_value


def get_optimal_values(state, step, performance_profile_1, performance_profile_2, buckets, bucket_size, intrinsic_value_multiplier, time_cost_mulitiplier):
    solution_quality_classes = range(bucket_size)
    value = 0

    estimated_solution_quality_class = utils.digitize(state, buckets)

    best_action = ''

    while True:
        delta = 0

        stop_value = 0
        for solution_quality_class in solution_quality_classes:
            current_solution_quality = utils.get_solution_quality(solution_quality_class, bucket_size)
            current_intrinsic_value = get_intrinsic_values(current_solution_quality, intrinsic_value_multiplier)
            current_time_cost = get_time_costs(step, time_cost_mulitiplier)
            current_comprehensive_value = get_comprehensive_values(current_intrinsic_value, current_time_cost)

            stop_value += performance_profile_2[estimated_solution_quality_class][step][solution_quality_class] * current_comprehensive_value

        continue_value = 0
        for solution_quality_class in solution_quality_classes:
            current_solution_quality = utils.get_solution_quality(solution_quality_class, bucket_size)
            current_intrinsic_value = get_intrinsic_values(current_solution_quality, intrinsic_value_multiplier)
            current_time_cost = get_time_costs(step, time_cost_mulitiplier)
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
