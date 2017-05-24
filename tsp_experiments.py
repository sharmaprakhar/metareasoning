from __future__ import division

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import tsp
import tsp_solver
import utils

TIME_COST_MULTIPLIER = 2
INTRINSIC_VALUE_MULTIPLIER = 200

INITIAL_GRAY = 0.9
TERMINAL_GRAY = 0
DIFFERENCE = INITIAL_GRAY - TERMINAL_GRAY

BUCKETS = np.linspace(0, 1, 20)
BUCKET_SIZE = len(BUCKETS) - 1


def get_mevc(solution_quality, step, performance_profile):
    current_solution_quality_class = utils.digitize(solution_quality, BUCKETS)
    current_solution_quality = utils.get_solution_quality(current_solution_quality_class, BUCKET_SIZE)
    current_intrinsic_value = utils.get_intrinsic_values(current_solution_quality, INTRINSIC_VALUE_MULTIPLIER)
    current_time_cost = utils.get_time_costs(step, TIME_COST_MULTIPLIER)
    current_comprehensive_value = utils.get_comprehensive_values(current_intrinsic_value, current_time_cost)

    solution_quality_classes = range(BUCKET_SIZE)
    estimated_next_intrinsic_value = 0

    for next_solution_quality_class in solution_quality_classes:
        next_solution_quality = utils.get_solution_quality(next_solution_quality_class, BUCKET_SIZE)
        next_intrinsic_value = utils.get_intrinsic_values(next_solution_quality, INTRINSIC_VALUE_MULTIPLIER)
        next_time_cost = utils.get_time_costs(step + 1, TIME_COST_MULTIPLIER)
        next_comprehensive_value = utils.get_comprehensive_values(next_intrinsic_value, next_time_cost)

        estimated_next_intrinsic_value += performance_profile[current_solution_quality_class][next_solution_quality_class] * next_comprehensive_value

    return estimated_next_intrinsic_value - current_comprehensive_value


def get_optimal_fixed_allocation_time(performance_profile):
    best_step = None
    best_value = float('-inf')

    time_limit = len(performance_profile.keys())
    steps = range(time_limit)
    solution_quality_classes = range(BUCKET_SIZE)

    for step in steps:
        estimated_value = 0

        for solution_quality_class in solution_quality_classes:
            adjusted_performance_profile = utils.get_adjusted_performance_profile(performance_profile, solution_quality_class)

            probability = adjusted_performance_profile[step][solution_quality_class]

            solution_quality = utils.get_solution_quality(solution_quality_class, BUCKET_SIZE)
            intrinsic_value = utils.get_intrinsic_values(solution_quality, INTRINSIC_VALUE_MULTIPLIER)
            time_cost = utils.get_time_costs(step, TIME_COST_MULTIPLIER)
            comprehensive_value = utils.get_comprehensive_values(intrinsic_value, time_cost)

            estimated_value += probability * comprehensive_value

        if estimated_value > best_value:
            best_step = step
            best_value = estimated_value

    return best_step


def save_performance_profiles(results_filename, directory):
    solution_quality_map = utils.get_solution_quality_map(results_filename)
    performance_profile = utils.get_dynamic_performance_profile(solution_quality_map, BUCKETS)
    intrinsic_value_averages = utils.get_intrinsic_value_averages(solution_quality_map, INTRINSIC_VALUE_MULTIPLIER)

    myopic_losses = []
    online_losses = []
    fixed_time_losses = []

    for instance_filename in solution_quality_map:
        print('Instance: %s' % instance_filename)

        solution_qualities = solution_quality_map[instance_filename]

        plt, online_loss, myopic_loss, fixed_time_loss = get_performance_profile(solution_qualities, intrinsic_value_averages, performance_profile, 10, 10)

        instance_id = utils.get_instance_name(instance_filename)
        plot_filename = directory + '/' + instance_id + '.png'
        plt.savefig(plot_filename)

        myopic_losses.append(myopic_loss)
        online_losses.append(online_loss)
        fixed_time_losses.append(fixed_time_loss)

    print("Online Monitoring Mean Loss: %f" % np.average(online_losses))
    print("Myopic Monitoring Mean Loss: %f" % np.average(myopic_losses))
    print("Fixed Time Allocation Mean Loss: %f" % np.average(fixed_time_losses))


def get_performance_profile(solution_qualities, intrinsic_value_averages, performance_profile, monitor_threshold, window):
    plt.figure()
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    intrinsic_values = utils.get_intrinsic_values(solution_qualities, INTRINSIC_VALUE_MULTIPLIER)

    time_limit = len(solution_qualities)
    steps = range(time_limit)

    plt.scatter(steps, intrinsic_values, color='r', zorder=3)
    plt.plot(steps, intrinsic_value_averages[:time_limit], color='b')

    time_costs = utils.get_time_costs(steps, TIME_COST_MULTIPLIER)
    plt.plot(steps, -time_costs, color='y')

    comprehensive_values = utils.get_comprehensive_values(intrinsic_values, time_costs)
    plt.plot(steps, comprehensive_values, color='g')

    optimal_stopping_point = utils.get_optimal_stopping_point(comprehensive_values)
    plt.scatter([optimal_stopping_point], comprehensive_values[optimal_stopping_point], color='m', zorder=4)
    plt.text(0, 40, "%0.2f - Best Value" % comprehensive_values[optimal_stopping_point], color='m')

    average_best_time = 0
    for step in steps:
        if step + 1 == time_limit:
            average_best_time = step
            break

        mevc = get_mevc(solution_qualities[step], step, performance_profile)

        if mevc <= 0:
            average_best_time = step
            break

    plt.scatter([average_best_time], comprehensive_values[average_best_time], color='y', zorder=4)
    plt.text(0, 20, "%0.2f - Best Value w/ Myopic Monitoring" % comprehensive_values[average_best_time], color='y')

    average_comprehensive_values = intrinsic_value_averages - utils.get_time_costs(range(len(intrinsic_value_averages)), TIME_COST_MULTIPLIER)
    fixed_best_time = utils.get_optimal_stopping_point(average_comprehensive_values)
    offset_fixed_best_time = fixed_best_time if fixed_best_time < time_limit else time_limit - 1
    plt.scatter([offset_fixed_best_time], comprehensive_values[offset_fixed_best_time], color='y', zorder=4)
    plt.text(0, 10, "%0.2f - Best Reward w/ Fixed Time Allocation" % comprehensive_values[offset_fixed_best_time], color='y')

    decrement = DIFFERENCE / (time_limit - monitor_threshold)
    current_color = INITIAL_GRAY

    for sample_limit in range(monitor_threshold, time_limit):
        try:
            start = sample_limit - window
            parameters, _ = curve_fit(utils.get_estimated_solution_qualities, steps[start:sample_limit], solution_qualities[start:sample_limit])

            estimated_solution_qualities = utils.get_estimated_solution_qualities(steps, parameters[0], parameters[1], parameters[2])
            estimated_intrinsic_values = utils.get_intrinsic_values(estimated_solution_qualities, INTRINSIC_VALUE_MULTIPLIER)
            plt.plot(steps, estimated_intrinsic_values, color=str(current_color))

            estimated_comprehensive_values = utils.get_comprehensive_values(estimated_intrinsic_values, time_costs)
            estimated_best_time = utils.get_optimal_stopping_point(estimated_comprehensive_values)

            if estimated_best_time > sample_limit:
                plt.scatter([estimated_best_time], comprehensive_values[estimated_best_time], color=str(current_color), zorder=3)

            # TODO Is my stopping criterion correct?
            if estimated_best_time <= sample_limit - 1:
                estimated_best_time = sample_limit - 1
                plt.scatter([estimated_best_time], comprehensive_values[estimated_best_time], color='c', zorder=4)
                plt.text(0, 30, "%0.2f - Best Value w/ Online Monitoring" % comprehensive_values[estimated_best_time], color='c')
                break

            current_color -= decrement
        except:
            pass

    online_loss = comprehensive_values[optimal_stopping_point] - comprehensive_values[estimated_best_time]
    plt.text(0, -10, "%0.2f - Online Monitoring Loss" % online_loss, color='c')

    myopic_loss = comprehensive_values[optimal_stopping_point] - comprehensive_values[average_best_time]
    plt.text(0, -20, "%0.2f - Myopic Monitoring Loss" % myopic_loss, color='y')

    fixed_time_loss = comprehensive_values[optimal_stopping_point] - comprehensive_values[offset_fixed_best_time]
    plt.text(0, -30, "%0.2f - Fixed Time Allocation Loss" % fixed_time_loss, color='y')

    return plt, online_loss, myopic_loss, fixed_time_loss


def print_solution_quality_map(instances_filename, get_solution_qualities):
    solution_quality_map = {}

    with open(instances_filename) as f:
        for line in f.readlines():
            instance_filename, optimal_distance = utils.get_line_components(line)

            cities, start_city = tsp.load_instance(instance_filename)
            statistics = {'time': [], 'distances': []}
            tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

            solution_quality_map[instance_filename] = get_solution_qualities(statistics['distances'], optimal_distance)

    print(json.dumps(solution_quality_map))


def main():
    save_performance_profiles('results/50-tsp-approximation-ratio-map.json', 'plots')


if __name__ == '__main__':
    main()
