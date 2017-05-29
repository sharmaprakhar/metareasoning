from __future__ import division

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import tsp
import tsp_solver
import utils
import tsp_problem
import computation
import performance_profile

TIME_COST_MULTIPLIER = 0.5
INTRINSIC_VALUE_MULTIPLIER = 100

INITIAL_GRAY = 0.9
TERMINAL_GRAY = 0.3
DIFFERENCE = INITIAL_GRAY - TERMINAL_GRAY

BUCKETS = np.linspace(0, 1, 201)
BUCKET_SIZE = len(BUCKETS) - 1

TOUR_SIZE = 50


def save_performance_profiles(results_filename, directory):
    solution_quality_map = utils.get_solution_quality_map(results_filename)
    performance_profile_1 = performance_profile.get_dynamic_performance_profile_1(solution_quality_map, BUCKETS)
    performance_profile_2 = performance_profile.get_dynamic_performance_profile_2(solution_quality_map, BUCKETS)
    intrinsic_value_averages = utils.get_intrinsic_value_averages(solution_quality_map, INTRINSIC_VALUE_MULTIPLIER)

    nonmyopic_losses = []
    myopic_losses = []
    online_losses = []
    fixed_time_losses = []

    for instance_filename in solution_quality_map:
        print('Instance: %s' % instance_filename)

        solution_qualities = solution_quality_map[instance_filename]['solution_qualities']
        estimated_solution_qualities = solution_quality_map[instance_filename]['estimated_solution_qualities']

        plt, online_loss, myopic_loss, nonmyopic_loss, fixed_time_loss = get_performance_profile(solution_qualities, estimated_solution_qualities, intrinsic_value_averages, performance_profile_1, performance_profile_2, 10, 20)

        instance_id = utils.get_instance_name(instance_filename)
        plot_filename = directory + '/' + instance_id + '.png'
        plt.savefig(plot_filename)

        nonmyopic_losses.append(nonmyopic_loss)
        myopic_losses.append(myopic_loss)
        online_losses.append(online_loss)
        fixed_time_losses.append(fixed_time_loss)

    print("Online Monitoring Mean Error: %f%%" % np.average(online_losses))
    print("Myopic Monitoring Mean Error: %f%%" % np.average(myopic_losses))
    print("Nonmyopic Monitoring Mean Error: %f%%" % np.average(nonmyopic_losses))
    print("Fixed Time Allocation Mean Error: %f%%" % np.average(fixed_time_losses))


def get_performance_profile(solution_qualities, estimated_solution_qualities, intrinsic_value_averages, performance_profile_1, performance_profile_2, monitor_threshold, window):
    plt.figure(figsize=(12, 10))
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.annotate("%d-TSP" % TOUR_SIZE, xy=(0, 0), xytext=(10, 147), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate("%d Discrete Solution Qualities" % BUCKET_SIZE, xy=(0, 0), xytext=(10, 137), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate("$q(s) = Length_{MST} / Length(s)$", xy=(0, 0), xytext=(10, 125), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate("$U_C(t) = %.2ft$" % TIME_COST_MULTIPLIER, xy=(0, 0), xytext=(10, 115), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate("$U_I(q) = %dq$" % INTRINSIC_VALUE_MULTIPLIER, xy=(0, 0), xytext=(10, 105), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate("$U(q, t) = U_C(t) - U_I(q)$", xy=(0, 0), xytext=(10, 95), va='bottom', xycoords='axes fraction', textcoords='offset points')

    intrinsic_values = computation.get_intrinsic_values(solution_qualities, INTRINSIC_VALUE_MULTIPLIER)
    estimated_intrinsic_values = computation.get_intrinsic_values(estimated_solution_qualities, INTRINSIC_VALUE_MULTIPLIER)

    time_limit = len(solution_qualities)
    steps = range(time_limit)

    plt.scatter(steps, intrinsic_values, color='g', zorder=3, label='Intrinsic Values')
    plt.scatter(steps, estimated_intrinsic_values, color='darkorange', zorder=3, label='Estimated Intrinsic Values')
    plt.plot(steps, intrinsic_value_averages[:time_limit], color='b', label='Expected Performance Profile')

    time_costs = computation.get_time_costs(steps, TIME_COST_MULTIPLIER)
    plt.plot(steps, -time_costs, color='r', label='Cost of Time')

    comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)
    plt.plot(steps, comprehensive_values, color='k', label='Comprehensive Values')

    optimal_stopping_point = utils.get_optimal_stopping_point(comprehensive_values)
    plt.scatter([optimal_stopping_point], comprehensive_values[optimal_stopping_point], color='limegreen', zorder=4, label='Optimal Stopping Point')
    plt.annotate("%0.2f - Best Value" % comprehensive_values[optimal_stopping_point], xy=(0, 0), xytext=(10, 75), va='bottom', xycoords='axes fraction', textcoords='offset points', color='limegreen')

    myopic_best_time = 0
    for step in steps:
        if step + 1 == time_limit:
            myopic_best_time = step
            break

        mevc = computation.get_mevc(estimated_solution_qualities[step], step, performance_profile_1, performance_profile_2, BUCKETS, BUCKET_SIZE, INTRINSIC_VALUE_MULTIPLIER, TIME_COST_MULTIPLIER)

        if mevc <= 0:
            myopic_best_time = step
            break

    nonmyopic_best_time = 0
    for step in steps:
        action = computation.get_optimal_values(estimated_solution_qualities[step], step, performance_profile_1, performance_profile_2, BUCKETS, BUCKET_SIZE, INTRINSIC_VALUE_MULTIPLIER, TIME_COST_MULTIPLIER)

        if action is 'stop':
            nonmyopic_best_time = step
            break

    plt.scatter([myopic_best_time], comprehensive_values[myopic_best_time], color='y', zorder=4, label='Myopic Stopping Point')
    plt.annotate("%0.2f - Best Value - Myopic Monitoring" % comprehensive_values[myopic_best_time], xy=(0, 0), xytext=(10, 55), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')

    average_comprehensive_values = intrinsic_value_averages - computation.get_time_costs(range(len(intrinsic_value_averages)), TIME_COST_MULTIPLIER)
    fixed_best_time = utils.get_optimal_stopping_point(average_comprehensive_values)
    offset_fixed_best_time = fixed_best_time if fixed_best_time < time_limit else time_limit - 1
    plt.scatter([offset_fixed_best_time], comprehensive_values[offset_fixed_best_time], color='c', zorder=4, label='Fixed Stopping Point')
    plt.annotate("%0.2f - Best Value - Fixed Time Allocation" % comprehensive_values[offset_fixed_best_time], xy=(0, 0), xytext=(10, 45), va='bottom', xycoords='axes fraction', textcoords='offset points', color='c')

    projected_intrinsic_value_groups = []

    for sample_limit in range(monitor_threshold, time_limit):
        try:
            start = sample_limit - window

            parameters, _ = curve_fit(utils.get_projected_solution_qualities, steps[start:sample_limit], estimated_solution_qualities[start:sample_limit])

            projected_solution_qualities = utils.get_projected_solution_qualities(steps, parameters[0], parameters[1], parameters[2])
            projected_intrinsic_values = computation.get_intrinsic_values(projected_solution_qualities, INTRINSIC_VALUE_MULTIPLIER)
            projected_comprehensive_values = computation.get_comprehensive_values(projected_intrinsic_values, time_costs)
            projected_best_time = utils.get_optimal_stopping_point(projected_comprehensive_values)

            projected_intrinsic_value_groups.append(projected_intrinsic_values)

            # TODO The performance always gets better when I remove =
            if projected_best_time <= sample_limit - 1:
                projected_best_time = sample_limit - 1
                plt.scatter([projected_best_time], comprehensive_values[projected_best_time], color='m', zorder=4, label='Online Stopping Point')
                plt.annotate("%0.2f - Best Value - Online Monitoring" % comprehensive_values[projected_best_time], xy=(0, 0), xytext=(10, 65), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
                break
        except:
            pass

    decrement = DIFFERENCE / len(projected_intrinsic_value_groups)
    current_color = INITIAL_GRAY

    for projected_intrinsic_values in projected_intrinsic_value_groups:
        plt.plot(steps, projected_intrinsic_values, color=str(current_color))
        current_color -= decrement

    online_loss = ((comprehensive_values[optimal_stopping_point] - comprehensive_values[projected_best_time]) / comprehensive_values[optimal_stopping_point]) * 100
    plt.annotate("%0.2f%% - Error - Online Monitoring" % online_loss, xy=(0, 0), xytext=(10, 25), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')

    myopic_loss = ((comprehensive_values[optimal_stopping_point] - comprehensive_values[myopic_best_time]) / comprehensive_values[optimal_stopping_point]) * 100
    plt.annotate("%0.2f%% - Error - Myopic Monitoring" % myopic_loss, xy=(0, 0), xytext=(10, 15), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')

    nonmyopic_loss = ((comprehensive_values[optimal_stopping_point] - comprehensive_values[nonmyopic_best_time]) / comprehensive_values[optimal_stopping_point]) * 100

    fixed_time_loss = ((comprehensive_values[optimal_stopping_point] - comprehensive_values[offset_fixed_best_time]) / comprehensive_values[optimal_stopping_point]) * 100
    plt.annotate("%0.2f%% - Error - Fixed Time Allocation" % fixed_time_loss, xy=(0, 0), xytext=(10, 5), va='bottom', xycoords='axes fraction', textcoords='offset points', color='c')

    plt.legend(bbox_to_anchor=(0.0, 1.05, 1.0, 0.102), loc=3, ncol=3, mode="expand", borderaxespad=0.0)

    return plt, online_loss, myopic_loss, nonmyopic_loss, fixed_time_loss


def print_solution_quality_map(instances_filename, get_solution_qualities):
    solution_quality_map = {}

    with open(instances_filename) as f:
        for line in f.readlines():
            instance_filename, optimal_distance = utils.get_line_components(line)

            cities, start_city = tsp.load_instance(instance_filename)
            statistics = {'time': [], 'distances': []}
            tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

            estimated_optimal_distance = tsp_problem.prim(start_city, list(cities)[1:])

            solution_quality_map[instance_filename] = {
                'solution_qualities': get_solution_qualities(statistics['distances'], optimal_distance),
                'estimated_solution_qualities': get_solution_qualities(statistics['distances'], estimated_optimal_distance)
            }

    print(json.dumps(solution_quality_map))


def main():
    save_performance_profiles('results/results.json', 'plots')


if __name__ == '__main__':
    main()
