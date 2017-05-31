from __future__ import division

import json

import matplotlib.pyplot as plt
import numpy as np

import computation
import monitor
import performance_profile
import tsp
import tsp_problem
import tsp_solver
import utils

TIME_COST_MULTIPLIER = 0.5
INTRINSIC_VALUE_MULTIPLIER = 100
SOLUTION_QUALITY_CLASSES = np.linspace(0, 1, 201)
SOLUTION_QUALITY_CLASS_LENGTH = len(SOLUTION_QUALITY_CLASSES) - 1
MONITOR_THRESHOLD = 10
WINDOW = 10

TOUR_SIZE = 50
INITIAL_GRAY = 0.9
TERMINAL_GRAY = 0.3
DIFFERENCE = INITIAL_GRAY - TERMINAL_GRAY


def save_performance_profiles(results_filename, directory):
    solution_quality_map = utils.get_solution_quality_map(results_filename)
    performance_profile_1 = performance_profile.get_dynamic_estimated_performance_profile(solution_quality_map, SOLUTION_QUALITY_CLASSES)
    performance_profile_2 = performance_profile.get_dynamic_estimation_map(solution_quality_map, SOLUTION_QUALITY_CLASSES)
    intrinsic_value_averages = utils.get_intrinsic_value_averages(solution_quality_map, INTRINSIC_VALUE_MULTIPLIER)

    online_losses = []
    nonmyopic_losses = []
    myopic_losses = []
    fixed_time_losses = []

    for instance_filename in solution_quality_map:
        print('Instance: %s' % instance_filename)

        solution_qualities = solution_quality_map[instance_filename]['solution_qualities']
        estimated_solution_qualities = solution_quality_map[instance_filename]['estimated_solution_qualities']

        plt, online_loss, myopic_loss, nonmyopic_loss, fixed_time_loss = get_performance_profile(solution_qualities, estimated_solution_qualities, intrinsic_value_averages, performance_profile_1, performance_profile_2)

        instance_id = utils.get_instance_name(instance_filename)
        plot_filename = directory + '/' + instance_id + '.png'
        plt.savefig(plot_filename)
        plt.close()

        online_losses.append(online_loss)
        nonmyopic_losses.append(nonmyopic_loss)
        myopic_losses.append(myopic_loss)
        fixed_time_losses.append(fixed_time_loss)

    print('Online Monitoring Mean Error: %f%%' % np.average(online_losses))
    print('Nonmyopic Monitoring Mean Error: %f%%' % np.average(nonmyopic_losses))
    print('Myopic Monitoring Mean Error: %f%%' % np.average(myopic_losses))
    print('Fixed Time Allocation Mean Error: %f%%' % np.average(fixed_time_losses))


def get_performance_profile(solution_qualities, estimated_solution_qualities, intrinsic_value_averages, performance_profile_1, performance_profile_2):
    time_limit = len(solution_qualities)
    steps = range(time_limit)

    intrinsic_values = computation.get_intrinsic_values(solution_qualities, INTRINSIC_VALUE_MULTIPLIER)
    time_costs = computation.get_time_costs(steps, TIME_COST_MULTIPLIER)
    comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)

    estimated_intrinsic_values = computation.get_intrinsic_values(estimated_solution_qualities, INTRINSIC_VALUE_MULTIPLIER)

    optimal_stopping_point = monitor.get_optimal_stopping_point(comprehensive_values)
    projected_stopping_point, projected_intrinsic_value_groups = monitor.get_projected_best_time(steps, estimated_solution_qualities, time_costs, MONITOR_THRESHOLD, time_limit, WINDOW, INTRINSIC_VALUE_MULTIPLIER)
    nonmyopic_stopping_point = monitor.get_nonmyopic_best_time(steps, estimated_solution_qualities, performance_profile_1, performance_profile_2, SOLUTION_QUALITY_CLASSES, SOLUTION_QUALITY_CLASS_LENGTH, INTRINSIC_VALUE_MULTIPLIER, TIME_COST_MULTIPLIER)
    myopic_stopping_point = monitor.get_myopic_best_time(steps, estimated_solution_qualities, performance_profile_1, performance_profile_2, SOLUTION_QUALITY_CLASSES, SOLUTION_QUALITY_CLASS_LENGTH, INTRINSIC_VALUE_MULTIPLIER, TIME_COST_MULTIPLIER, time_limit)
    fixed_stopping_point = monitor.get_fixed_stopping_point(intrinsic_value_averages, TIME_COST_MULTIPLIER, time_limit)

    online_loss = utils.get_percent_error(comprehensive_values[optimal_stopping_point], comprehensive_values[projected_stopping_point])
    nonmyopic_loss = utils.get_percent_error(comprehensive_values[optimal_stopping_point], comprehensive_values[nonmyopic_stopping_point])
    myopic_loss = utils.get_percent_error(comprehensive_values[optimal_stopping_point], comprehensive_values[myopic_stopping_point])
    fixed_time_loss = utils.get_percent_error(comprehensive_values[optimal_stopping_point], comprehensive_values[fixed_stopping_point])

    plt.figure(figsize=(12, 12))
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.annotate('%d-TSP' % TOUR_SIZE, xy=(0, 0), xytext=(10, 167), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('%d Discrete Solution Qualities' % SOLUTION_QUALITY_CLASS_LENGTH, xy=(0, 0), xytext=(10, 157), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$q(s) = Length_{MST} / Length(s)$', xy=(0, 0), xytext=(10, 145), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$U_C(t) = %.2ft$' % TIME_COST_MULTIPLIER, xy=(0, 0), xytext=(10, 135), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$U_I(q) = %dq$' % INTRINSIC_VALUE_MULTIPLIER, xy=(0, 0), xytext=(10, 125), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$U(q, t) = U_C(t) - U_I(q)$', xy=(0, 0), xytext=(10, 115), va='bottom', xycoords='axes fraction', textcoords='offset points')

    plt.scatter(steps, intrinsic_values, color='g', zorder=3, label='Intrinsic Values')
    plt.scatter(steps, estimated_intrinsic_values, color='darkorange', zorder=3, label='Estimated Intrinsic Values')

    plt.plot(steps, intrinsic_value_averages[:time_limit], color='b', label='Expected Performance Profile')
    plt.plot(steps, -time_costs, color='r', label='Cost of Time')
    plt.plot(steps, comprehensive_values, color='k', label='Comprehensive Values')

    plt.scatter([optimal_stopping_point], comprehensive_values[optimal_stopping_point], color='limegreen', zorder=4, label='Optimal Stopping Point')
    plt.scatter([nonmyopic_stopping_point], comprehensive_values[nonmyopic_stopping_point], color='pink', zorder=4, label='Nonmyopic Stopping Point')
    plt.scatter([myopic_stopping_point], comprehensive_values[myopic_stopping_point], color='y', zorder=4, label='Myopic Stopping Point')
    plt.scatter([fixed_stopping_point], comprehensive_values[fixed_stopping_point], color='c', zorder=4, label='Fixed Stopping Point')
    plt.scatter([projected_stopping_point], comprehensive_values[projected_stopping_point], color='m', zorder=4, label='Online Stopping Point')

    plt.annotate('%0.2f - Best Value' % comprehensive_values[optimal_stopping_point], xy=(0, 0), xytext=(10, 95), va='bottom', xycoords='axes fraction', textcoords='offset points', color='limegreen')
    plt.annotate('%0.2f - Best Value - Nonmyopic Monitoring' % comprehensive_values[nonmyopic_stopping_point], xy=(0, 0), xytext=(10, 75), va='bottom', xycoords='axes fraction', textcoords='offset points', color='pink')
    plt.annotate('%0.2f - Best Value - Myopic Monitoring' % comprehensive_values[myopic_stopping_point], xy=(0, 0), xytext=(10, 65), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')
    plt.annotate('%0.2f - Best Value - Fixed Time Allocation' % comprehensive_values[fixed_stopping_point], xy=(0, 0), xytext=(10, 55), va='bottom', xycoords='axes fraction', textcoords='offset points', color='c')
    plt.annotate('%0.2f - Best Value - Online Monitoring' % comprehensive_values[projected_stopping_point], xy=(0, 0), xytext=(10, 85), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')

    decrement = DIFFERENCE / len(projected_intrinsic_value_groups)
    current_color = INITIAL_GRAY

    for projected_intrinsic_values in projected_intrinsic_value_groups:
        plt.plot(steps, projected_intrinsic_values, color=str(current_color))
        current_color -= decrement

    plt.annotate('%0.2f%% - Error - Online Monitoring' % online_loss, xy=(0, 0), xytext=(10, 35), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
    plt.annotate('%0.2f%% - Error - Nonmyopic Monitoring' % nonmyopic_loss, xy=(0, 0), xytext=(10, 25), va='bottom', xycoords='axes fraction', textcoords='offset points', color='pink')
    plt.annotate('%0.2f%% - Error - Myopic Monitoring' % myopic_loss, xy=(0, 0), xytext=(10, 15), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')
    plt.annotate('%0.2f%% - Error - Fixed Time Allocation' % fixed_time_loss, xy=(0, 0), xytext=(10, 5), va='bottom', xycoords='axes fraction', textcoords='offset points', color='c')

    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, 0.102), loc=3, ncol=3, mode='expand', borderaxespad=0.0)

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
