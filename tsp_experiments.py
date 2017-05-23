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

# BUCKETS = np.linspace(0, 1, 500)
# BUCKETS = [0, 0.65, 0.75, 0.85, 0.90, 0.95, 1]
# BUCKETS = [0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
# BUCKETS = [0, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
BUCKETS = np.linspace(0, 1, 7)
# BUCKETS = np.insert(BUCKETS, 0, 0)
# BUCKETS = [0, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .75, .85, .90, .95, 1]


def get_mevc(solution_quality, time, performance_profile):
    solution_quality_class = utils.digitize(solution_quality, BUCKETS)

    current_intrinsic_value = utils.get_intrinsic_values(solution_quality_class, INTRINSIC_VALUE_MULTIPLIER)
    current_time_cost = utils.get_time_costs(time, TIME_COST_MULTIPLIER)
    current_comprehensive_value = utils.get_comprehensive_values(current_intrinsic_value, current_time_cost)

    # adjusted_performance_profile = utils.get_adjusted_performance_profile(performance_profile, solution_quality_class)

    estimated_next_intrinsic_value = 0
    for i in range(len(BUCKETS) - 1):
        next_intrinsic_value = utils.get_intrinsic_values(i, INTRINSIC_VALUE_MULTIPLIER)
        next_time_cost = utils.get_time_costs(time + 1, TIME_COST_MULTIPLIER)
        next_comprehensive_value = utils.get_comprehensive_values(next_intrinsic_value, next_time_cost)

        # estimated_next_intrinsic_value += adjusted_performance_profile[time + 1][i] * next_comprehensive_value
        estimated_next_intrinsic_value += (performance_profile[solution_quality_class][i] * next_comprehensive_value)

    return estimated_next_intrinsic_value - current_comprehensive_value


def save_performance_profiles(results_filename, instances_filename, directory):
    online_pp_losses = []
    average_pp_losses = []

    solution_quality_map = utils.get_solution_quality_map(results_filename)
    intrinsic_value_averages = utils.get_intrinsic_value_averages(solution_quality_map, INTRINSIC_VALUE_MULTIPLIER)
    # performance_profile = utils.get_naive_performance_profile(solution_quality_map, BUCKETS)
    performance_profile = utils.get_dynamic_performance_profile(solution_quality_map, BUCKETS)
    print performance_profile

    with open(instances_filename) as f:
        lines = f.readlines()#[1:]

        for line in lines:
            instance_filename, _ = utils.get_line_components(line)

            solution_qualities = solution_quality_map[instance_filename]
            plt, online_pp_loss, average_pp_loss = get_performance_profile(solution_qualities, intrinsic_value_averages, performance_profile)

            instance_name = utils.get_instance_name(instance_filename)
            plot_filename = directory + '/' + instance_name + '.png'
            plt.savefig(plot_filename)

            online_pp_losses.append(online_pp_loss)
            average_pp_losses.append(average_pp_loss)

    print("Online Performance Profile Mean Loss: %f" % np.average(online_pp_losses))
    print("Average Performance Profile Mean Loss: %f" % np.average(average_pp_losses))


def get_performance_profile(solution_qualities, intrinsic_value_averages, performance_profile, sample_start=10, sample_end=50):
    plt.figure()
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    intrinsic_values = utils.get_intrinsic_values(solution_qualities, INTRINSIC_VALUE_MULTIPLIER)

    time_limit = len(intrinsic_values)
    time_steps = range(time_limit)

    plt.scatter(time_steps, intrinsic_values, color='r', zorder=3)
    plt.plot(time_steps, intrinsic_value_averages[:time_limit], color='b')

    time_costs = utils.get_time_costs(time_steps, TIME_COST_MULTIPLIER)
    plt.plot(time_steps, -time_costs, color='y')

    comprehensive_values = utils.get_comprehensive_values(intrinsic_values, time_costs)
    plt.plot(time_steps, comprehensive_values, color='g')

    true_best_time = utils.get_optimal_stopping_point(comprehensive_values)
    plt.scatter([true_best_time], comprehensive_values[true_best_time], color='m', zorder=4)
    plt.text(0, 30, "%0.2f - Best Value" % comprehensive_values[true_best_time], color='m')

    decrement = DIFFERENCE / (sample_end - sample_start)
    current_color = INITIAL_GRAY

    average_best_time = None
    for i in range(time_limit):
        if i + 1 == time_limit:
            average_best_time = i
            break

        mevc = get_mevc(solution_qualities[i], i, performance_profile)

        if mevc > 0:
            average_best_time = i
        else:
            break

    plt.scatter([average_best_time], comprehensive_values[average_best_time], color='y', zorder=4)
    plt.text(0, 10, "%0.2f - Best Value w/ Average Performance Profile" % comprehensive_values[average_best_time], color='y')

    # average_comprehensive_values = intrinsic_value_averages - utils.get_time_costs(range(len(intrinsic_value_averages)), TIME_COST_MULTIPLIER)
    # average_best_time = utils.get_optimal_stopping_point(average_comprehensive_values)
    # plt.scatter([average_best_time], comprehensive_values[average_best_time], color='y', zorder=3)
    # plt.text(0, 10, "%0.2f - Best Reward w/ Average Performance Profile" % comprehensive_values[average_best_time], color='y')

    for sample_limit in range(sample_start, sample_end):
        try:
            parameters, _ = curve_fit(utils.get_estimated_intrinsic_value, time_steps[:sample_limit], intrinsic_values[:sample_limit])

            estimated_intrinsic_value = utils.get_estimated_intrinsic_value(time_steps, parameters[0], parameters[1], parameters[2])
            plt.plot(time_steps, estimated_intrinsic_value, color=str(current_color))

            estimated_comprehensive_values = utils.get_comprehensive_values(estimated_intrinsic_value, time_costs)
            estimated_best_time = utils.get_optimal_stopping_point(estimated_comprehensive_values)

            if estimated_best_time <= sample_limit:
                plt.scatter([estimated_best_time], comprehensive_values[estimated_best_time], color='c', zorder=4)
                plt.text(0, 20, "%0.2f - Best Value w/ Online Performance Profile" % comprehensive_values[estimated_best_time], color='c')
                break
            else:
                plt.scatter([estimated_best_time], comprehensive_values[estimated_best_time], color=str(current_color), zorder=3)

            current_color -= decrement
        except Exception as e:
            pass

    online_pp_loss = comprehensive_values[true_best_time] - comprehensive_values[estimated_best_time]
    plt.text(0, -10, "%0.2f - Online Performance Profile Loss" % online_pp_loss)

    average_pp_loss = comprehensive_values[true_best_time] - comprehensive_values[average_best_time]
    plt.text(0, -20, "%0.2f - Average Performance Profile Loss" % average_pp_loss)

    return plt, online_pp_loss, average_pp_loss


def display_solution_qualities(instances_filename):
    solution_quality_map = {}

    with open(instances_filename) as f:
        lines = f.readlines()

        for line in lines:
            instances_filename, optimal_distance = utils.get_line_components(line)

            cities = tsp.load_instance(instances_filename)
            start_city = list(cities)[0]
            statistics = {'time': [], 'distances': []}
            tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

            solution_quality_map[instances_filename] = utils.get_naive_solution_qualities(statistics['distances'], optimal_distance)

    print(json.dumps(solution_quality_map))


def display_performance_profiles(instances_filename):
    with open(instances_filename) as f:
        lines = f.readlines()

        solution_quality_groups = [utils.get_solution_qualities(line) for line in lines]
        max_length = utils.get_max_length(solution_quality_groups)
        trimmed_solution_quality_groups = utils.get_trimmed_groups(solution_quality_groups, max_length)
        solution_quality_averages = utils.get_intrinsic_value_averages(trimmed_solution_quality_groups)

        plt.title('Performance Profile')
        plt.xlabel('Time')
        plt.ylabel('Solution Quality')

        plt.plot(range(len(solution_quality_averages)), solution_quality_averages, color='b')

        for solution_qualities in solution_quality_groups:
            plt.scatter(range(len(solution_qualities)), solution_qualities, color='r', s=0.1)

        plt.show()


def display_projections(instance_filename, optimal_distance, results_filename):
    with open(results_filename) as f:
        lines = f.readlines()

        solution_quality_groups = [utils.get_solution_qualities(line) for line in lines]
        max_length = utils.get_max_length(solution_quality_groups)
        trimmed_solution_quality_groups = utils.get_trimmed_groups(solution_quality_groups, max_length)
        solution_quality_averages = utils.get_intrinsic_value_averages(trimmed_solution_quality_groups)

        cities = tsp.load_instance(instance_filename)
        start_city = list(cities)[0]
        statistics = {'time': [], 'distances': []}
        tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

        solution_qualities = utils.get_naive_solution_qualities(statistics['distances'], optimal_distance)

        plt.title('Performance Profile')
        plt.xlabel('Time')
        plt.ylabel('Solution Quality')

        plt.scatter(statistics['time'], solution_qualities, color='r')
        plt.plot(range(len(solution_quality_averages)), solution_quality_averages, color='b')

        for sample_limit in range(4, 20):
            try:
                parameters, _ = curve_fit(utils.get_estimated_intrinsic_value, statistics['time'][:sample_limit], solution_qualities[:sample_limit])
                estimates = utils.get_estimated_intrinsic_value(statistics['time'], parameters[0], parameters[1], parameters[2], parameters[3])
                plt.plot(statistics['time'], estimates, color='r')
            except:
                print('Encountered exception')

        plt.show()


def main():
    save_performance_profiles('results/naive.json', 'instances/50-tsp/instances.csv', 'plots')
    # display_solution_qualities('instances/50-tsp/instances.csv')

if __name__ == '__main__':
    main()
