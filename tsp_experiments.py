import json

import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
import numpy as np

import computation
import monitor
import performance
import tsp
import tsp_solver
import utils

TIME_COST_MULTIPLIER = 0.1
INTRINSIC_VALUE_MULTIPLIER = 200
SOLUTION_QUALITY_CLASS_COUNT = 15
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(0, 1, SOLUTION_QUALITY_CLASS_COUNT + 1)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)
MONITOR_THRESHOLD = 10
WINDOW = None

CONFIG = {
    'time_cost_multiplier': TIME_COST_MULTIPLIER,
    'intrinsic_value_multiplier': INTRINSIC_VALUE_MULTIPLIER,
    'solution_quality_classes': SOLUTION_QUALITY_CLASSES,
    'solution_quality_class_bounds': SOLUTION_QUALITY_CLASS_BOUNDS,
    'solution_quality_class_count': SOLUTION_QUALITY_CLASS_COUNT,
    'monitor_threshold': MONITOR_THRESHOLD,
    'window': WINDOW
}

TOUR_SIZE = 50
INITIAL_GRAY = 0.9
TERMINAL_GRAY = 0.3
DIFFERENCE = INITIAL_GRAY - TERMINAL_GRAY


def run_experiments(instances, directory):
    profile_1 = performance.get_performance_profile(instances, CONFIG, performance.TYPE_1)
    profile_2 = performance.get_performance_profile(instances, CONFIG, performance.TYPE_2)
    profile_3 = performance.get_performance_profile(instances, CONFIG, performance.TYPE_3)
    average_intrinsic_values = utils.get_average_intrinsic_values(instances, INTRINSIC_VALUE_MULTIPLIER)

    projected_monitoring_losses = []
    nonmyopic_monitoring_losses = []
    myopic_monitoring_losses = []
    fixed_time_allocation_losses = []

    for instance in instances:
        print('Experiment: %s' % instance)

        qualities = instances[instance]['solution_qualities']
        estimated_qualities = instances[instance]['estimated_solution_qualities']

        plt, results = run_experiment(qualities, estimated_qualities, average_intrinsic_values, profile_1, profile_2, profile_3)

        filename = directory + '/' + instance + '.png'
        plt.savefig(filename)
        plt.close()

        projected_monitoring_losses.append(results['projected_monitoring_loss'])
        nonmyopic_monitoring_losses.append(results['nonmyopic_monitoring_loss'])
        myopic_monitoring_losses.append(results['myopic_monitoring_loss'])
        fixed_time_allocation_losses.append(results['fixed_time_allocation_loss'])

    print('Projected Monitoring Average Percent Error: %f%%' % np.average(projected_monitoring_losses))
    print('Nonmyopic Monitoring Average Percent Error: %f%%' % np.average(nonmyopic_monitoring_losses))
    print('Myopic Monitoring Average Percent Error: %f%%' % np.average(myopic_monitoring_losses))
    print('Fixed Time Allocation Average Percent Error: %f%%' % np.average(fixed_time_allocation_losses))


def run_experiment(qualities, estimated_qualities, average_intrinsic_values, profile_1, profile_2, profile_3):
    time_limit = len(qualities)
    steps = range(time_limit)

    intrinsic_values = computation.get_intrinsic_values(qualities, INTRINSIC_VALUE_MULTIPLIER)
    time_costs = computation.get_time_costs(steps, TIME_COST_MULTIPLIER)
    comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)

    estimated_intrinsic_values = computation.get_intrinsic_values(estimated_qualities, INTRINSIC_VALUE_MULTIPLIER)

    optimal_stopping_point = monitor.get_optimal_stopping_point(comprehensive_values)
    projected_stopping_point, projected_intrinsic_value_groups = monitor.get_projected_stopping_point(estimated_qualities, steps, time_limit, CONFIG)
    nonmyopic_stopping_point = monitor.get_nonmyopic_stopping_point(estimated_qualities, steps, profile_2, profile_3, time_limit, CONFIG)
    myopic_stopping_point = monitor.get_myopic_stopping_point(estimated_qualities, steps, profile_1, profile_3, time_limit, CONFIG)
    fixed_stopping_point = monitor.get_fixed_stopping_point(average_intrinsic_values, time_limit, CONFIG)

    optimal_value = comprehensive_values[optimal_stopping_point]
    projected_loss = utils.get_percent_error(optimal_value, comprehensive_values[projected_stopping_point])
    nonmyopic_loss = utils.get_percent_error(optimal_value, comprehensive_values[nonmyopic_stopping_point])
    myopic_loss = utils.get_percent_error(optimal_value, comprehensive_values[myopic_stopping_point])
    fixed_loss = utils.get_percent_error(optimal_value, comprehensive_values[fixed_stopping_point])

    results = {
        'projected_monitoring_loss': projected_loss,
        'nonmyopic_monitoring_loss':  nonmyopic_loss,
        'myopic_monitoring_loss': myopic_loss,
        'fixed_time_allocation_loss': fixed_loss
    }

    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.annotate('%d-TSP' % TOUR_SIZE, xy=(0, 0), xytext=(10, 172), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('%d Discrete Solution Qualities' % SOLUTION_QUALITY_CLASS_COUNT, xy=(0, 0), xytext=(10, 162), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$q(s) = Length_{MST} / Length(s)$', xy=(0, 0), xytext=(10, 152), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$U_C(t) = -e^{%.2ft}$' % TIME_COST_MULTIPLIER, xy=(0, 0), xytext=(10, 135), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$U_I(q) = %dq$' % INTRINSIC_VALUE_MULTIPLIER, xy=(0, 0), xytext=(10, 125), va='bottom', xycoords='axes fraction', textcoords='offset points')
    plt.annotate('$U(q, t) = U_C(t) - U_I(q)$', xy=(0, 0), xytext=(10, 115), va='bottom', xycoords='axes fraction', textcoords='offset points')

    plt.scatter(steps, intrinsic_values, color='g', zorder=3, label='Intrinsic Values')
    plt.scatter(steps, estimated_intrinsic_values, color='darkorange', zorder=3, label='Estimated Intrinsic Values')

    plt.plot(steps, average_intrinsic_values[:time_limit], color='b', label='Expected Performance Profile')
    plt.plot(steps, -time_costs, color='r', label='Cost of Time')
    plt.plot(steps, comprehensive_values, color='k', label='Comprehensive Values')

    plt.scatter([optimal_stopping_point], comprehensive_values[optimal_stopping_point], color='limegreen', zorder=4, label='Optimal Stopping Point')
    plt.scatter([projected_stopping_point], comprehensive_values[projected_stopping_point], color='m', zorder=4, label='Projected Stopping Point')
    plt.scatter([nonmyopic_stopping_point], comprehensive_values[nonmyopic_stopping_point], color='maroon', zorder=4, label='Nonmyopic Stopping Point')
    plt.scatter([myopic_stopping_point], comprehensive_values[myopic_stopping_point], color='y', zorder=4, label='Myopic Stopping Point')
    plt.scatter([fixed_stopping_point], comprehensive_values[fixed_stopping_point], color='c', zorder=4, label='Fixed Stopping Point')

    plt.annotate('%0.2f - Best Value' % comprehensive_values[optimal_stopping_point], xy=(0, 0), xytext=(10, 95), va='bottom', xycoords='axes fraction', textcoords='offset points', color='limegreen')
    plt.annotate('%0.2f - Best Value - Projected Monitoring' % comprehensive_values[projected_stopping_point], xy=(0, 0), xytext=(10, 85), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
    plt.annotate('%0.2f - Best Value - Nonmyopic Monitoring' % comprehensive_values[nonmyopic_stopping_point], xy=(0, 0), xytext=(10, 75), va='bottom', xycoords='axes fraction', textcoords='offset points', color='maroon')
    plt.annotate('%0.2f - Best Value - Myopic Monitoring' % comprehensive_values[myopic_stopping_point], xy=(0, 0), xytext=(10, 65), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')
    plt.annotate('%0.2f - Best Value - Fixed Time Allocation' % comprehensive_values[fixed_stopping_point], xy=(0, 0), xytext=(10, 55), va='bottom', xycoords='axes fraction', textcoords='offset points', color='c')

    decrement = DIFFERENCE / len(projected_intrinsic_value_groups)
    current_color = INITIAL_GRAY

    for projected_intrinsic_values in projected_intrinsic_value_groups:
        plt.plot(steps, projected_intrinsic_values, color=str(current_color))
        current_color -= decrement

    plt.annotate('%0.2f%% - Error - Projected Monitoring' % projected_loss, xy=(0, 0), xytext=(10, 35), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
    plt.annotate('%0.2f%% - Error - Nonmyopic Monitoring' % nonmyopic_loss, xy=(0, 0), xytext=(10, 25), va='bottom', xycoords='axes fraction', textcoords='offset points', color='maroon')
    plt.annotate('%0.2f%% - Error - Myopic Monitoring' % myopic_loss, xy=(0, 0), xytext=(10, 15), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')
    plt.annotate('%0.2f%% - Error - Fixed Time Allocation' % fixed_loss, xy=(0, 0), xytext=(10, 5), va='bottom', xycoords='axes fraction', textcoords='offset points', color='c')

    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, 0.102), loc=3, ncol=3, mode='expand', borderaxespad=0.0)

    return plt, results


def print_solution_quality_map(instances_filename, get_solution_qualities):
    solution_quality_map = {}

    with open(instances_filename) as f:
        for line in f.readlines():
            instance_filename, optimal_distance = utils.get_line_components(line)

            cities, start_city = tsp.load_instance(instance_filename)
            statistics = {'time': [], 'distances': []}
            tsp_solver.k_opt_solve(cities, start_city, statistics, 100)

            estimated_optimal_distance = tsp.get_mst_distance(start_city, cities)

            solution_quality_map[instance_filename] = {
                'solution_qualities': get_solution_qualities(statistics['distances'], optimal_distance),
                'estimated_solution_qualities': get_solution_qualities(statistics['distances'], estimated_optimal_distance)
            }

    print(json.dumps(solution_quality_map))


def get_statistics(instances):
    optimal_stopping_points = []

    for instance in instances:
        qualities = instances[instance]['solution_qualities']

        limit = len(qualities)
        steps = range(limit)

        intrinsic_values = computation.get_intrinsic_values(qualities, INTRINSIC_VALUE_MULTIPLIER)
        time_costs = computation.get_time_costs(steps, TIME_COST_MULTIPLIER)
        comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)

        stopping_point = monitor.get_optimal_stopping_point(comprehensive_values)
        optimal_stopping_points.append(stopping_point)

    sorted_stopping_points = sorted(optimal_stopping_points)
    fit = stats.norm.pdf(sorted_stopping_points, np.mean(sorted_stopping_points), np.std(sorted_stopping_points))
    pl.plot(sorted_stopping_points, fit, '-o')
    pl.hist(sorted_stopping_points, normed=True)
    pl.show()

    return {
        'mean': np.mean(optimal_stopping_points),
        'std': np.std(optimal_stopping_points),
        'variance': np.var(optimal_stopping_points),
        'median': np.median(optimal_stopping_points),
        'min': np.amin(optimal_stopping_points),
        'max': np.amax(optimal_stopping_points)
    }


def main():
    instances = utils.get_instance_map('maps/50-tsp-naive-solution-quality-map.json')
    run_experiments(instances, 'plots')


if __name__ == '__main__':
    main()
