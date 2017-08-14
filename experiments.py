import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import scipy.stats as stats

import computation
import monitor
import performance
import tsp
import tsp_solver
import utils

TIME_COST_MULTIPLIER = 0.15
INTRINSIC_VALUE_MULTIPLIER = 200

SOLUTION_QUALITY_CLASS_COUNT = 20
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(0, 1, SOLUTION_QUALITY_CLASS_COUNT + 1)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

MONITOR_THRESHOLD = 30
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


def run_proposal_experiments(instances, directory):
    myopic_projected_monitoring_losses = []
    nonmyopic_projected_monitoring_losses = []

    for instance in instances:
        print('Experiment: %s' % instance)

        qualities = instances[instance]['qualities']
        estimated_qualities = instances[instance]['estimated_qualities']

        file_path = directory + '/' + instance + '.png'
        results = run_projected_monitoring_experiment(qualities, estimated_qualities, file_path)

        myopic_projected_monitoring_losses.append(results['myopic_projected_monitoring_loss'])
        nonmyopic_projected_monitoring_losses.append(results['nonmyopic_projected_monitoring_loss'])

    print('Nonmyopic Projected Monitoring Average Percent Error: %f%%' % np.average(nonmyopic_projected_monitoring_losses))
    print('Myopic Projected Monitoring Average Percent Error: %f%%' % np.average(myopic_projected_monitoring_losses))


def run_proposal_experiment(qualities, estimated_qualities, file_path):
    time_limit = len(qualities)
    steps = range(time_limit)

    estimated_intrinsic_values = computation.get_intrinsic_values(estimated_qualities, INTRINSIC_VALUE_MULTIPLIER)

    intrinsic_values = computation.get_intrinsic_values(qualities, INTRINSIC_VALUE_MULTIPLIER)
    time_costs = computation.get_time_costs(steps, TIME_COST_MULTIPLIER)
    comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)

    optimal_stopping_point = monitor.get_optimal_stopping_point(comprehensive_values)
    myopic_projected_stopping_point, myopic_projected_intrinsic_value_groups = monitor.get_myopic_projected_stopping_point(estimated_qualities, steps, time_limit, CONFIG)
    nonmyopic_projected_stopping_point, nonmyopic_projected_intrinsic_value_groups = monitor.get_nonmyopic_projected_stopping_point(estimated_qualities, steps, time_limit, CONFIG)

    optimal_value = comprehensive_values[optimal_stopping_point]
    myopic_projected_loss = utils.get_percent_error(optimal_value, comprehensive_values[myopic_projected_stopping_point])
    nonmyopic_projected_loss = utils.get_percent_error(optimal_value, comprehensive_values[nonmyopic_projected_stopping_point])

    results = {
        'myopic_projected_monitoring_loss': myopic_projected_loss,
        'nonmyopic_projected_monitoring_loss': nonmyopic_projected_loss
    }

    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    axes = plt.gca()
    axes.set_ylim(bottom=time_costs[-1] * -1.1, top=intrinsic_values[-1] * 1.1)

    plt.plot(steps, -time_costs, color='r', label='Cost of Time')
    plt.plot(steps, comprehensive_values, color='k', label='Comprehensive Values')
    plt.plot(steps, myopic_projected_intrinsic_value_groups[-1], color='m')
    plt.plot(steps, nonmyopic_projected_intrinsic_value_groups[-1], color='y')

    plt.scatter(steps, intrinsic_values, color='g', zorder=3, label='Intrinsic Values')
    plt.scatter(steps, estimated_intrinsic_values, color='darkorange', zorder=3, label='Estimated Intrinsic Values')
    plt.scatter([optimal_stopping_point], comprehensive_values[optimal_stopping_point], color='limegreen', zorder=4, label='Optimal Stopping Point')
    plt.scatter([myopic_projected_stopping_point], comprehensive_values[myopic_projected_stopping_point], color='m', zorder=4, label='Myopic Projected Stopping Point')
    plt.scatter([nonmyopic_projected_stopping_point], comprehensive_values[nonmyopic_projected_stopping_point], color='y', zorder=4, label='Nonmyopic Projected Stopping Point')

    plt.annotate('%0.2f - Best Value' % comprehensive_values[optimal_stopping_point], xy=(0, 0), xytext=(10, 160), va='bottom', xycoords='axes fraction', textcoords='offset points', color='limegreen')
    plt.annotate('%0.2f - Best Value - myopic Projected Monitoring' % comprehensive_values[myopic_projected_stopping_point], xy=(0, 0), xytext=(10, 150), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
    plt.annotate('%0.2f - Best Value - Nonmyopic Projected Monitoring' % comprehensive_values[nonmyopic_projected_stopping_point], xy=(0, 0), xytext=(10, 140), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')
    plt.annotate('%0.2f%% - Error - Myopic Projected Monitoring' % myopic_projected_loss, xy=(0, 0), xytext=(10, 110), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
    plt.annotate('%0.2f%% - Error - Nonmyopic Projected Monitoring' % nonmyopic_projected_loss, xy=(0, 0), xytext=(10, 100), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')

    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, 0.102), loc=3, ncol=3, mode='expand', borderaxespad=0.0)

    plt.savefig(file_path)
    plt.close()

    return results


def run_benchmark_experiments(instances, directory):
    average_intrinsic_values = utils.get_average_intrinsic_values(instances, INTRINSIC_VALUE_MULTIPLIER)

    profile_1 = performance.get_dynamic_performance_profile(instances, CONFIG, performance.TYPE_1)
    profile_2 = performance.get_dynamic_performance_profile(instances, CONFIG, performance.TYPE_2)
    profile_3 = performance.get_dynamic_performance_profile(instances, CONFIG, performance.TYPE_3)

    myopic_monitoring_losses = []
    nonmyopic_monitoring_losses = []

    for instance in instances:
        print('Experiment: %s' % instance)

        qualities = instances[instance]['qualities']
        estimated_qualities = instances[instance]['estimated_qualities']

        file_path = directory + '/' + instance + '.png'
        results = run_benchmark_experiment(qualities, estimated_qualities, average_intrinsic_values, profile_1, profile_2, profile_3, file_path)

        myopic_monitoring_losses.append(results['myopic_monitoring_loss'])
        nonmyopic_monitoring_losses.append(results['nonmyopic_monitoring_loss'])

    print('Nonmyopic Monitoring Average Percent Error: %f%%' % np.average(nonmyopic_monitoring_losses))
    print('Myopic Monitoring Average Percent Error: %f%%' % np.average(myopic_monitoring_losses))


def run_benchmark_experiment(qualities, estimated_qualities, average_intrinsic_values, profile_1, profile_2, profile_3, file_path):
    time_limit = len(qualities)
    steps = range(time_limit)

    estimated_intrinsic_values = computation.get_intrinsic_values(estimated_qualities, INTRINSIC_VALUE_MULTIPLIER)

    intrinsic_values = computation.get_intrinsic_values(qualities, INTRINSIC_VALUE_MULTIPLIER)
    time_costs = computation.get_time_costs(steps, TIME_COST_MULTIPLIER)
    comprehensive_values = computation.get_comprehensive_values(intrinsic_values, time_costs)

    optimal_stopping_point = monitor.get_optimal_stopping_point(comprehensive_values)
    myopic_stopping_point = monitor.get_myopic_stopping_point(estimated_qualities, steps, profile_1, profile_3, time_limit, CONFIG)
    nonmyopic_stopping_point = monitor.get_nonmyopic_stopping_point(estimated_qualities, steps, profile_2, profile_3, time_limit, CONFIG)

    optimal_value = comprehensive_values[optimal_stopping_point]
    myopic_loss = utils.get_percent_error(optimal_value, comprehensive_values[myopic_stopping_point])
    nonmyopic_loss = utils.get_percent_error(optimal_value, comprehensive_values[nonmyopic_stopping_point])

    results = {
        'myopic_monitoring_loss': myopic_loss,
        'nonmyopic_monitoring_loss':  nonmyopic_loss
    }

    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Performance Profile')
    plt.xlabel('Time')
    plt.ylabel('Value')

    axes = plt.gca()
    axes.set_ylim(bottom=time_costs[-1] * -1.1, top=intrinsic_values[-1] * 1.1)

    plt.plot(steps, average_intrinsic_values[:time_limit], color='b', label='Expected Performance Profile')
    plt.plot(steps, -time_costs, color='r', label='Cost of Time')
    plt.plot(steps, comprehensive_values, color='k', label='Comprehensive Values')

    plt.scatter(steps, intrinsic_values, color='g', zorder=3, label='Intrinsic Values')
    plt.scatter(steps, estimated_intrinsic_values, color='darkorange', zorder=3, label='Estimated Intrinsic Values')
    plt.scatter([optimal_stopping_point], comprehensive_values[optimal_stopping_point], color='limegreen', zorder=4, label='Optimal Stopping Point')
    plt.scatter([myopic_stopping_point], comprehensive_values[myopic_stopping_point], color='m', zorder=4, label='Myopic Stopping Point')
    plt.scatter([nonmyopic_stopping_point], comprehensive_values[nonmyopic_stopping_point], color='y', zorder=4, label='Nonmyopic Stopping Point')

    plt.annotate('%0.2f - Best Value' % comprehensive_values[optimal_stopping_point], xy=(0, 0), xytext=(10, 160), va='bottom', xycoords='axes fraction', textcoords='offset points', color='limegreen')
    plt.annotate('%0.2f - Best Value - Myopic Monitoring' % comprehensive_values[myopic_stopping_point], xy=(0, 0), xytext=(10, 140), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')
    plt.annotate('%0.2f - Best Value - Nonmyopic Monitoring' % comprehensive_values[nonmyopic_stopping_point], xy=(0, 0), xytext=(10, 150), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')
    plt.annotate('%0.2f%% - Error - Myopic Monitoring' % myopic_loss, xy=(0, 0), xytext=(10, 100), va='bottom', xycoords='axes fraction', textcoords='offset points', color='m')    
    plt.annotate('%0.2f%% - Error - Nonmyopic Monitoring' % nonmyopic_loss, xy=(0, 0), xytext=(10, 110), va='bottom', xycoords='axes fraction', textcoords='offset points', color='y')

    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, 0.102), loc=3, ncol=3, mode='expand', borderaxespad=0.0)

    plt.savefig(file_path)
    plt.close()

    return results


def main():
    instances = utils.get_instances('simulations/30-tsp-0.1s.json')
    run_benchmark_experiments(instances, 'plots')
    # run_proposal_experiments(instances, 'plots')


if __name__ == '__main__':
    main()
