import matplotlib.pyplot as plt
import numpy as np
import utils
import monitor
from scipy.optimize import curve_fit

INTRINSIC_VALUE_MULTIPLIER = 100
TIME_COST_MULTIPLIER = 0.035

instances = utils.get_instances('simulations/50-tsp-0.1s.json')

qualities = instances['instance-4']['qualities']
average_intrinsic_values = utils.get_average_intrinsic_values(instances, INTRINSIC_VALUE_MULTIPLIER)

TIME_LIMIT = len(qualities)
STEPS = range(TIME_LIMIT)

INTRINSIC_VALUES = average_intrinsic_values[:TIME_LIMIT]
TIME_COSTS = -np.exp(np.multiply(TIME_COST_MULTIPLIER, STEPS))
COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
HISTORY_THRESHOLD = 10

plt.figure(figsize=(7, 3.5))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
ax = plt.gca()
ax.spines['right'].set_visible(False)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
plt.rcParams['grid.linestyle'] = "-"
plt.grid(True)

# Value vs. Time Plot instance 21
# plt.xlabel('Time Steps')
# plt.ylabel('Solution Quality')

# optimal_stopping_point = monitor.get_optimal_stopping_point(COMPREHENSIVE_VALUES) 

# plt.plot(range(50), np.log(range(50)), color='green', linewidth=2)
# plt.plot(np.divide(STEPS, 10), TIME_COSTS, color='red', linewidth=2)
# plt.plot(np.divide(STEPS, 10), COMPREHENSIVE_VALUES, linewidth=2)
# plt.scatter([optimal_stopping_point / 10], COMPREHENSIVE_VALUES[optimal_stopping_point], color='orange', zorder=5)
# plt.annotate('intrinsic\nvalue function', xy=(0, 0), xytext=(60, 135), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('cost of time', xy=(0, 0), xytext=(303, 10), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(343, 64), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('optimal stopping\npoint', xy=(0, 0), xytext=(218, 105), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.tight_layout()
# plt.show()

# Performance History and Performance Projection Plot instance 4
# plt.xlabel('Time Steps')
# plt.ylabel('Solution Quality')

# plt.scatter(np.divide(STEPS[:HISTORY_THRESHOLD + 45][0::5], 10), qualities[:HISTORY_THRESHOLD + 45][0::5], color='blue', marker='v', zorder=5, s=10)
# plt.scatter(np.divide(STEPS[HISTORY_THRESHOLD + 45:TIME_LIMIT][0::5], 10), qualities[HISTORY_THRESHOLD + 45:TIME_LIMIT][0::5], color='g', marker='^', zorder=5, s=10)

# model = lambda x, a, b, c: a * np.arctan(x + b) + c
# skipper = 5
# start = 0.9
# end = 0
# changer = (start - end) / 3
# shade = start
# for i in range(3, 9, 2):
#     try:
#         shade -= changer
#         params, _ = curve_fit(model, STEPS[:HISTORY_THRESHOLD + skipper * i], qualities[:HISTORY_THRESHOLD + skipper * i])
#         projection = model(STEPS, params[0], params[1], params[2])
#         plt.scatter(np.divide(STEPS[HISTORY_THRESHOLD + skipper * i:][0::5], 10), projection[HISTORY_THRESHOLD + skipper * i:][0::5], color=str(shade), marker='s', zorder=2, s=7)
#     except:
#         pass

# plt.annotate('performance\nhistory', xy=(0, 0), xytext=(89, 67), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('performance\nprojections', xy=(0, 0), xytext=(197, 118), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('true\nperformance', xy=(0, 0), xytext=(330, 59), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.annotate(r'$\vec p^i$', xy=(0, 0), xytext=(383, 152), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', fontsize=10)
# plt.annotate(r'$\vec p^{i+1}$', xy=(0, 0), xytext=(388, 118), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', fontsize=10)
# plt.annotate(r'$\vec p^{i+2}$', xy=(0, 0), xytext=(388, 106), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', fontsize=10)
# plt.annotate(r'$\vec p^{*}$', xy=(0, 0), xytext=(383, 90), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', fontsize=10)

# plt.show()

# Myopic vs. Nonmyopic Stopping Condition
# plt.xlabel('Time Steps')
# plt.ylabel('Utility')

# INTRINSIC_VALUES[30:37] = list(reversed(np.arange(39, 43.381634140656061, 0.1)))[0:7]
# COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
# optimal_stopping_point = monitor.get_optimal_stopping_point(COMPREHENSIVE_VALUES) 

# COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS

# # plt.plot(np.divide(STEPS, 10), INTRINSIC_VALUES, color='green')
# # plt.plot(np.divide(STEPS, 10), TIME_COSTS, color='red')
# plt.plot(np.divide(STEPS, 10), COMPREHENSIVE_VALUES, linewidth=2)
# plt.scatter([optimal_stopping_point / 10], COMPREHENSIVE_VALUES[optimal_stopping_point], color='green', zorder=5)
# plt.scatter(2.9, 40.65, color='red', zorder=5)
# plt.annotate('myopic\nstopping point', xy=(0, 0), xytext=(64, 132), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('nonmyopic\nstopping point', xy=(0, 0), xytext=(203, 121), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('projected\ntime-dependent\nutility function', xy=(0, 0), xytext=(309, 9), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.show()

# plt.xlabel('Executions')
# plt.ylabel('Value')

# plt.scatter(range(21), np.log(range(21)), zorder=5, s=10, color='green')#, zorder=5, color='green', linewidth=2)
# plt.scatter(range(21, 41), np.log(range(10, 30)), zorder=3, s=10, color='red')#, color='red', linewidth=2, zorder=3)
# # plt.plot([70, 70], [100, 250], 'k-', lw=2)
# # plt.plot([20, 21], [np.log(20), np.log(10)], zorder=3, color='red', linewidth=2)
# plt.plot([20.5, 20.5], [0, np.log(20) * 1.2], zorder=2, color='0.2', linewidth=1.75, linestyle='--', dashes=(1, 1))
# # plt.plot([20] * 2, [0, 1], zorder=2, color='C0', linewidth=2)
# plt.annotate('50-TSP', xy=(0, 0), xytext=(80, 112), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('60-TSP', xy=(0, 0), xytext=(330, 115), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('problem\ntransition', xy=(0, 0), xytext=(228, 37), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.show()

# plt.xlabel('Steps')
# plt.ylabel('Solution Quality')

# plt.scatter(range(20), np.log(range(20)), zorder=5, s=10, color='green')
# plt.scatter(range(19, 34), (1/2) * np.log(range(15)) + np.log(19) + 0.25, zorder=3, s=10, color='red')
# plt.scatter(range(33, 43), (1/3) * np.log(range(10)) + np.log(19) + np.log(24) - 1.5, zorder=3, s=10, color='blue')
# plt.plot([19.5, 19.5], [0, 5.35], zorder=2, color='0.2', linewidth=1.75, linestyle='--', dashes=(1, 1))
# plt.plot([33.5, 33.5], [0, 5.35], zorder=2, color='0.2', linewidth=1.75, linestyle='--', dashes=(1, 1))

# plt.annotate('Lin-Kernighan\nHeuristic', xy=(0, 0), xytext=(83, 85), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('Simulated\nAnnealing', xy=(0, 0), xytext=(243, 83), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('Genetic\nAlgorithm', xy=(0, 0), xytext=(345, 112), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.plot([70, 70], [100, 250], 'k-', lw=2)
# plt.plot([20, 21], [np.log(20), np.log(10)], zorder=3, color='red', linewidth=2)

# plt.plot([20] * 2, [0, 1], zorder=2, color='C0', linewidth=2)
# plt.annotate('50-TSP', xy=(0, 0), xytext=(80, 112), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('60-TSP', xy=(0, 0), xytext=(330, 115), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('problem\ntransition', xy=(0, 0), xytext=(222, 37), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.show()

# plt.xlabel('Steps')
# plt.ylabel('Solution Quality')

# end = 2.5
# bins = 100
# plt.plot(np.linspace(0, end, bins), np.arctan(np.multiply(1.1, np.linspace(0, end, bins))), zorder=6,  linewidth=2, color='green')#, zorder=5, color='green', linewidth=2)
# plt.plot(np.linspace(0, end, bins), np.arctan(np.power(np.linspace(0, end, bins), 1.7)), zorder=5,  linewidth=2, color='red')
# plt.scatter([1.16], [0.91], color='blue', zorder=77)
# plt.scatter(range(21, 41), np.arctan(range(10, 30)), zorder=3, s=10, color='red')#, color='red', linewidth=2, zorder=3)
# plt.plot([70, 70], [100, 250], 'k-', lw=2)
# plt.plot([20, 21], [np.log(20), np.log(10)], zorder=3, color='red', linewidth=2)
# plt.plot([20.5, 20.5], [0, np.log(20) * 1.2], zorder=2, color='0.2', linewidth=1.75, linestyle='--', dashes=(1, 1))
# # plt.plot([20] * 2, [0, 1], zorder=2, color='C0', linewidth=2)
# plt.annotate('50-TSP', xy=(0, 0), xytext=(80, 112), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('60-TSP', xy=(0, 0), xytext=(330, 115), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('Weight 0.60', xy=(0, 0), xytext=(170, 55), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('Weight 0.75', xy=(0, 0), xytext=(61, 75), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('problem\ntransition', xy=(0, 0), xytext=(228, 37), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.show()

# plt.xlabel('Executions')
# plt.ylabel('Utility')

# plt.scatter(range(21), np.log(range(21)), zorder=5, s=10, color='green')#, zorder=5, color='green', linewidth=2)
# plt.scatter(range(21, 41), np.log(range(10, 30)), zorder=3, s=10, color='red')#, color='red', linewidth=2, zorder=3)
# # plt.plot([70, 70], [100, 250], 'k-', lw=2)
# # plt.plot([20, 21], [np.log(20), np.log(10)], zorder=3, color='red', linewidth=2)
# plt.plot([20.5, 20.5], [0, np.log(20) * 1.2], zorder=2, color='0.2', linewidth=1.75, linestyle='--', dashes=(1, 1))
# # plt.plot([20] * 2, [0, 1], zorder=2, color='C0', linewidth=2)
# plt.annotate('50-TSP', xy=(0, 0), xytext=(80, 112), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('60-TSP', xy=(0, 0), xytext=(330, 115), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('problem\ntransition', xy=(0, 0), xytext=(228, 37), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.show()

# x = [19.2126, 35.5453, 61.5723, 107.0217, 170.9356]
# y = [3, 3, 3.31174983333, 11.14225, 27.11834]
# z = np.add(x, y)

# p1 = plt.bar(range(5), x, 0.6, alpha=0.4, color='r', zorder=5)
# p2 = plt.bar(range(5), y, 0.6, bottom=x, alpha=0.4, color='b', zorder=5)

# def func(x, a, b):
#     return a * np.exp(b * x)

# popt, pcov = curve_fit(func, range(len(z)), z)

# curve = func(range(len(z)), *popt)

# plt.plot(range(len(curve)), curve, c='crimson', zorder=7)


# plt.ylabel('Time (mins)')
# plt.xticks(range(5), ('50-TSP', '60-TSP', '70-TSP', '80-TSP', '90-TSP'))
# plt.yticks(np.arange(0, 201, 50))
# plt.legend((p1[0], p2[0]), ('Performance Profile', 'Monitoring Policy'))
# plt.tight_layout()
# plt.show()

# IJCAI Presentation

# Performance Curve
plt.xlabel('Time Steps')
plt.ylabel('Solution Quality')
m = np.max(np.arctan(np.multiply(1/5, range(30))))
plt.plot(range(30), np.divide(np.arctan(np.multiply(1/5, range(30))), m), color='steelblue', linewidth=2)
plt.tight_layout()
plt.show()

# Meta-Level Control Problem
# plt.xlabel('Time Steps')
# plt.ylabel('Utility')
# time_steps = range(25)
# scale_down_factor = 1 / 4
# qualities = np.arctan(np.multiply(scale_down_factor, time_steps))
# intrinsic_values = np.multiply(100, qualities)
# cost_of_times = -np.exp(np.multiply(0.195, time_steps))
# comprehensive_values = np.add(intrinsic_values, cost_of_times)

# plt.yticks([-100, 0, 100])

# intrinsic_value_toggle = 1
# cost_of_time_toggle = 1
# comprehensive_value_toggle = 1
# optimal_stopping_point_toggle = 1

# plt.plot(time_steps, intrinsic_values, color='mediumseagreen', linewidth=2, alpha=intrinsic_value_toggle)
# plt.annotate('intrinsic\nvalue function', xy=(0, 0), xytext=(55, 117), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center', alpha=intrinsic_value_toggle)

# plt.plot(time_steps, cost_of_times, color='indianred', linewidth=2, alpha=cost_of_time_toggle)
# plt.annotate('cost of time', xy=(0, 0), xytext=(337, 6), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', alpha=cost_of_time_toggle)

# plt.plot(time_steps, comprehensive_values, color='steelblue', linewidth=2, alpha=comprehensive_value_toggle)
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(369, 48), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', alpha=comprehensive_value_toggle)

# optimal_stopping_point = monitor.get_optimal_stopping_point(comprehensive_values) 
# plt.scatter(optimal_stopping_point, comprehensive_values[optimal_stopping_point], color='orange', zorder=5, alpha=optimal_stopping_point_toggle)
# plt.annotate('optimal stopping\npoint', xy=(0, 0), xytext=(225, 90), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center', alpha=optimal_stopping_point_toggle)

# plt.tight_layout()
# plt.show()

# Online Performance Prediction
# plt.xlabel('Time Steps')
# plt.ylabel('Solution Quality')

# x = 15

# plt.annotate('performance\nhistory', xy=(0, 0), xytext=(55, 42), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.scatter(np.divide(STEPS[:HISTORY_THRESHOLD + x][0::5], 10), qualities[:HISTORY_THRESHOLD + x][0::5], color='steelblue', marker='v', zorder=5, s=10)

# plt.annotate('true\nperformance', xy=(0, 0), xytext=(330, 38), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.scatter(np.divide(STEPS[HISTORY_THRESHOLD + x:TIME_LIMIT][0::5], 10), qualities[HISTORY_THRESHOLD + x:TIME_LIMIT][0::5], color='mediumseagreen', marker='^', zorder=5, s=10)

# # plt.annotate('performance\nprojections', xy=(0, 0), xytext=(197, 99), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# model = lambda x, a, b, c: a * np.arctan(x + b) + c
# skipper = 5
# start = 0.9
# end = 0
# changer = (start - end) / 3
# shade = start
# # end_points 5 7 9
# for i in range(3, 9, 2):
#     try:
#         shade -= changer
#         params, _ = curve_fit(model, STEPS[:HISTORY_THRESHOLD + skipper * i], qualities[:HISTORY_THRESHOLD + skipper * i])
#         projection = model(STEPS, params[0], params[1], params[2])
#         plt.scatter(np.divide(STEPS[HISTORY_THRESHOLD + skipper * i:][0::5], 10), projection[HISTORY_THRESHOLD + skipper * i:][0::5], color=str(shade), marker='s', zorder=2, s=7, alpha=0)
#     except:
#         pass

# plt.xticks([0, 2, 4, 6, 8, 10, 12])
# plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
# plt.tight_layout()
# plt.show()

# x = [19.2126, 35.5453, 61.5723, 107.0217, 170.9356]
# y = [3, 3, 3.31174983333, 11.14225, 27.11834]
# z = np.add(x, y)

# p1 = plt.bar(range(5), z, 0.6, alpha=0.4, color='r', zorder=5)
# # p2 = plt.bar(range(5), y, 0.6, bottom=x, alpha=0.4, color='b', zorder=5)

# def func(x, a, b):
#     return a * np.exp(b * x)

# popt, pcov = curve_fit(func, range(len(z)), z)

# curve = func(range(len(z)), *popt)

# plt.plot(range(len(curve)), curve, c='crimson', zorder=7)


# plt.ylabel('Time (mins)')
# plt.xticks(range(5), ('50-TSP', '60-TSP', '70-TSP', '80-TSP', '90-TSP'))
# plt.yticks(np.arange(0, 201, 50))
# # plt.legend((p1[0], p2[0]), ('Performance Profile', 'Monitoring Policy'))
# plt.tight_layout()
# plt.show()