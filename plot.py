import matplotlib.pyplot as plt
import numpy as np
import utils
import monitor
from scipy.optimize import curve_fit

INTRINSIC_VALUE_MULTIPLIER = 100
TIME_COST_MULTIPLIER = 0.035

# instances = utils.get_instances('simulations/50-tsp-0.1s.json')

qualities = instances['instance-14']['qualities']
average_intrinsic_values = utils.get_average_intrinsic_values(instances, INTRINSIC_VALUE_MULTIPLIER)

TIME_LIMIT = len(qualities)
STEPS = range(TIME_LIMIT)
# STEPS = range(50)
INTRINSIC_VALUES = average_intrinsic_values[:TIME_LIMIT]
TIME_COSTS = -np.exp(np.multiply(TIME_COST_MULTIPLIER, STEPS))
COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
HISTORY_THRESHOLD = 10

plt.figure(figsize=(7, 3))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
plt.rcParams['grid.linestyle'] = "-"
plt.grid(True)

# Value vs. Time Plot instance 21
# plt.xlabel('Time Steps')
# plt.ylabel('Utility')

# optimal_stopping_point = monitor.get_optimal_stopping_point(COMPREHENSIVE_VALUES) 

# plt.plot(np.divide(STEPS, 10), INTRINSIC_VALUES, color='green', linewidth=2)
# plt.plot(np.divide(STEPS, 10), TIME_COSTS, color='red', linewidth=2)
# plt.plot(np.divide(STEPS, 10), COMPREHENSIVE_VALUES, linewidth=2)
# plt.scatter([optimal_stopping_point / 10], COMPREHENSIVE_VALUES[optimal_stopping_point], color='orange', zorder=5)
# plt.annotate('intrinsic\nvalue function', xy=(0, 0), xytext=(60, 135), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('cost of time', xy=(0, 0), xytext=(303, 10), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(343, 64), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('optimal stopping\npoint', xy=(0, 0), xytext=(218, 105), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
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

# # Myopic vs. Nonmyopic Stopping Condition
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
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(305, 9), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.show()

plt.xlabel('Executions')
plt.ylabel('Value')

plt.scatter(range(21), np.log(range(21)), zorder=5, s=10, color='green')#, zorder=5, color='green', linewidth=2)
plt.scatter(range(21, 41), np.log(range(10, 30)), zorder=3, s=10, color='red')#, color='red', linewidth=2, zorder=3)
# plt.plot([70, 70], [100, 250], 'k-', lw=2)
# plt.plot([20, 21], [np.log(20), np.log(10)], zorder=3, color='red', linewidth=2)
plt.plot([20.5, 20.5], [0, np.log(20) * 1.2], zorder=2, color='0.2', linewidth=1.75, linestyle='--', dashes=(1, 1))
# plt.plot([20] * 2, [0, 1], zorder=2, color='C0', linewidth=2)
plt.annotate('50-TSP', xy=(0, 0), xytext=(80, 112), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
plt.annotate('60-TSP', xy=(0, 0), xytext=(330, 115), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
plt.annotate('problem\ntransition', xy=(0, 0), xytext=(228, 37), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

plt.show()

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
