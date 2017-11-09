import matplotlib.pyplot as plt
import numpy as np
import utils
import monitor
from scipy.optimize import curve_fit

# INTRINSIC_VALUE_MULTIPLIER = 100
# TIME_COST_MULTIPLIER = 0.025

# instances = utils.get_instances('simulations/50-tsp-0.1s.json')

# qualities = instances['instance-4']['qualities']
# average_intrinsic_values = utils.get_average_intrinsic_values(instances, INTRINSIC_VALUE_MULTIPLIER)

# TIME_LIMIT = len(qualities)
# STEPS = range(TIME_LIMIT)
# INTRINSIC_VALUES = average_intrinsic_values[:TIME_LIMIT]
# TIME_COSTS = -np.exp(np.multiply(TIME_COST_MULTIPLIER, STEPS))
# COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
# HISTORY_THRESHOLD = 10

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

# Value vs. Time Plot
# plt.xlabel('Time')
# plt.ylabel('Value')

# optimal_stopping_point = monitor.get_optimal_stopping_point(COMPREHENSIVE_VALUES) 

# plt.plot(np.divide(STEPS, 10), INTRINSIC_VALUES, color='green')
# plt.plot(np.divide(STEPS, 10), TIME_COSTS, color='red')
# plt.plot(np.divide(STEPS, 10), COMPREHENSIVE_VALUES)
# plt.scatter([optimal_stopping_point / 10], COMPREHENSIVE_VALUES[optimal_stopping_point], color='orange', zorder=5)
# plt.annotate('intrinsic\nvalue function', xy=(0, 0), xytext=(65, 165), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('cost of time', xy=(0, 0), xytext=(303, 18), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(343,78), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('optimal stopping\npoint', xy=(0, 0), xytext=(201, 133), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# Performance History and Performance Projection Plot
# plt.xlabel('Time')
# plt.ylabel('Solution Quality')


# plt.scatter(np.divide(STEPS[:HISTORY_THRESHOLD + 45], 10), qualities[:HISTORY_THRESHOLD + 45], color='blue', marker='D', zorder=5, s=2)
# plt.scatter(np.divide(STEPS[HISTORY_THRESHOLD + 45:TIME_LIMIT], 10), qualities[HISTORY_THRESHOLD + 45:TIME_LIMIT], color='g', zorder=5, s=2)

# model = lambda x, a, b, c: a * np.arctan(x + b) + c
# skipper = 5
# start = 0.9
# end = 0
# changer = (start - end) / (10 - 3)
# shade = start
# for i in range(4, 10):
#     try:
#         shade -= changer
#         params, _ = curve_fit(model, STEPS[:HISTORY_THRESHOLD + skipper * i], qualities[:HISTORY_THRESHOLD + skipper * i])
#         projection = model(STEPS, params[0], params[1], params[2])
#         plt.scatter(np.divide(STEPS[HISTORY_THRESHOLD + skipper * i:], 10), projection[HISTORY_THRESHOLD + skipper * i:], color=str(shade), zorder=2, s=2)
#     except:
#         pass

# plt.annotate('performance\nhistories', xy=(0, 0), xytext=(84, 71), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('performance\nprojections', xy=(0, 0), xytext=(194, 121), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('true\nperformance', xy=(0, 0), xytext=(330, 67), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# plt.show()

# Myopic vs. Nonmyopic Stopping Condition
# plt.xlabel('Time')
# plt.ylabel('Value')

# print(np.arange(40, 43.381634140656061, 0.1))
# INTRINSIC_VALUES[30:37] = list(reversed(np.arange(39, 43.381634140656061, 0.1)))[0:7]
# print(INTRINSIC_VALUES)
# COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
# optimal_stopping_point = monitor.get_optimal_stopping_point(COMPREHENSIVE_VALUES) 

# COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS

# # plt.plot(np.divide(STEPS, 10), INTRINSIC_VALUES, color='green')
# # plt.plot(np.divide(STEPS, 10), TIME_COSTS, color='red')
# plt.plot(np.divide(STEPS, 10), COMPREHENSIVE_VALUES)
# plt.scatter([optimal_stopping_point / 10], COMPREHENSIVE_VALUES[optimal_stopping_point], color='green', zorder=5)
# plt.scatter(2.9, 41.3, color='red', zorder=5)
# plt.annotate('myopic\nstopping point', xy=(0, 0), xytext=(47, 115), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('nonmyopic\nstopping point', xy=(0, 0), xytext=(200, 122), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(305, 9), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# Value vs. Time Plot
plt.xlabel('Trials')
plt.ylabel('Value')

plt.plot(range(21), np.log(range(21)), zorder=5, color='C0', s=2)
plt.plot(range(20, 40), np.log(range(10, 30)), color='C0', s=2)
# plt.plot([70, 70], [100, 250], 'k-', lw=2)
plt.plot([20] * 2, [np.log(20), np.log(10)], zorder=2, color='C0')

plt.show()