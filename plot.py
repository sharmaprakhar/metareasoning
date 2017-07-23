import matplotlib.pyplot as plt
import numpy as np
import utils
import monitor
from scipy.optimize import curve_fit

INTRINSIC_VALUE_MULTIPLIER = 500
TIME_COST_MULTIPLIER = 0.11

instances = utils.get_instances('maps/50-tsp-naive-map.json')

qualities = instances['instance-20']['solution_qualities']
average_intrinsic_values = utils.get_average_intrinsic_values(instances, INTRINSIC_VALUE_MULTIPLIER)

TIME_LIMIT = 50
STEPS = range(TIME_LIMIT)
INTRINSIC_VALUES = average_intrinsic_values[:TIME_LIMIT]
TIME_COSTS = -np.exp(np.multiply(TIME_COST_MULTIPLIER, STEPS))
COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
HISTORY_THRESHOLD = 20

plt.figure(dpi=150)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
ax = plt.gca()
ax.spines['right'].set_visible(False)
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
# plt.annotate('intrinsic\nvalue function', xy=(0, 0), xytext=(320, 225), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('cost of time', xy=(0, 0), xytext=(285, 30), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(315, 125), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
# plt.annotate('optimal stopping\npoint', xy=(0, 0), xytext=(201, 198), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# Performance History and Performance Projection Plot
# plt.xlabel('Time')
# plt.ylabel('Solution Quality')

# steps = range(TIME_LIMIT)
# model = lambda x, a, b, c: a * np.arctan(x + b) + c
# params, _ = curve_fit(model, steps[0:HISTORY_THRESHOLD], qualities[0:HISTORY_THRESHOLD])
# projection = model(steps, params[0], params[1], params[2])

# plt.scatter(np.divide(steps[:HISTORY_THRESHOLD], 10), qualities[:HISTORY_THRESHOLD], color='blue', zorder=5)
# plt.scatter(np.divide(steps[HISTORY_THRESHOLD:], 10), projection[HISTORY_THRESHOLD:], color='red', zorder=5)
# plt.annotate('performance\nhistory', xy=(0, 0), xytext=(35, 105), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
# plt.annotate('performance\nprojection', xy=(0, 0), xytext=(313Therefore, a myopic estimate of the EVC is often used., 205), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

# Myopic vs. Nonmyopic Stopping Condition
plt.xlabel('Time')
plt.ylabel('Value')

INTRINSIC_VALUES[24:30] = 6 * [278.26668788996642]
print(INTRINSIC_VALUES)
COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS
optimal_stopping_point = monitor.get_optimal_stopping_point(COMPREHENSIVE_VALUES) 

COMPREHENSIVE_VALUES = INTRINSIC_VALUES + TIME_COSTS

# plt.plot(np.divide(STEPS, 10), INTRINSIC_VALUES, color='green')
# plt.plot(np.divide(STEPS, 10), TIME_COSTS, color='red')
plt.plot(np.divide(STEPS, 10), COMPREHENSIVE_VALUES)
plt.scatter([optimal_stopping_point / 10], COMPREHENSIVE_VALUES[optimal_stopping_point], color='green', zorder=5)
plt.scatter(2.30511, 265.298, color='red', zorder=5)
plt.annotate('myopic stopping point', xy=(0, 0), xytext=(110, 245), va='bottom', xycoords='axes fraction', textcoords='offset points',  ha='center')
plt.annotate('nonmyopic stopping point', xy=(0, 0), xytext=(285, 255), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')
plt.annotate('time-dependent\nutility function', xy=(0, 0), xytext=(290, 25), va='bottom', xycoords='axes fraction', textcoords='offset points', ha='center')

plt.show()