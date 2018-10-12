import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

def plot_data(y_data, x_data=None):
	if not x_data:
		x_axis = range(len(y_data))
	plt.plot(x_axis, y_data, linestyle='--')
	plt.title('SARSA')
	plt.xlabel('episodes')
	plt.ylabel('reward - single trial')
	plt.grid(True)
	plt.show()

def get_instances(filename):
    with open(filename) as file:
        return json.load(file)

