import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

def plot_data(y_data, err=None, x_data=None):
	if not x_data:
		x_axis = range(len(y_data))
	# plt.errorbar(x_axis, y_data, yerr=err, linestyle='--')
	plt.errorbar(x_axis, y_data, yerr=err, ecolor='r')
	plt.title('PERFORMANCE')
	plt.xlabel('episodes')
	plt.ylabel('cumulative reward')
	plt.grid(True)
	plt.show()

def get_instances(filename):
    with open(filename) as file:
        return json.load(file)

def plot_mean(arr):
	# print(arr.shape)
	std = np.std(arr, axis=0, keepdims=True)
	mean = arr.mean(axis=0, keepdims=True)
	# print(mean.shape)
	mean=mean.reshape(mean.shape[1])
	std=std.reshape(std.shape[1])
	plot_data(mean, std)