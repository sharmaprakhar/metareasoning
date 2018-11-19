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

def init_interface(numfeatures, numactions):
    lamda = .888
    #LSTD params
    w = np.zeros((numfeatures * numactions, 1))
    numTheta = numfeatures * numactions
    A = np.zeros((numfeatures + numTheta, numfeatures + numTheta))
    b = np.zeros((numfeatures + numTheta,1))
    z_stat=np.zeros((numfeatures + numTheta,1))
    theta = np.zeros((numfeatures * numactions, 1))
    # w = np.zeros((numfeatures * numactions, 1))
    return theta, lamda, A, b, z_stat

def getactionProbabilities(features, numactions, theta):
    # actionProbabilities = np.zeros((1,numactions))
    actionProbabilities = np.zeros((numactions))
    numfeatures = len(features)
    for a in range(numactions):
        actionProbabilities[a] = features.T.dot(theta[numfeatures*a : (numfeatures*a + numfeatures)])
    actionProbabilities_exp = np.exp(actionProbabilities)
    actionProbabilities_sum = np.sum(actionProbabilities_exp)
    actionProbabilities = actionProbabilities_exp/actionProbabilities_sum
    return actionProbabilities
    
def dlnpi(features, theta, numactions, action, numfeatures):
    action += 1
    actionProbabilities = getactionProbabilities(features, numactions, theta)
    result = np.zeros((1, (numactions * numfeatures))) #row vector
    for a in range(numactions):
        if a==action:
            result[0, numfeatures*action : (numfeatures*action + numfeatures)] = features.T * (1 - actionProbabilities[action])
        else: 
            result[0, numfeatures*a : (numfeatures*a + numfeatures)] = -1 * features.T * actionProbabilities[a]
    return result

def getAction(actionProbabilities, actions): #specific to three actions
    #probs = np.ndarray.flatten(actionProbabilities)
    action = np.random.choice(actions, p=actionProbabilities)
    return action