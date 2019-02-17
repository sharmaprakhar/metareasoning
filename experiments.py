import matplotlib.pyplot as plt
import numpy as np

import env
import linear_agent as agent
import utils

CONVERGENCE_THRESHOLD = 0.001
CONVERGENCE_PERIOD = 50

PROBLEM = "problems/tsp/50-tsp.json"
ALPHA = 200
BETA = 0.3
INCREMENT = 1

PARAMS = {
    "alpha": 0.00001,
    "epsilon": 0.1,
    "order": 5,
    "gamma": 1.0,
    "decay": 0.999,
    "episodes": 5000
}


def get_results(data):
    threshold_iterations = 0

    for i in range(len(data) - 1):
        difference = abs(data[i] - data[i - 1])

        if difference <= CONVERGENCE_THRESHOLD:
            threshold_iterations += 1
            if threshold_iterations >= CONVERGENCE_PERIOD:
                return {"episode": i, "error": data[i]}
        else:
            threshold_iterations = 0


def test():
    statistics = {
        "errors": [],
        "smoothed_errors": [],
        "stopping_points": [],
        "smoothed_stopping_points": []
    }

    metareasoning_env = env.Environment(PROBLEM, ALPHA, BETA, INCREMENT)
    prakhar = agent.Agent(PARAMS, metareasoning_env)
    prakhar.run_sarsa(statistics)

    fig = plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)

    axis1 = fig.add_subplot(1, 1, 1)
    # axis1.plot(range(50), sorted(statistics["errors"][-50:]), color="b")
    axis1.plot(range(len(statistics["smoothed_errors"])), statistics["smoothed_errors"], color="b")
    axis1.set_ylabel("Error", color="b")

    print(np.average(statistics["errors"][-50:]))

    plt.tight_layout()
    plt.show()


def plot():
    metareasoning_env = env.Environment(PROBLEM, ALPHA, BETA, INCREMENT)

    quality, time = metareasoning_env.reset()
    qualities = [quality]

    utility = utils.get_time_dependent_utility(quality, time, ALPHA, BETA)
    utilities = [utility]

    while True:
        (quality, time), _, is_episode_done = metareasoning_env.step(metareasoning_env.CONTINUE_ACTION)

        qualities.append(quality)
        utilities.append(utils.get_time_dependent_utility(quality, time, ALPHA, BETA))

        if is_episode_done:
            break

    plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Steps")
    plt.ylabel("Qualities")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    axis.set_xlim([0, 2 * utilities.index(max(utilities))])
    axis.set_ylim([utilities[0], 1.05 * max(utilities)])

    plt.plot(range(len(utilities)), utilities, color="r")
    plt.tight_layout()
    plt.show()


def main():
    test()


if __name__ == "__main__":
    main()
