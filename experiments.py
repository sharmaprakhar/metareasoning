import matplotlib.pyplot as plt

import env
import table_agent as agent
import utils

CONVERGENCE_THRESHOLD = 0.0001
CONVERGENCE_PERIOD = 50

PROBLEM = "problems/tsp/70-tsp.json"
ALPHA = 200
BETA = 0.0175
INCREMENT = 5

PARAMS = {
    "order": 5,
    "gamma": 1.0,
    "alpha": 0.1,
    "epsilon": 0.2,
    "decay": 0.999,
    "episodes": 2000
}


def test():
    metareasoning_env = env.Environment(PROBLEM, ALPHA, BETA, INCREMENT)
    statistics = {
        "errors": [],
        "smoothed_errors": [],
        "stopping_points": [],
        "smoothed_stopping_points": []
    }

    prakhar = agent.Agent(PARAMS, metareasoning_env)
    prakhar.run_sarsa(statistics)

    data = statistics["smoothed_errors"]
    threshold_iterations = 0

    for i in range(len(data) - 1):
        difference = abs(data[i] - data[i - 1])

        if difference <= CONVERGENCE_THRESHOLD:
            threshold_iterations += 1
            if threshold_iterations >= CONVERGENCE_PERIOD:
                print({"episode": i, "error": data[i]})
                break
        else:
            threshold_iterations = 0

    print("Generating the learning curve...")
    fig = plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)

    axis1 = fig.add_subplot(111)
    axis1.plot(range(len(statistics["smoothed_errors"])), statistics["smoothed_errors"], color='b')
    axis1.set_ylabel('Error', color='b')

    axis2 = axis1.twinx()
    axis2.plot(range(len(statistics["smoothed_stopping_points"])), statistics["smoothed_stopping_points"], color='r')
    axis2.set_ylabel('Stopping Points', color='r')

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

    print("Generating the performance curve...")
    fig = plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Steps")
    plt.ylabel("Qualities")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)

    plt.plot(range(len(utilities)), utilities, color='r')
    plt.tight_layout()
    plt.show()


def main():
    test()
    # plot()


if __name__ == "__main__":
    main()
