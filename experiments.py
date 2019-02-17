import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import env
import linear_agent
import table_agent
import utils

WINDOW_SIZE = 50
PLOT_WINDOW_SIZE = 200

PROBLEM_DIRECTORY = "problems/"
RESULTS_DIRECTORY = "results/"

PROBLEM_FILE = "50-tsp.json"
ALPHA = 200
BETA = 0.3
INCREMENT = 1

PROBLEM_FILE_PATH = PROBLEM_DIRECTORY + PROBLEM_FILE


def run_tabular_sarsa_experiments(params):
    statistics = {
        "errors": [],
        "stopping_points": [],
    }

    filename = RESULTS_DIRECTORY + "tabular-sarsa-[{}]-[{}]-{}".format(params['alpha'], params['epsilon'], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = table_agent.Agent(metareasoning_env, params)
    prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return {
        "mean": np.average(statistics["errors"][-WINDOW_SIZE:]),
        "standard_deviation": stats.sem(statistics["errors"][-WINDOW_SIZE:]),
        "smoothed_values": utils.get_smoothed_values(statistics["errors"], PLOT_WINDOW_SIZE)
    }


def run_tabular_q_learning_experiments(params):
    statistics = {
        "errors": [],
        "stopping_points": [],
    }

    filename = RESULTS_DIRECTORY + "tabular-q-[{}]-[{}]-{}".format(params['alpha'], params['epsilon'], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = table_agent.Agent(metareasoning_env, params)
    prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return {
        "mean": np.average(statistics["errors"][-WINDOW_SIZE:]),
        "standard_deviation": stats.sem(statistics["errors"][-WINDOW_SIZE:]),
        "smoothed_values": utils.get_smoothed_values(statistics["errors"], PLOT_WINDOW_SIZE)
    }


def run_linear_sarsa_experiments(params):
    statistics = {
        "errors": [],
        "stopping_points": [],
    }

    filename = RESULTS_DIRECTORY + "linear-sarsa-[{}]-[{}]-[{}]-{}".format(params['alpha'], params['epsilon'], params['order'], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = linear_agent.Agent(metareasoning_env, params)
    prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return {
        "mean": np.average(statistics["errors"][-WINDOW_SIZE:]),
        "standard_deviation": stats.sem(statistics["errors"][-WINDOW_SIZE:]),
        "smoothed_values": utils.get_smoothed_values(statistics["errors"], PLOT_WINDOW_SIZE)
    }


def run_linear_q_learning_experiments(params):
    statistics = {
        "errors": [],
        "stopping_points": [],
    }

    filename = RESULTS_DIRECTORY + "linear-q-[{}]-[{}]-[{}]-{}".format(params['alpha'], params['epsilon'], params['order'], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = linear_agent.Agent(metareasoning_env, params)
    prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return {
        "mean": np.average(statistics["errors"][-WINDOW_SIZE:]),
        "standard_deviation": stats.sem(statistics["errors"][-WINDOW_SIZE:]),
        "smoothed_values": utils.get_smoothed_values(statistics["errors"], PLOT_WINDOW_SIZE)
    }


def run():
    tabular_sarsa_data = run_tabular_sarsa_experiments({
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print({
        "mean": tabular_sarsa_data["mean"],
        "standard_deviation": tabular_sarsa_data["standard_deviation"]
    })

    tabular_q_learning_data = run_tabular_q_learning_experiments({
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print({
        "mean": tabular_q_learning_data["mean"],
        "standard_deviation": tabular_q_learning_data["standard_deviation"]
    })

    linear_sarsa_data = run_linear_sarsa_experiments({
        "alpha": 0.00001,
        "epsilon": 0.1,
        "order": 7,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print({
        "mean": linear_sarsa_data["mean"],
        "standard_deviation": linear_sarsa_data["standard_deviation"]
    })

    linear_q_learning_data = run_linear_q_learning_experiments({
        "alpha": 0.00001,
        "epsilon": 0.1,
        "order": 7,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print({
        "mean": linear_q_learning_data["mean"],
        "standard_deviation": linear_q_learning_data["standard_deviation"]
    })

    plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.ylabel("Error")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    p1 = plt.plot(range(len(tabular_sarsa_data["smoothed_values"])), tabular_sarsa_data["smoothed_values"], color="r")
    p2 = plt.plot(range(len(linear_sarsa_data["smoothed_values"])), linear_sarsa_data["smoothed_values"], color="b")
    p3 = plt.plot(range(len(tabular_q_learning_data["smoothed_values"])), tabular_q_learning_data["smoothed_values"], color="g")
    p4 = plt.plot(range(len(linear_q_learning_data["smoothed_values"])), linear_q_learning_data["smoothed_values"], color="y")

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('SARSA(Table)', 'SARSA(Fourier)', 'Q-learning(Table)', 'Q-learning(Fourier)'), loc=1, ncol=2)

    plt.tight_layout()
    plt.show()


def plot():
    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)

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
    run()


if __name__ == "__main__":
    main()
