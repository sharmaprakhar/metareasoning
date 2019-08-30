import matplotlib.pyplot as plt
import numpy as np

import env
import fourier_agent
import table_agent
import utils

WINDOW_SIZE = 50
PLOT_WINDOW_SIZE = 300

PROBLEM_DIRECTORY = "problems/"
RESULTS_DIRECTORY = "statistics/"

PROBLEM_FILES = [("80-tsp.json", 1), ("90-tsp.json", 1)]

ALPHA = 200
BETA = 0.3

TRAINING_EPISODES = 5000
TRANSFER_EPISODES = 1000
TRANSFER_EPSILON = 0.1


def run_tabular_sarsa_experiments(transfer_episodes, transfer_epsilon, params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "tabular-sarsa-transfer-[{}]-[{}].json".format(params["alpha"], params["epsilon"])

    metareasoning_env = env.Environment(PROBLEM_DIRECTORY + PROBLEM_FILES[0][0], ALPHA, BETA, PROBLEM_FILES[0][1])
    prakhar = table_agent.Agent(metareasoning_env, params)
    prakhar.run_sarsa(statistics)

    for problem_file in PROBLEM_FILES[1:]:
        problem_file_path = PROBLEM_DIRECTORY + problem_file[0]
        increment = problem_file[1]

        params["episodes"] = transfer_episodes
        params["epsilon"] = transfer_epsilon

        print("Shifting to {} with [increment = {}]".format(problem_file_path, increment))
        metareasoning_env = env.Environment(problem_file_path, ALPHA, BETA, increment)
        prakhar = table_agent.Agent(metareasoning_env, params, prakhar.action_value_function)
        prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_tabular_q_learning_experiments(transfer_episodes, transfer_epsilon, params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "tabular-q-transfer-[{}]-[{}].json".format(params["alpha"], params["epsilon"])

    metareasoning_env = env.Environment(PROBLEM_DIRECTORY + PROBLEM_FILES[0][0], ALPHA, BETA, PROBLEM_FILES[0][1])
    prakhar = table_agent.Agent(metareasoning_env, params)
    prakhar.run_q_learning(statistics)

    for problem_file in PROBLEM_FILES[1:]:
        problem_file_path = PROBLEM_DIRECTORY + problem_file[0]
        increment = problem_file[1]

        params["episodes"] = transfer_episodes
        params["epsilon"] = transfer_epsilon

        print("Shifting to {} with [increment = {}]".format(problem_file_path, increment))
        metareasoning_env = env.Environment(problem_file_path, ALPHA, BETA, increment)
        prakhar = table_agent.Agent(metareasoning_env, params, prakhar.action_value_function)
        prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_fourier_sarsa_experiments(transfer_episodes, transfer_epsilon, params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "fourier-sarsa-transfer-[{}]-[{}]-[{}].json".format(params["alpha"], params["epsilon"], params["order"])

    print("Training on {} with [increment = {}]".format(PROBLEM_DIRECTORY + PROBLEM_FILES[0][0], PROBLEM_FILES[0][1]))
    metareasoning_env = env.Environment(PROBLEM_DIRECTORY + PROBLEM_FILES[0][0], ALPHA, BETA, PROBLEM_FILES[0][1])
    prakhar = fourier_agent.Agent(metareasoning_env, params)
    prakhar.run_sarsa(statistics)

    for problem_file in PROBLEM_FILES[1:]:
        problem_file_path = PROBLEM_DIRECTORY + problem_file[0]
        increment = problem_file[1]

        params["episodes"] = transfer_episodes
        params["epsilon"] = transfer_epsilon

        print("Shifting to {} with [increment = {}]".format(problem_file_path, increment))
        metareasoning_env = env.Environment(problem_file_path, ALPHA, BETA, increment)
        prakhar = fourier_agent.Agent(metareasoning_env, params, prakhar.function_approximator.weights, prakhar.function_approximator.action_value_function)
        prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_fourier_q_learning_experiments(transfer_episodes, transfer_epsilon, params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "fourier-q-transfer-[{}]-[{}]-[{}].json".format(params["alpha"], params["epsilon"], params["order"])

    print("Training on {} with [increment = {}]".format(PROBLEM_DIRECTORY + PROBLEM_FILES[0][0], PROBLEM_FILES[0][1]))
    metareasoning_env = env.Environment(PROBLEM_DIRECTORY + PROBLEM_FILES[0][0], ALPHA, BETA, PROBLEM_FILES[0][1])
    prakhar = fourier_agent.Agent(metareasoning_env, params)
    prakhar.run_q_learning(statistics)

    for problem_file in PROBLEM_FILES[1:]:
        problem_file_path = PROBLEM_DIRECTORY + problem_file[0]
        increment = problem_file[1]

        params["episodes"] = transfer_episodes
        params["epsilon"] = transfer_epsilon

        print("Shifting to {} with [increment = {}]".format(problem_file_path, increment))
        metareasoning_env = env.Environment(problem_file_path, ALPHA, BETA, increment)
        prakhar = fourier_agent.Agent(metareasoning_env, params, prakhar.function_approximator.weights, prakhar.function_approximator.action_value_function)
        prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run():
    print("Adaptation experiment with [alpha = {}, beta = {}]".format(ALPHA, BETA))

    tabular_sarsa_results = run_tabular_sarsa_experiments(TRANSFER_EPISODES, TRANSFER_EPSILON, {
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": TRAINING_EPISODES
    })
    print("Error: {} +/- {}".format(tabular_sarsa_results["mean_error"], tabular_sarsa_results["standard_deviation_error"]))

    tabular_q_learning_results = run_tabular_q_learning_experiments(TRANSFER_EPISODES, TRANSFER_EPSILON, {
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": TRAINING_EPISODES
    })
    print("Error: {} +/- {}".format(tabular_q_learning_results["mean_error"], tabular_q_learning_results["standard_deviation_error"]))

    fourier_sarsa_results = run_fourier_sarsa_experiments(TRANSFER_EPISODES, TRANSFER_EPSILON, {
        "alpha": 0.00001,
        "epsilon": 0.1,
        "order": 10,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": TRAINING_EPISODES
    })
    print("Error: {} +/- {}".format(fourier_sarsa_results["mean_error"], fourier_sarsa_results["standard_deviation_error"]))

    fourier_q_learning_results = run_fourier_q_learning_experiments(TRANSFER_EPISODES, TRANSFER_EPSILON, {
        "alpha": 0.00001,
        "epsilon": 0.1,
        "order": 10,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": TRAINING_EPISODES
    })
    print("Error: {} +/- {}".format(fourier_q_learning_results["mean_error"], fourier_q_learning_results["standard_deviation_error"]))


def plot():
    plt.figure(figsize=(7, 2.5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams['grid.linestyle'] = "-"
    plt.grid(True)
    plt.ylabel('Time (ms)')
    plt.xlabel('Degree')

    # plt.xticks(range(8), ('2', '3', '4', '5', '6', '7', '8', '9'))
    plt.xticks(range(4), ('2', '3', '4', '5'))

    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    a = np.multiply([0.033, 0.125, 0.183, 0.366], 1000)
    b = np.multiply([0.028, 0.061, 0.085, 0.134], 1000)
    # a = np.multiply([0.39, 0.113, 0.253, 0.839, 1.295, 2.553, 2.862, 3.189], 1000)
    # b = np.multiply([0.032, 0.057, 0.085, 0.136, 0.199, 0.333, 0.670, 0.941], 1000)

    p1 = plt.bar([x + 0.2 for x in range(8)], a, 0.4, alpha=0.6, color='indianred', zorder=5)
    p2 = plt.bar([x - 0.2 for x in range(8)], b, 0.4, alpha=0.6, color='steelblue', zorder=5)

    plt.legend((p1[0], p2[0]), ('Centralized', 'Distributed'), loc=2, fontsize="small")

    plt.tight_layout(True)
    plt.show()


def test():
    filenames = [
        "fourier-q-transfer-[1e-05]-[0.1]-[10]-40-50.json",
        "fourier-q-transfer-[1e-05]-[0.1]-[10]-50-60.json",
        "fourier-q-transfer-[1e-05]-[0.1]-[10]-60-70.json",
        "fourier-q-transfer-[1e-05]-[0.1]-[10]-70-80.json",
        "fourier-q-transfer-[1e-05]-[0.1]-[10]-80-90.json",
        "fourier-sarsa-transfer-[1e-05]-[0.1]-[10]-40-50.json",
        "fourier-sarsa-transfer-[1e-05]-[0.1]-[10]-50-60.json",
        "fourier-sarsa-transfer-[1e-05]-[0.1]-[10]-60-70.json",
        "fourier-sarsa-transfer-[1e-05]-[0.1]-[10]-70-80.json",
        "fourier-sarsa-transfer-[1e-05]-[0.1]-[10]-80-90.json",
    ]
    for filename in filenames:
        if filename.endswith(".json"):
            statistics = utils.load(RESULTS_DIRECTORY + filename)
            smoothed_errors = utils.get_smoothed_values(statistics["errors"][5000:], 100)
            print(utils.get_convergence_point(smoothed_errors, 0.01, 100), filename)


def main():
    run()


if __name__ == "__main__":
    main()
