import matplotlib.pyplot as plt

import env
import linear_agent
import table_agent
import utils

WINDOW_SIZE = 50
PLOT_WINDOW_SIZE = 350

PROBLEM_DIRECTORY = "problems/"
RESULTS_DIRECTORY = "statistics/"

PROBLEM_FILE = "200-qap.json"
ALPHA = 200
BETA = 0.3
INCREMENT = 50

PROBLEM_FILE_PATH = PROBLEM_DIRECTORY + PROBLEM_FILE


def run_tabular_sarsa_experiments(params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "tabular-sarsa-[{}]-[{}]-{}".format(params["alpha"], params["epsilon"], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = table_agent.Agent(metareasoning_env, params)
    prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_tabular_q_learning_experiments(params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "tabular-q-[{}]-[{}]-{}".format(params["alpha"], params["epsilon"], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = table_agent.Agent(metareasoning_env, params)
    prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_linear_sarsa_experiments(params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "linear-sarsa-[{}]-[{}]-[{}]-{}".format(params["alpha"], params["epsilon"], params["order"], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = linear_agent.Agent(metareasoning_env, params)
    prakhar.run_sarsa(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run_linear_q_learning_experiments(params):
    statistics = {"errors": [], "stopping_points": [], "utilities": []}

    filename = RESULTS_DIRECTORY + "linear-q-[{}]-[{}]-[{}]-{}".format(params["alpha"], params["epsilon"], params["order"], PROBLEM_FILE)

    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)
    prakhar = linear_agent.Agent(metareasoning_env, params)
    prakhar.run_q_learning(statistics)

    utils.save(filename, statistics)

    return utils.get_results(statistics["errors"], WINDOW_SIZE, PLOT_WINDOW_SIZE)


def run():
    print("Convergence experiment [{}] with [alpha = {}, beta = {}, increment = {}]".format(PROBLEM_FILE, ALPHA, BETA, INCREMENT))

    # tabular_sarsa_results = run_tabular_sarsa_experiments({
    #     "alpha": 0.1,
    #     "epsilon": 0.1,
    #     "gamma": 1.0,
    #     "decay": 0.999,
    #     "episodes": 5000
    # })
    # print("Error: {} +/- {}".format(tabular_sarsa_results["mean_error"], tabular_sarsa_results["standard_deviation_error"]))

    tabular_q_learning_results = run_tabular_q_learning_experiments({
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 1.0,
        "decay": 0.999,
        "episodes": 5000
    })
    print("Error: {} +/- {}".format(tabular_q_learning_results["mean_error"], tabular_q_learning_results["standard_deviation_error"]))

    # linear_sarsa_results = run_linear_sarsa_experiments({
    #     "alpha": 0.00001,
    #     "epsilon": 0.1,
    #     "order": 7,
    #     "gamma": 1.0,
    #     "decay": 0.999,
    #     "episodes": 5000
    # })
    # print("Error: {} +/- {}".format(linear_sarsa_results["mean_error"], linear_sarsa_results["standard_deviation_error"]))

    # linear_q_learning_results = run_linear_q_learning_experiments({
    #     "alpha": 0.00001,
    #     "epsilon": 0.1,
    #     "order": 7,
    #     "gamma": 1.0,
    #     "decay": 0.999,
    #     "episodes": 5000
    # })
    # print("Error: {} +/- {}".format(linear_q_learning_results["mean_error"], linear_q_learning_results["standard_deviation_error"]))

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

    p1 = plt.plot(range(len(tabular_sarsa_results["smoothed_errors"])), tabular_sarsa_results["smoothed_errors"], color="r")
    p2 = plt.plot(range(len(linear_sarsa_results["smoothed_errors"])), linear_sarsa_results["smoothed_errors"], color="b")
    p3 = plt.plot(range(len(tabular_q_learning_results["smoothed_errors"])), tabular_q_learning_results["smoothed_errors"], color="g")
    p4 = plt.plot(range(len(linear_q_learning_results["smoothed_errors"])), linear_q_learning_results["smoothed_errors"], color="y")

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ("SARSA(Table)", "SARSA(Fourier)", "Q-learning(Table)", "Q-learning(Fourier)"), loc=1, ncol=2)

    plt.tight_layout()
    plt.show()


def plot():
    figure = plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.ylabel("Utility")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    filename = "qap-learning-curve.pdf"
    tabular_sarsa_statistics = utils.load(RESULTS_DIRECTORY + "tabular-sarsa-[0.1]-[0.1]-150-qap.json")
    tabular_q_learning_statistics = utils.load(RESULTS_DIRECTORY + "tabular-q-[0.1]-[0.1]-150-qap.json")
    linear_sarsa_statistics = utils.load(RESULTS_DIRECTORY + "linear-sarsa-[1e-05]-[0.1]-[7]-150-qap.json")
    linear_q_learning_statistics = utils.load(RESULTS_DIRECTORY + "linear-q-[1e-05]-[0.1]-[7]-150-qap.json")

    tabular_sarsa_utitilities = utils.get_smoothed_values(tabular_sarsa_statistics["utilities"], PLOT_WINDOW_SIZE)
    linear_sarsa_utilities = utils.get_smoothed_values(linear_sarsa_statistics["utilities"], PLOT_WINDOW_SIZE)
    tabular_q_learning_utilities = utils.get_smoothed_values(tabular_q_learning_statistics["utilities"], PLOT_WINDOW_SIZE)
    linear_q_learning_utilities = utils.get_smoothed_values(linear_q_learning_statistics["utilities"], PLOT_WINDOW_SIZE)

    p1 = plt.plot(range(len(tabular_sarsa_utitilities)), tabular_sarsa_utitilities, color="khaki")
    p2 = plt.plot(range(len(tabular_q_learning_utilities)), tabular_q_learning_utilities, color="seagreen")
    p3 = plt.plot(range(len(linear_sarsa_utilities)), linear_sarsa_utilities, color="indianred")
    p4 = plt.plot(range(len(linear_q_learning_utilities)), linear_q_learning_utilities, color="steelblue")

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ("SARSA(Table)", "Q-learning(Table)", "SARSA(Fourier)", "Q-learning(Fourier)"), loc=4, fontsize="small")
    plt.tight_layout()


    figure.savefig(filename, bbox_inches="tight")
    # plt.show()


def test():
    metareasoning_env = env.Environment(PROBLEM_FILE_PATH, ALPHA, BETA, INCREMENT)

    quality, time = metareasoning_env.reset()
    qualities = [quality]

    utility = utils.get_time_dependent_utility(quality, time, ALPHA, BETA)
    utilities = [utility]

    while True:
        (quality, time), _, is_episode_done = metareasoning_env.step(metareasoning_env.CONTINUE_ACTION)

        qualities.append(quality)
        print(quality)
        print(time)
        utilities.append(utils.get_time_dependent_utility(quality, time, ALPHA, BETA))

        if is_episode_done:
            break

    figure = plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Steps")
    plt.ylabel("Qualities")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["top"].set_visible(False)
    # axis.set_xlim([0, 2 * utilities.index(max(utilities))])
    # axis.set_ylim([utilities[0], 1.05 * max(utilities)])

    plt.plot(range(len(utilities)), utilities, color="r")
    plt.tight_layout()
    plt.show()


def main():
    run()


if __name__ == "__main__":
    main()
