import argparse
import sys

import matplotlib.pyplot as plt

import env
import fourier_agent
import table_agent


def main():
    # TODO Add more accurate fields to each argument
    # TODO Add back in trials if necessary
    # TODO Should this really be a fork?
    parser = argparse.ArgumentParser(description="Run a reinforcement learning agent with a function approximation.")
    parser.add_argument("-method", type=str, required=True, help="The reinforcement learning agent")
    parser.add_argument("-problem", type=str, required=True, help="The problem file")
    parser.add_argument("-function", type=str, required=False, help="The function approximation")
    parser.add_argument("-order", type=int, required=False, help="The Fourier basis order")
    parser.add_argument("-gamma", type=float, required=True, help="The discount factor")
    parser.add_argument("-alpha", type=float, required=True, help="The learning rate")
    parser.add_argument("-epsilon", type=float, required=True, help="The initial epsilon-greedy probability")
    parser.add_argument("-decay", type=float, required=False, help="The epsilon-greedy probability decay")
    parser.add_argument("-episodes", type=int, required=True, help="The number of episodes")

    args = parser.parse_args()
    params = {
        "order": args.order,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "decay": args.decay,
        "episodes": args.episodes
    }

    anytime_algorithm_env = env.Environment(args.problem)

    statistics = {
        "errors": [],
        "smoothed_errors": [],
        "stopping_points": []
    }

    if not args.function:
        print("Using a table function...")
        prakhar = table_agent.Agent(params, anytime_algorithm_env)
    elif args.function == "fourier":
        print("Using the Fourier function approximation...")
        prakhar = fourier_agent.Agent(params, anytime_algorithm_env)
    else:
        print("Encountered an unrecognized function approximation:", args.function)
        sys.exit()

    if args.method == "q-learning":
        print("Running Q-learning on {}...".format(args.problem))
        prakhar.run_q_learning(statistics)
    elif args.method == "sarsa":
        print("Running SARSA on {}...".format(args.problem))
        prakhar.run_sarsa(statistics)
    else:
        print("Encountered an unrecognized reinforcement learning method:", args.method)
        sys.exit()

    print("Generating the learning curve...")
    plt.figure(figsize=(7, 3))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["grid.linestyle"] = "-"
    plt.xlabel("Episodes")
    plt.ylabel("Error (%)")
    plt.grid(True)

    axis = plt.gca()
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)

    plt.plot(range(args.episodes), statistics["smoothed_errors"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
