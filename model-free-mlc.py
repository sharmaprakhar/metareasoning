import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process, Manager
import time
import utils
import operator
import tsp
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
import pylab as pl

SOLUTION_QUALITY_CLASS_COUNT = 100 #20
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(0, 1, SOLUTION_QUALITY_CLASS_COUNT)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

TIME_CLASSES = range(20)
ACTIONS = [0, 1]

EPISODES = 1500
LEARNING_RATE = 0.01

SLEEP_INTERVAL = 0.5


def get_S():
    return list(itertools.product(SOLUTION_QUALITY_CLASSES, TIME_CLASSES))


def get_initial_Q_function():
    # return {s: {0: 0.1, 1: 0} for s in get_S()}
    return {s: {0: random.random(), 1: random.random()} for s in get_S()}


def get_pi(Q):
    return {s: max(Q[s].items(), key=operator.itemgetter(1))[0] for s in get_S()}


def U(q, t):
    return 1000 * q


def get_Q_values(Q, s):
    return [Q[s][a] for a in Q[s]]


def k_opt_tsp_solver(states, start_state, iterations, memory):
    tour = tsp.get_initial_random_tour(states, start_state)
    cities = tsp.get_swappable_cities(tour)
    distance = tsp.get_tour_distance(tour)

    for t in range(iterations):
        has_changed = False

        best_tour = tour
        best_distance = distance
        for first_key, second_key in itertools.combinations(cities, 2):
            current_tour = tsp.get_mutated_tour(tour, first_key, second_key)
            current_distance = tsp.get_tour_distance(current_tour)

            if current_distance < best_distance:
                best_tour = current_tour
                best_distance = current_distance

                has_changed = True

        tour = best_tour
        distance = best_distance

        memory['q'] = distance
        memory['t'] = time.time()

        if not has_changed:
            break

    return tour


def get_Q_function():
    # Create a list to keep track of the stopping points of each episode
    stopping_points = []
    values = []

    # Generate a random Q-value function and its associated policy
    Q = get_initial_Q_function()
    pi = get_pi(Q)

    # Learn the Q-value function for all episodes
    for episode in range(EPISODES):
        print('Episode %d' % episode)

        # Initialize the previous solution quality and computation time
        previous_q = 0
        start_t = time.time()

        # Initialize the dictionary passed between the meta-level controller and the TSP solver
        memory = Manager().dict()
        q = memory['q'] = previous_q
        t = memory['t'] = start_t

        # Initialize the current state and action
        q_class = 0
        t_class = 0
        s = (q_class, t_class)
        a = memory['a'] = 1

        # Initialize a random instance of a k-TSP
        states = tsp.get_instance(50, 0, 10000, 1)
        start_state = list(states)[0]

        # Calculate the heuristic to determine the solution quality at each time step
        heuristic = tsp.get_mst_distance(start_state, states)

        # Start the TSP solver
        process = Process(target=k_opt_tsp_solver, args=(states, start_state, 1000, memory))
        process.start()

        # Sleep for the sleep interval to let the anytime algorithm produce a solution
        time.sleep(SLEEP_INTERVAL)

        # Repeat the following until the anytime algorithm has been interrupted
        while process.is_alive():
            # Calculate the new solution quality and time
            next_q = heuristic / (memory['q'] + 0.00001)
            next_t = memory['t']

            # Set up the next state
            next_q_class = utils.digitize(next_q, SOLUTION_QUALITY_CLASS_BOUNDS)
            next_t_class = round((next_t - start_t) / SLEEP_INTERVAL)
            next_s = (next_q_class, next_t_class)
     
            # Update the Q-value function and the policy
            r = U(next_q_class, next_t_class) - U(q_class, t_class)
            next_Q_values = get_Q_values(Q, next_s)
            Q[s][a] += LEARNING_RATE * (r + max(next_Q_values) - Q[s][a])
            pi[s] = max(Q[s].items(), key=operator.itemgetter(1))[0]

            # Retrieve the action recommended by our exploration policy
            if episode < 1000:
                a = pi[next_s] if random.random() > 0.1 else random.choice(ACTIONS)
            else:
                a = pi[next_s]

            # Update the current state
            q = next_q
            t = next_t
            q_class = next_q_class
            t_class = next_t_class
            s = (next_q_class, next_t_class)

            # Stop the anytime algorithm accordingly
            if a == 0:
                # process.terminate()
                stopping_points.append(t_class)
                values.append(np.mean(stopping_points[-10:]))
                # break
            print((t_class, q_class, r))

            # Sleep to give the anytime algorithm time to compute more solutions
            time.sleep(SLEEP_INTERVAL)

        print(np.mean(stopping_points[-10:]))


    plt.figure()
    plt.title('Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.plot(range(EPISODES), values)
    plt.savefig('test.png')
    plt.close()

    return Q


def main():
    get_Q_function()


if __name__ == "__main__":
    main()
