import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process, Manager
import time
import utils
import operator
import tsp
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import keras
# from keras.layers import Dense, Activation
# from keras.models import Sequential
# from keras import optimizers
# from keras import losses
from sklearn import svm
from sklearn import linear_model


SOLUTION_QUALITY_CLASS_COUNT = 200
SOLUTION_QUALITY_CLASS_BOUNDS = np.linspace(0, 1, SOLUTION_QUALITY_CLASS_COUNT)
SOLUTION_QUALITY_CLASSES = range(SOLUTION_QUALITY_CLASS_COUNT)

ACTIONS = [0, 1]

TIME_CLASSES = range(40)

EPISODES = 1000
SLEEP_INTERVAL = 0.5
LEARNING_RATE = 0.1

FUDGE = 0.00001


def get_S():
    return list(itertools.product(SOLUTION_QUALITY_CLASSES, TIME_CLASSES))


# def get_initial_Q_function():
#     return {s: {a: random.random() for a in ACTIONS} for s in get_S()}


def get_initial_Q_function():
    Q = linear_model.SGDRegressor(warm_start=True)
    Q.partial_fit([[0, 0, 0]], [[0]])
    return Q


# def get_initial_Q_function():
#     model = Sequential()
#     model.add(Dense(8, input_shape=(3,), activation="relu"))
#     model.add(Dense(1))
#     sgd = optimizers.SGD()
#     model.compile(optimizer=sgd, loss='mean_squared_error')
#     return model


# def get_pi(Q):
#     return {s: max(Q[s].items(), key=operator.itemgetter(1))[0] for s in get_S()}


def get_pi(Q):
    pi = {}

    for q_class, t_class in get_S():
        stop_value = Q.predict([[q_class, t_class, 0]])
        continue_value = Q.predict([[q_class, t_class, 1]])

        pi[(q_class, t_class)] = 0 if stop_value >= continue_value else 1

    return pi


# def get_pi(Q):
#     return {(q, t): max({a: Q.predict(np.array([q, t, a]).reshape(1, 3))[0][0] for a in ACTIONS}.items(), key=operator.itemgetter(1))[0] for q, t in get_S()}


def U(q, t):
    return 1000 * q - 0.01 * t


# def get_Q_values(Q, s):
#     return [Q[s][a] for a in Q[s]]


def get_Q_values(Q, s):
    q_class, t_class = s

    return [
        Q.predict([[q_class, t_class, 0]])[0],
        Q.predict([[q_class, t_class, 1]])[0]
    ]


# def get_Q_values(Q, s):
#     q, t = s

#     return [
#         Q.predict(np.array([q, t, 0]).reshape(1, 3)),
#         Q.predict(np.array([q, t, 1]).reshape(1, 3))
#     ]


def k_opt_solve(states, start_state, iterations, memory):
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
    stopping_times = []

    Q = get_initial_Q_function()
    pi = get_pi(Q)

    for episode in range(EPISODES):
        print("Episode %d" % episode)

        previous_q = 0
        start_t = time.time()

        memory = Manager().dict()
        q = memory['q'] = previous_q
        t = memory['t'] = start_t

        q_class = 0
        t_class = 0
        s = (q_class, t_class)
        a = memory['a'] = 1

        states = tsp.get_instance(50, 0, 10000, 1)
        start_state = list(states)[0]
        heuristic = tsp.get_mst_distance(start_state, states)

        process = Process(target=k_opt_solve, args=(states, start_state, 1000, memory))
        process.start()

        time.sleep(SLEEP_INTERVAL)

        while process.is_alive():
            print("a = %s, Q = %s, s = %s" % (a, str(get_Q_values(Q, s)), str(s)))

            next_q = heuristic / memory['q'] + FUDGE
            next_t = memory['t']

            next_q_class = utils.digitize(next_q, SOLUTION_QUALITY_CLASS_BOUNDS)
            next_t_class = round((next_t - start_t) / SLEEP_INTERVAL)
            next_s = (next_q_class, next_t_class)

            r = U(next_q_class, next_t_class) - U(q_class, t_class)
            next_Q_values = get_Q_values(Q, next_s)           

            # Q[s][a] += LEARNING_RATE * (r + max(next_Q_values) - Q[s][a])
            # pi[s] = max(Q[s].items(), key=operator.itemgetter(1))[0]
            
            x = [[q_class, t_class, a]]
            y = [r + max(next_Q_values)]
            Q.partial_fit(x, y)

            stop_value = Q.predict([[q_class, t_class, 0]])[0]
            continue_value = Q.predict([[q_class, t_class, 1]])[0]
            pi[s] = 0 if stop_value >= continue_value else 1

            # x = np.array([q_class, t_class, a]).reshape(1, 3)
            # y = np.array([r + max(next_Q_values)]).reshape(1, 1)
            # Q.train_on_batch(x, y)

            # stop_value = Q.predict(np.array([q_class, t_class, 0]).reshape(1, 3))
            # continue_value = Q.predict(np.array([q_class, t_class, 1]).reshape(1, 3))
            # pi[s] = 0 if stop_value >= continue_value else 1

            q = next_q
            t = next_t
            q_class = next_q_class
            t_class = next_t_class
            s = (next_q_class, next_t_class)

            a = pi[next_s] if random.random() > 0.1 else random.choice(ACTIONS)

            if a == 0:
                print("a = %s" % a)
                # Q[s][a] = 0
                process.terminate()
                stopping_times.append(t_class)
                break

            time.sleep(SLEEP_INTERVAL)

        print(np.mean(stopping_times[-10:]))

    return Q


def main():
    Q = get_Q_function()
    
if __name__ == "__main__":
    main()
