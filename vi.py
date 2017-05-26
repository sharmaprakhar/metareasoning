def argmax(args, get_value):
    best_arg = args[0]
    best_value = float('-inf')

    for arg in args:
        value = get_value(arg)

        if value > best_value:
            best_arg, best_value = arg, value

    return best_arg


def get_actions(mdp, state):
    actions = []

    for action in mdp.actions:
        for next_state in mdp.states:
            if mdp.get_transition_probability(state, action, next_state) > 0:
                actions.append(action)
                break

    return actions


def get_initial_values(mdp, default_value=0):
    return {state: default_value for state in mdp.states}


def get_expected_action_value(mdp, state, action, values):
    expected_value = 0

    for next_state in mdp.states:
        expected_value += mdp.get_transition_probability(state, action, next_state) * values[next_state]

    return expected_value


def get_expected_action_values(mdp, state, values):
    return [get_expected_action_value(mdp, state, action, values) for action in get_actions(mdp, state)]


def get_optimal_action(mdp, state, values):
    def get_value(action):
        return get_expected_action_value(mdp, state, action, values)

    actions = get_actions(mdp, state)
    return argmax(actions, get_value)


def get_optimal_policy(mdp, values):
    return {state: get_optimal_action(mdp, state, values) for state in mdp.states}


def get_optimal_values(epsilon):
    values = {state: default_value for state in mdp.states}

    while True:
        new_values = values.copy()

        delta = 0

        for state in mdp.states:
            new_values[state] = mdp.get_reward(state) + mdp.gamma * max(get_expected_action_values(mdp, state, values))
            delta = max(delta, abs(new_values[state] - values[state]))
            values[state] = new_values[state]

        if delta < epsilon:
            return values


def solve(mdp, epsilon=0.001):
    values = get_optimal_values(mdp, epsilon)
    return values, get_optimal_policy(mdp, values)
