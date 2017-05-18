import numpy as np


class Problem(object):
    def __init__(self, start_state, is_goal, get_successors, get_cost, get_heuristic):
        self.start_state = np.copy(start_state)
        self.is_goal = is_goal
        self.get_successors = get_successors
        self.get_cost = get_cost
        self.get_heuristic = get_heuristic


class Node(object):
    def __init__(self, state, parent=None, path_cost=0, depth=0, action=None):
        self.state = np.copy(state)
        self.parent = parent
        self.path_cost = path_cost
        self.depth = depth
        self.action = action


class OpenList(object):
    def __init__(self):
        self.items = []
        self.cache = {}

    def add(self, node, value):
        self.items.append((value, node))
        self.items.sort(key=lambda item: item[0])
        self.cache[key(node.state)] = node

    def remove(self):
        node = self.items.pop(0)[1]
        return node

    def __len__(self):
        return len(self.items)

    def __contains__(self, new_node):
        return key(new_node.state) in self.cache

    def __getitem__(self, new_node):
        return self.cache[key(new_node.state)]

    def __delitem__(self, new_node):
        for i, (value, node) in enumerate(self.items):
            if key(new_node.state) == key(node.state):
                self.items.pop(i)

        del self.cache[key(new_node.state)]


def get_children(problem, parent):
    children = []

    for successor in problem.get_successors(parent.state):
        path_cost = parent.path_cost + problem.get_cost(parent.state, successor['action'], successor['state'])
        child = Node(successor['state'], parent, path_cost, parent.depth + 1, successor['action'])
        children.append(child)

    return children


def get_solution(node):
    if node.parent is None:
        return []
    return get_solution(node.parent) + [node.action]


def key(state):
    return str(state.tolist())
