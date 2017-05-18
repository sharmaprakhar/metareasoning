from utils import Node, get_children, get_solution, key, OpenList


def solve(problem, statistics):
    start_node = Node(problem.start_state)
    start_node_value = start_node.path_cost + problem.get_heuristic(start_node.state)

    open_list = OpenList()
    closed_list = set()

    open_list.add(start_node, start_node_value)

    while open_list:
        current_node = open_list.remove()
        current_node_key = key(current_node.state)

        statistics['expanded_nodes'] += 1

        if problem.is_goal(current_node.state):
            return get_solution(current_node)

        closed_list.add(current_node_key)

        for child_node in get_children(problem, current_node):
            child_node_key = key(child_node.state)

            if child_node_key not in closed_list and child_node not in open_list:
                value = child_node.path_cost + problem.get_heuristic(child_node.state)
                open_list.add(child_node, value)
            elif child_node in open_list:
                previous_child_node = open_list[child_node]

                previous_child_node_value = previous_child_node.path_cost + problem.get_heuristic(previous_child_node.state)
                child_node_value = child_node.path_cost + problem.get_heuristic(child_node.state)

                if child_node_value < previous_child_node_value:
                    del open_list[previous_child_node]
                    open_list.add(child_node, child_node_value)

    return None
