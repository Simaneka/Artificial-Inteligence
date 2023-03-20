# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # # create fringe to store nodes
    # fringe = util.Stack()
    # # track visited nodes
    # visited = []
    # # push initial state to fringe
    # fringe.push((problem.getStartState(), [], 1))

    # while not fringe.isEmpty():
    #     node = fringe.pop()
    #     state = node[0]
    #     actions = node[1]
    #     # visited node
    #     # goal check
    #     if problem.isGoalState(state):
    #         return actions
    #     if state not in visited:
    #         visited.append(state)
    #         # visit child nodes
    #         successors = problem.getSuccessors(state)
    #         for child in successors:
    #             # store state, action and cost = 1
    #             child_state = child[0]
    #             child_action = child[1]
    #             if child_state not in visited:
    #                 # add child nodes
    #                 child_action = actions + [child_action]
    #                 fringe.push((child_state, child_action, 1))

    stackDFS = util.Stack()
    done = set()  # store passed node
    startNode = (problem.getStartState(), 0, [])  # (node, cost, path)
    stackDFS.push(startNode)

    while not stackDFS.isEmpty():
        (node, cost, path) = stackDFS.pop()
        if problem.isGoalState(node):
            return path
        if not node in done:
            done.add(node)
            for next_node, next_action, next_cost in problem.getSuccessors(node):
                totalCost = cost + next_cost
                totalPath = path + [next_action]
                totalState = (next_node, totalCost, totalPath)
                stackDFS.push(totalState)

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # # create fringe to store nodes
    # fringe = util.Queue()
    # # track visited nodes
    # visited = []
    # # push initial state to fringe
    # fringe.push((problem.getStartState(), [], 1))

    # while not fringe.isEmpty():
    #     node = fringe.pop()
    #     state = node[0]
    #     actions = node[1]
    #     # goal check
    #     if problem.isGoalState(state):
    #         return actions
    #     if state not in visited:
    #         visited.append(state)
    #         # visit child nodes
    #         successors = problem.getSuccessors(state)
    #         for child in successors:
    #             # store state, action and cost = 1
    #             child_state = child[0]
    #             child_action = child[1]
    #             if child_state not in visited:
    #                 # add child nodes
    #                 child_action = actions + [child_action]
    #                 fringe.push((child_state, child_action, 1))

    queueBFS = util.Queue()
    done = set()  # store passed state
    # start contains node, cost, and path
    startNode = (problem.getStartState(), 0, [])
    queueBFS.push(startNode)

    while not queueBFS.isEmpty():
        (node, cost, path) = queueBFS.pop()
        if problem.isGoalState(node):
            return path
        if not node in done:
            done.add(node)
            for next_node, next_action, next_cost in problem.getSuccessors(node):
                totalCost = cost + next_cost
                totalPath = path + [next_action]
                totalState = (next_node, totalCost, totalPath)
                queueBFS.push(totalState)
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    priorityQueueUCS = util.PriorityQueue()
    done = set()  # store passed state
    # start contains node, cost, and path
    startNode = (problem.getStartState(), 0, [])
    priorityQueueUCS.push(startNode, 0)

    while not priorityQueueUCS .isEmpty():
        (node, cost, path) = priorityQueueUCS .pop()
        if problem.isGoalState(node):
            return path
        if not node in done:
            done.add(node)
            for next_node, next_action, next_cost in problem.getSuccessors(node):
                totalCost = cost + next_cost
                totalPath = path + [next_action]
                totalState = (next_node, totalCost, totalPath)
                priorityQueueUCS.push(totalState, totalCost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Use a priority queue, so the cost of actions is calculated with a provided heuristic
    fringe = util.PriorityQueue()
    # Make list of explored nodes  empty 
    visited = []
    # Make list of actions empty
    actionList = []
    # Place the starting point in the priority queue
    fringe.push((problem.getStartState(), actionList),
                heuristic(problem.getStartState(), problem))
    while fringe:
        node, actions = fringe.pop()
        if not node in visited:
            visited.append(node)
            if problem.isGoalState(node):
                return actions
            for successor in problem.getSuccessors(node):
                coordinate, direction, cost = successor
                nextActions = actions + [direction]
                nextCost = problem.getCostOfActions(nextActions) + \
                    heuristic(coordinate, problem)
                fringe.push((coordinate, nextActions), nextCost)
    return []

    util.raiseNotDefined()


# Abbreviations
bfs = BreadthFirstSearch
dfs = DepthFirstSearch
astar = A*Search
ucs = UniformCostSearch
