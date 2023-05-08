# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
       # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # Determining distance between the agent and the dot which is the greatest distance away
        ListofFood = newFood.asList()
        smallestDistanceToFood = -1
        for food in ListofFood:
            distance = util.manhattanDistance(newPos, food)
            if smallestDistanceToFood >= distance or smallestDistanceToFood == -1:
                smallestDistanceToFood = distance

        # Determining distance between packman and enemy, and also how close the other ghost is
        distanceToEnemy = 1
        closenessToEnemy = 0
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distanceToEnemy += distance
            if distance <= 1:
                closenessToEnemy += 1

        # calcunating to return what choice to make
        return successorGameState.getScore() + (1 / float(smallestDistanceToFood)) - (1 / float(distanceToEnemy)) - closenessToEnemy
        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth, gameState):
            # return the utility in case the defined depth is reached or the game is won/lost.
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:  # maximize for pacman
                return max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # minize for ghosts
                # calculate the next agent and increase depth accordingly.
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))

        """Performing maximize action for the root node i.e. pacman"""
        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            utility = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_fun(self, gameState, depth, agentIndex, alpha, beta):
        # The index for pacman is 0.
        pacmanIndex = 0
        # pacman_actions denotes the legal actions that it can take.
        pacman_actions = gameState.getLegalActions(pacmanIndex)
        # ubound denotes the negative infinity or high value for Minimax algorithm.
        ubound = -100000
        # Terminal test to Check if we have reach the cut-off state or leaf node.
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # Loop to generate successors.
        for action in pacman_actions:
            # Removing Directions.STOP from legal actions as given in question.
            if action != Directions.STOP:
                # Generate successor for the pacman using action from actions.
                next_node = gameState.generateSuccessor(
                    pacmanIndex, action)
                # Minimize next agent.
                ghostIndex = pacmanIndex+1
                value = self.min_fun(
                    next_node, depth, ghostIndex, alpha, beta)
                if value > beta:
                    # Update value to remove the unvisited branch of tree.
                    return value
                # Check if value is greater than negative infinity.
                if value > ubound:  # and action!= Directions.STOP:
                    # Update value of negative infinity
                    ubound = max(value, ubound)
                    # Update the action taken by max-player.
                    max_result = action
                # Update alpha as per algorithm
                alpha = max(alpha, ubound)
                # Return ation taken for depth being 1.
                # Else return the new value of negative infinity
        return (ubound, max_result)[depth == 1]

    # The min_fun take a gameState, depth of tree and agentIndex.
    # It computes the minimum value in AlphaBeta algorithm for min-player.

    def min_fun(self, gameState, depth, agentIndex, alpha, beta):
        # Ghost actions denotes legal action the ghost agent can take.
        ghost_actions = gameState.getLegalActions(agentIndex)
        # lbound denotes the positive inifinity value of MinMax algorithm.
        lbound = 100000
        # agent_count dentoes the total number of enemy agents in game.
        agent_count = gameState.getNumAgents()
        # Terminal test to check if the state is terminal state so as to cut-off.
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # Loop for every action in legal ghost/agent actions.
        for action in ghost_actions:
            # Remove action from legal actions according to question.
            if action != Directions.STOP:
                next_node = gameState.generateSuccessor(agentIndex, action)
                # Decrement the agent_count to check if ghost/agent left.
                if agentIndex == agent_count-1:
                    # Check if leaf node reached.
                    if depth == self.depth:
                        value = self.evaluationFunction(next_node)
                    # Else call max_fun to maximize value in next ply.
                    else:
                        pacmanIndex = 0
                        # Maximize for pacman.
                        value = self.max_fun(
                            next_node, depth+1, pacmanIndex, alpha, beta)
                else:
                    # For remaining ghosts, minimize the value.
                    value = self.min_fun(
                        next_node, depth, agentIndex+1, alpha, beta)
            # Update value to remove the unvisited branch of tree.
            if value < alpha:
                return value
            # Update the value of positive infinity
            if value < lbound:  # and action!= Directions.STOP:
                lbound = min(value, lbound)
                min_result = action
            # Update beta as per algorithm
            beta = min(beta, value)
        return lbound

    # The minmax function computes the action taken to maximize the value for max player.
    def minmax(self, gameState):
        depth = 0
        depth += 1
        pacmanIndex = 0
        max_result = self.max_fun(
            gameState, depth, pacmanIndex, -100000, 100000)
        return max_result

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.minmax(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Format of result = [action, score]
        action, score = self.get_value(gameState, 0, 0)

        return action

    def get_value(self, game_state, index, depth):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Expectation-agent
        """
        # Terminal states:
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(game_state)

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(game_state, index, depth)

        # Expectation-agent: Ghost has index > 0
        else:
            return self.expected_value(game_state, index, depth)

    def max_value(self, game_state, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = game_state.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            current_action, current_value = self.get_value(
                successor, successor_index, successor_depth)

            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_action, max_value

    def expected_value(self, game_state, index, depth):
        """
        Returns the max utility value-action for max-agent
        """
        legalMoves = game_state.getLegalActions(index)
        expected_value = 0
        expected_action = ""

        # Find the current successor's probability using a uniform distribution
        successor_probability = 1.0 / len(legalMoves)

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value = self.get_value(
                successor, successor_index, successor_depth)

            # Update expected_value with the current_value and successor_probability
            expected_value += successor_probability * current_value

        return expected_action, expected_value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Setup information to be used as arguments in evaluation function
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    capsule_count = len(currentGameState.getCapsules())
    closest_food = 1

    game_score = currentGameState.getScore()

    # Find distances from pacman to all food
    food_distances = [manhattanDistance(
        pacman_position, food_position) for food_position in food_list]

    # Set value for closest food if there is still food left
    if food_count > 0:
        closest_food = min(food_distances)

    # Find distances from pacman to ghost(s)
    for ghost_position in ghost_positions:
        ghost_distance = manhattanDistance(pacman_position, ghost_position)

        # If ghost is too close to pacman, prioritize escaping instead of eating the closest food
        # by resetting the value for closest distance to food
        if ghost_distance < 2:
            closest_food = 99999

    features = [1.0 / closest_food,
                game_score,
                food_count,
                capsule_count]

    weights = [10,
               200,
               -100,
               -10]

    # Linear combination of features
    return sum([feature * weight for feature, weight in zip(features, weights)])
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
