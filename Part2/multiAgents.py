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
        # The method gets the legal moves for Pacman from the current game state using the getLegalActions method of the gameState object.

        Pacman_legal_moves = gameState.getLegalActions(self.PACMAN_INDEX)
        # The "value = -self.INF" method initializes the variable value to negative infinity, which will be used to keep track of the best value found so far.
        value = -self.INF
        for pacman_move in pacman_legal_moves:
            current_value = self._get_value(gameState.generateSuccessor(
                self.PACMAN_INDEX, pacman_move), 1, self.INIT_DEPTH)
            if value < current_value:
                value = current_value
                best_move = pacman_move
        return best_move

    inf = 100000000000
    Pacman_index = 0
    Initial_depth = 0

    # The "_getValue" method below is used to calculate the value of a given game state,
    #  taking into account the possible moves of both the maximizing and minimizing players.

    def _getValue(self, gameState, agent_index, current_search_depth):
        if self._check_is_terminal_state(gameState, current_search_depth):
            return self.evaluationFunction(gameState)
        elif agent_index == self.PACMAN_INDEX:
            return self._get_max_value(gameState, current_search_depth)
        else:
            return self._get_min_value(gameState, agent_index, current_search_depth)

    # The method "_check_terminal_state"  takes two parameters: gameState and current_search_depth
    # to check if current_search_depth is equal to the depth of the search tree.
    # If it is, then the method returns True to indicate that the current node is a terminal state.

    def _check_terminal_state(self, gameState, current_search_depth):
        if current_search_depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False

    def _get_successor_of_state(self, gameState, agent_index):
        agent_legal_moves = gameState.getLegalActions(agent_index)
        state_successors = []
        for agent_move in agent_legal_moves:
            state_successors.append(
                gameState.generateSuccessor(agent_index, agent_move))
        return state_successors

    def _get_maximum_value(self, gameState, search_depth):
        value = -self.INF
        pacman_successors = self._get_successor_of_state(
            gameState, self.Pacman_index)
        for successor in pacman_successors:
            # 1 is the index of the ghost
            value = max(value, self._get_value(successor, 1, search_depth))
        return value

    def _get_min_value(self, gameState, agent_index, search_depth):
        value = self.INF
        ghost_num = gameState.getNumAgents() - 1
        ghost_successors = self._get_successor_of_state(gameState, agent_index)
        if agent_index == ghost_num:
            for successor in ghost_successors:
                value = min(value, self._get_value(
                    successor, self.Pacman_index, search_depth + 1))     # min layer finished
        else:
            for successor in ghost_successors:
                value = min(value, self._get_value(
                    successor, agent_index + 1, search_depth))
        return value

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)

    """
    # AlphaBetaAgent class serves to implement the minimax algorithm with alpha-beta pruning so it efficiently search the game tree and
    # find the best action for the Pacman player while considering the moves of the ghost players.

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Format of result = [action, score]
        # Initial state: index = 0, depth = 0, alpha = -infinity, beta = +infinity

        result = self.getBestActionAndScore(
            game_state, 0, 0, float("-inf"), float("inf"))

        # Return the action from result
        return result[0]

    def getBestActionAndScore(self, game_state, index, depth, alpha, beta):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(game_state.getLegalActions(index)) == 0 or depth == self.depth:
            return "", game_state.getScore()

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(game_state, index, depth, alpha, beta)

        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(game_state, index, depth, alpha, beta)

    def max_value(self, game_state, index, depth, alpha, beta):
        """
        Returns the max utility action-score for max-agent with alpha-beta pruning
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

            # Calculate the action-score for the current successor
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Update max_value and max_action for maximizer agent
            if current_value > max_value:
                max_value = current_value
                max_action = action

            # Update alpha value for current maximizer
            alpha = max(alpha, max_value)

            # Pruning: Returns max_value because next possible max_value(s) of maximizer
            # can get worse for beta value of minimizer when coming back up
            if max_value > beta:
                return max_action, max_value

        return max_action, max_value

    def min_value(self, game_state, index, depth, alpha, beta):
        """
        Returns the min utility action-score for min-agent with alpha-beta pruning
        """
        legalMoves = game_state.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = game_state.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == game_state.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Update min_value and min_action for minimizer agent
            if current_value < min_value:
                min_value = current_value
                min_action = action

            # Update beta value for current minimizer
            beta = min(beta, min_value)

            # Pruning: Returns min_value because next possible min_value(s) of minimizer
            # can get worse for alpha value of maximizer when coming back up
            if min_value < alpha:
                return min_action, min_value

        return min_action, min_value
        util.raiseNotDefined()


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
