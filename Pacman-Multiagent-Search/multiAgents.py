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
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self):
        self.succList = []

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        self.bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(self.bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        self.succList.insert(0, gameState.generatePacmanSuccessor(legalMoves[chosenIndex]).getPacmanPosition())
        if len(self.succList) > 4:
            self.succList.pop()
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
        "*** YOUR CODE HERE ***"
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
#         ghostDist = []
#         for index in range(len(newGhostStates)):
#             ghostDist += [util.manhattanDistance(newPos, newGhostStates[index].getPosition())]
#         foodDist = []
#         for food in newFood.asList():
#             foodDist += [util.manhattanDistance(newPos, food)]
#         inverseFoodDist = 0
#         if len(foodDist) > 0:
#             inverseFoodDist = 1 / float(min(foodDist))
#         return successorGameState.getScore() + (max(ghostDist) * ((inverseFoodDist) ** 2))
        closestFood = 0
        if newFood[newPos[0] - 1][newPos[1]]:
            closestFood += 2
        if newFood[newPos[0] + 1][newPos[1]]:
            closestFood += 2
        if newFood[newPos[0]][newPos[1] - 1]:
            closestFood += 2
        if newFood[newPos[0]][newPos[1] + 1]:
            closestFood += 2
            
        eatenFood = 0
        if (newPos[0], newPos[1]) in currentGameState.getFood():
            eatenFood = 10
            
        eatGhost = 0
        for index in range(len(newGhostStates)):
            dist = util.manhattanDistance(newPos, newGhostStates[index].getPosition())
            if dist <= newScaredTimes[index]:
                if dist != 0:
                    eatGhost += (1 / dist) * 100
            elif dist <= 3:
                if dist != 0:
                    eatGhost -= (1 / dist) * 100
        
        newStateScore = 0
        if newPos in self.succList:
            newStateScore = -30
            
        return successorGameState.getScore() + closestFood + eatenFood + eatGhost + newStateScore

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
#         if self.depth % 2 == 0:
#             value = self.evaluateDepth(gameState, self.depth)
#         else:
#             value = self.evaluateDepth(gameState, self.depth - 1)
        value = self.evaluateDepth(gameState, 0)
        return value[0]
    
    def evaluateDepth(self, gameState, counter):
#         if gameState.isWin() or gameState.isLose() or abs(counter) == abs(self.depth * gameState.getNumAgents()):
        if gameState.isWin() or gameState.isLose() or counter == self.depth * gameState.getNumAgents():
            return (None, self.evaluationFunction(gameState))
        if counter % gameState.getNumAgents() == 0:
            return self.maxFunction(gameState, counter)  # pacman : write Max function
        else:
            return self.minFunction(gameState, counter)  # ghost : write Min function
#         util.raiseNotDefined()
    
    def maxFunction(self, gameState, counter):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return(None, self.evaluationFunction(gameState))
        maxScore = (None, -float("inf"))
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.evaluateDepth(successor, counter + 1)
            if value[1] > maxScore[1]:
                maxScore = (action, value[1])
        return maxScore
    
    def minFunction(self, gameState, counter):
        actions = gameState.getLegalActions(counter % gameState.getNumAgents())
        if len(actions) == 0:
            return(None, self.evaluationFunction(gameState))
        minScore = (None, float("inf"))
        for action in actions:
#             successor = gameState.generateSuccessor(abs(counter % gameState.getNumAgents()), action)
            successor = gameState.generateSuccessor(counter % gameState.getNumAgents(), action)
            value = self.evaluateDepth(successor, counter + 1)
            if value[1] < minScore[1]:
                minScore = (action, value[1])
        return minScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        result = self.evaluateDepth(gameState, 0, -float("inf"), float("inf"))
        return result[0]
    
    def evaluateDepth(self, gameState, counter, alpha, beta):
            if gameState.isWin() or gameState.isLose() or (counter == self.depth * gameState.getNumAgents()):
                return (None, self.evaluationFunction(gameState))
            if counter % gameState.getNumAgents() == 0:
                return self.maxFunction(gameState, counter, alpha, beta)  # pacman : write Max function
            else:
                return self.minFunction(gameState, counter, alpha, beta)  # ghost : write Min function
#         util.raiseNotDefined()
    
    def maxFunction(self, gameState, counter, alpha, beta):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return(None, self.evaluationFunction(gameState))
        maxScore = (None, -float("inf"))
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.evaluateDepth(successor, counter + 1, alpha, beta)
            if value[1] > maxScore[1]:
                maxScore = (action, value[1])
            if maxScore[1] > beta:
                return maxScore
            alpha = max(alpha, maxScore[1])
        return maxScore
    
    def minFunction(self, gameState, counter, alpha, beta):
        actions = gameState.getLegalActions(counter % gameState.getNumAgents())
        if len(actions) == 0:
            return(None, self.evaluationFunction(gameState))
        minScore = (None, float("inf"))
        for action in actions:
            successor = gameState.generateSuccessor(counter % gameState.getNumAgents(), action)
            value = self.evaluateDepth(successor, counter + 1, alpha, beta)
            if value[1] < minScore[1]:
                minScore = (action, value[1])
            if minScore[1] < alpha:
                return minScore
            beta = min(beta, minScore[1])
        return minScore

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
#         util.raiseNotDefined()
        value = self.evaluateDepth(gameState, 0)
        return value[0]
    
    def evaluateDepth(self, gameState, counter):
#         if gameState.isWin() or gameState.isLose() or abs(counter) == abs(self.depth * gameState.getNumAgents()):
        if gameState.isWin() or gameState.isLose() or counter == self.depth * gameState.getNumAgents():
            return (None, self.evaluationFunction(gameState))
        if counter % gameState.getNumAgents() == 0:
            return self.maxFunction(gameState, counter)  # pacman : write Max function
        else:
            return self.expectimaxFunction(gameState, counter)  # ghost : write Min function
#         util.raiseNotDefined()
    
    def maxFunction(self, gameState, counter):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return(None, self.evaluationFunction(gameState))
        maxScore = (None, -float("inf"))
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.evaluateDepth(successor, counter + 1)
            if value[1] > maxScore[1]:
                maxScore = (action, value[1])
        return maxScore
    
    def expectimaxFunction(self, gameState, counter):
        actions = gameState.getLegalActions(counter % gameState.getNumAgents())
        if len(actions) == 0:
            return(None, self.evaluationFunction(gameState))
        exp_probability = 0
        for action in actions:
#             successor = gameState.generateSuccessor(abs(counter % gameState.getNumAgents()), action)
            successor = gameState.generateSuccessor(counter % gameState.getNumAgents(), action)
            value = self.evaluateDepth(successor, counter + 1)
            exp_probability += value[1] / (len(actions))
        return (None, exp_probability)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
#     successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  
      
    closestFood = 0
    if newFood[newPos[0] - 1][newPos[1]]:
        closestFood += 2
    if newFood[newPos[0] + 1][newPos[1]]:
        closestFood += 2
    if newFood[newPos[0]][newPos[1] - 1]:
        closestFood += 2
    if newFood[newPos[0]][newPos[1] + 1]:
        closestFood += 2
        
    eatenFood = 0
    if (newPos[0], newPos[1]) in currentGameState.getFood():
        eatenFood = 10
        
    eatGhost = 0
    for index in range(len(newGhostStates)):
        dist = util.manhattanDistance(newPos, newGhostStates[index].getPosition())
        if dist <= newScaredTimes[index]:
            if dist != 0:
                eatGhost += (1 / dist) * 100
        elif dist <= 3:
            if dist != 0:
                eatGhost -= (1 / dist) * 100
    return currentGameState.getScore() + closestFood + eatenFood + eatGhost

# Abbreviation
better = betterEvaluationFunction

