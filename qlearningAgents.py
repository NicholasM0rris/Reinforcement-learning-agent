# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent
    
    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update
      
    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.gamma (discount rate)
    
    Functions you should use
      - self.getLegalActions(state) 
        which returns legal actions
        for a state
  """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q = util.Counter()

    def getQValue(self, state, action):
        """
      Returns Q(state,action)    
      Should return 0.0 if we never seen
      a state or (state,action) tuple 
    """
        "*** YOUR CODE HERE ***"
        return self.q[(state, action)]

    def getValue(self, state):
        """
      Returns max_action Q(state,action)        
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        "*** YOUR CODE HERE ***"
        q_values = []
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0

        for anaction in actions:
            q_values.append(self.getQValue(state, anaction))

        return max(q_values)

    def getPolicy(self, state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return None
        q_max = None
        best_action = None
        # Iterate over legal actions, if QValue is better than q_max, make q_max the new value
        for anaction in actions:
            q_current = self.getQValue(state, anaction)
            if q_max is None or q_current > q_max:
                q_max = q_current
                best_action = anaction
        return best_action

    def getAction(self, state):
        """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.
    
      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        if not legalActions:
            return None
        # Pick random action epsilon fraction
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        # Otherwise pick best action
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a 
      state = action => nextState and reward transition.
      You should do your Q-Value update here
      
      NOTE: You should never call this function,
      it will be called on your behalf
    """
        "*** YOUR CODE HERE ***"
        self.q[state, action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
                    reward + self.gamma * self.getValue(nextState))
        # print('q', self.q[state, action])
        # print('qqq',self.q)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
    
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
     ApproximateQLearningAgent
     
     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        "*** YOUR CODE HERE ***"
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
       Should update your weights based on transition  
    """
        "*** YOUR CODE HERE ***"
        # difference = (r + gamma * max(Q(s', a')) - Q(s, a)
        difference = reward + self.gamma * self.getValue(nextState) - self.getValue(state)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            # w_i = w_i + alpha * difference * f_i(s, a)
            self.weights[feature] = self.weights[feature] + self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            pass
