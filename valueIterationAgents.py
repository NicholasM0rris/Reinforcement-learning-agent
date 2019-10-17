# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import copy
import sys

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"
        for i in range(0, iterations):
            new_values = copy.copy(self.values)
            for s in mdp.getStates():

                if mdp.isTerminal(s):
                    new_values[s] = 0.0
                else:
                    values = []
                    actions = mdp.getPossibleActions(s)
                    for anaction in actions:
                        values.append(self.getQValue(s, anaction))
                    new_values[s] = max(values)
            self.values = new_values

    def getValue(self, state):
        """
      Return the value of the state (computed in __init__).
    """
        return self.values[state]

    def getQValue(self, state, action):

        """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
        "*** YOUR CODE HERE ***"
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        # print('transitions',transitions)
        # print('state', state)
        value = 0.0
        # Iterate over transitions and return the q value
        for atransition in transitions:
            reward = self.mdp.getReward(state, action, atransition[0])
            value += atransition[1] * (reward + self.discount * self.values[atransition[0]])
        return value

    def getPolicy(self, state):
        """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
        "*** YOUR CODE HERE ***"
        # Return None if there are no legal actions
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
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
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
