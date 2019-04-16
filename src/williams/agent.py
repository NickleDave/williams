import numpy as np


class Agent:
    def __init__(self, actions, net):
        self.actions = actions
        self.net = net

    def act(self, inputs=None):
        raise NotImplementedError


class EpsilonAgent(Agent):
    def __init__(self, actions, net, epsilon):
        super().__init__(actions, net)
        self.epsilon = epsilon

    def act(self, inputs=None):
        xi = np.random.uniform()
        if xi > self.epsilon:  # act greedily
            action = self.net.forward(input)
        else:  # explore
            action = np.random.choice(self.actions)
        return action
