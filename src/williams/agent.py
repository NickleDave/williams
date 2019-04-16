import numpy as np


class Agent:
    def __init__(self, n_actions, net):
        """__init__ for Agent class

        Parameters
        ----------
        n_actions : int
            number of actions the agent can take
        net : class
            that implements a method 'forward' which accepts inputs with
            shape (batch size, vector length) and returns outputs of shape
            (batch size,), where each element in outputs is a member of the
            set defined by range(self.actions)
        """
        if type(n_actions) != int:
            raise TypeError(f'n_actions must be an integer but got type {type(n_actions)}')
        if n_actions < 2:
            raise ValueError(f'value of n_actions must be 2 or greater but got value {n_actions}')

        self.n_actions = n_actions
        self.actions_arr = np.arange(n_actions)
        self.net = net

    def act(self, inputs=None):
        """

        Parameters
        ----------
        inputs : numpy.ndarray
            input to network that selects actions, with shape (batch size, length).
            Default is None, in which case an input of np.ones(self.net.input_size)
            is forwarded through the unit. This is useful for environments where
            the only "input signal" is reward, e.g., non-associative learning tasks
            like a multi-arm bandit.

        Returns
        -------
        out : numpy.ndarray
            with shape (batch size,). A vector of actions to take where each action
            is represented by an integer and is in the set self.actions
        """
        if inputs is None:
            inputs = np.ones(shape=self.net.input_size)
        out = self.net.forward(inputs)
        return out


class EpsilonGreedyAgent(Agent):
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
