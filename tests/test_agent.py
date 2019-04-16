import unittest

from williams.units import BernoulliLogisticUnit
from williams.agent import Agent, EpsilonGreedyAgent


class TestAgent(unittest.TestCase):
    def test_init(self):
        batch_size = 100
        vector_size = 20
        input_size = (batch_size, vector_size)
        blu = BernoulliLogisticUnit(input_size=input_size)
        n_actions = 2
        agent = Agent(n_actions=n_actions, net=blu)
        self.assertTrue(agent.n_actions == n_actions)
        self.assertTrue(agent.net == blu)


class TestEpsilonGreedyAgent(unittest.TestCase):
    def test_init(self):
        batch_size = 100
        vector_size = 20
        input_size = (batch_size, vector_size)
        blu = BernoulliLogisticUnit(input_size=input_size)
        n_actions = 2
        epsilon = 0.1
        agent = EpsilonGreedyAgent(n_actions=n_actions, net=blu, epsilon=epsilon)
        self.assertTrue(agent.n_actions == n_actions)
        self.assertTrue(agent.net == blu)
        self.assertTrue(agent.epsilon == epsilon)


if __name__ == '__main__':
    unittest.main()
