import numpy as np

from williams.units.blu import BernoulliLogisticUnit
from williams.agent import Agent
from williams.bandit import MultiArmBandit
from williams.trainer import Trainer


def main():
    blu = BernoulliLogisticUnit(input_size=(100, 20))
    agent = Agent(n_actions=2, net=blu)
    p_easy = np.asarray([0.1, 0.9])
    easy_bandit = MultiArmBandit(p=p_easy)
    trainer = Trainer(agent=agent, bandit=easy_bandit, alpha=0.01)
    trainer.train(epochs=10000)


if __name__ == '__main__':
    main()
