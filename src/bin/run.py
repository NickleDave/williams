import numpy as np

from williams.unit import BernoulliLogisticUnit
from williams.bandit import MultiArmBandit
from williams.trainer import Trainer


def main():
    blu = BernoulliLogisticUnit(input_size=20)
    p_easy = np.asarray([0.1, 0.9])
    easy_bandit = MultiArmBandit(p=p_easy)
    trainer = Trainer(agent=blu, bandit=easy_bandit, alpha=0.01)
    trainer.train(epochs=10000, trials=100)


if __name__ == '__main__':
    main()
