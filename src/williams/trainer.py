import tqdm
import numpy as np


class Trainer:
    def __init__(self,
                 agent,
                 bandit,
                 alpha):
        self.agent = agent
        self.bandit = bandit
        self.alpha = alpha

    def train(self, epochs=100, trials=1000):
        for epoch in range(epochs):
            trial_r = []
            pbar = tqdm(range(trials))
            for trial in pbar:
                arms = self.agent.forward()
                r = self.bandit.pull(arms)
                trial_r.append(r)
                delta_w = self.alpha * r * self.agent.e()
                self.agent.w += delta_w
            pbar.update(f'epoch {epoch}, mean reward: {np.mean(trial_r)}')
