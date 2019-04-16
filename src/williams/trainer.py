from time import sleep

from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self,
                 agent,
                 bandit,
                 alpha):
        self.agent = agent
        self.bandit = bandit
        self.alpha = alpha
        self.max_reward_p = np.max(bandit.p)

    def train(self, epochs=100, trials=100):
        pbar = tqdm(range(epochs))
        original_w = np.copy(self.agent.w)
        for epoch in pbar:
            trial_r = []
            trial_arms = []
            trial_p = []
            for trial in range(trials):
                arms = self.agent.forward()
                trial_p.append(self.agent.p)
                trial_arms.append(arms)
                r = self.bandit.pull(arms)
                trial_r.append(r)
                delta_w = self.alpha * r * self.agent.e()
                self.agent.w += delta_w
            mean_reward = np.mean(trial_r)
            uniq_arms, counts = np.unique(trial_arms, return_counts=True)
            p_arm = np.zeros(shape=(self.bandit.num_arms,))
            for arm_id, count in zip(uniq_arms, counts):
                p_arm[arm_id] = count / np.sum(counts)
            mean_trial_p = np.mean(trial_p)
            below_max = self.max_reward_p - mean_reward
            change_in_w = np.sum(np.abs(original_w - self.agent.w))
            pbar.set_description(
                f'epoch {epoch}, below max:{below_max: 4.3f}, mn trial p:{mean_trial_p: 4.3f}, '
                f'change w:{change_in_w: 4.3f}, p(arm):{p_arm}, mean reward:{mean_reward: 4.3f}'
            )
            sleep(0.25)
