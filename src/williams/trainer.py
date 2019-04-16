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

    def train(self, epochs):
        pbar = tqdm(range(epochs))
        original_w = np.copy(self.agent.net.w)

        mean_reward_epoch = []
        p_arm_epoch = []
        mean_trial_p_epoch = []
        below_max_epoch = []
        change_in_w_epoch = []

        for epoch in pbar:
            arms = self.agent.act()
            r = self.bandit.pull(arms)
            delta_ws = self.alpha * r * self.agent.net.e()
            for delta_w in delta_ws:
                self.agent.net.w += delta_w

            mean_reward = np.mean(r)
            uniq_arms, counts = np.unique(arms, return_counts=True)
            p_arm = np.zeros(shape=(self.bandit.num_arms,))
            for arm_id, count in zip(uniq_arms, counts):
                p_arm[arm_id] = count / np.sum(counts)
            mean_trial_p = np.mean(self.agent.net.p)
            below_max = self.max_reward_p - mean_reward
            change_in_w = np.sum(np.abs(original_w - self.agent.net.w))

            pbar.set_description(
                f'epoch {epoch}, below max:{below_max: 4.3f}, mn trial p:{mean_trial_p: 4.3f}, '
                f'change w:{change_in_w: 4.3f}, p(arm):{p_arm}, mean reward:{mean_reward: 4.3f}'
            )

            mean_reward_epoch.append(mean_reward)
            p_arm_epoch.append(p_arm)
            mean_trial_p_epoch.append(mean_trial_p)
            below_max_epoch.append(below_max)
            change_in_w_epoch.append(change_in_w)

        summary = {
            'mean_reward': mean_reward_epoch,
            'p_arm': p_arm_epoch,
            'mean_trial_p': mean_trial_p_epoch,
            'below_max': below_max_epoch
            'change_in_w': change_in_w_epoch
        }

        return summary

