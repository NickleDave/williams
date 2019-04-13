import numpy as np


class MultiArmBandit:
    def __init__(self, p):
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError('values in p must be between 0 and 1')
        if p.ndim != 1:
            raise ValueError('number of dimensions of p must be 1 '
                             '(i.e. must be a vector)')
        self.p = p
        self.num_arms = p.shape[0]

    @staticmethod
    def isinteger(x):
        return np.equal(np.mod(x, 1), 0)

    def pull(self, arm):
        #         if np.any(arms < 0) or not np.all(self.isinteger(arms)):
        #             raise ValueError('arm must be a non-negative integer')
        #         if np.any(arms > self.num_arms):
        #             raise ValueError(f"Values in arms that are greater than "
        #                              f"number of 'arms' that MultiArmedBandit has, {self.num_arms}.")
        #         if arms.ndim != 1:
        #             raise ValueError('number of dimensions of arms must be 1 '
        #                              '(i.e. must be a vector)')
        p_arm = self.p[arm]
        r = np.ceil(p_arm - np.random.uniform(size=arm.shape)).astype(int)
        return r
