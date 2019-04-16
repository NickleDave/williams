import unittest

import numpy as np

from williams.bandit import MultiArmBandit


class TestMultiArmBandit(unittest.TestCase):
    def test_pull(self):
        p_easy = np.asarray([0.45, 0.55])
        easy_bandit = MultiArmBandit(p=p_easy)
        arms_0 = np.asarray([0] * 1000000)
        arms_1 = np.asarray([1] * 1000000)
        for _ in range(10):
            r_0 = easy_bandit.pull(arms_0)
            r_1 = easy_bandit.pull(arms_1)
            r_pct_0 = np.sum(r_0 == 1) / np.shape(r_0)[0]
            self.assertTrue(
                np.allclose(p_easy[0], r_pct_0, atol=1e-3, rtol=1e-3)
            )
            r_pct_1 = np.sum(r_1 == 1) / np.shape(r_1)[0]
            self.assertTrue(
                np.allclose(p_easy[1], r_pct_1, atol=1e-3, rtol=1e-3)
            )


if __name__ == '__main__':
    unittest.main()
