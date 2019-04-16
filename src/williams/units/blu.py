"""Bernoulli Logistic Unit, as described in Williams 1992"""
import numpy as np
from scipy.special import expit as logistic

from .ssu import StochasticSemilinearUnit


class BernoulliLogisticUnit(StochasticSemilinearUnit):
    """Bernoulli Logistic Unit, as described in Williams 1992

    Attributes
    ----------
    x : numpy.ndarray
        inputs
    w : numpy.ndarray
        weights
    s : numpy.ndarray
        dot product of x and w
    f : logistic
        scipy.special.expit, used as squashing function applied to s to get p
    p : numpy.ndarray
        parameter for Bernoulli distribution, computed as self.f(self.s)
    y : numpy.ndarray
        outputs

    Methods
    -------
    forward
        "forward pass" through unit; accepts inputs, returns outputs
    sample
        static method that accepts parameter p and returns samples from Bernoulli distribution
    f
        squashing function, for this unit it is logistic, i.e. scipy.special.expit
    e
        characteristic eligibility, as derived in Williams 1992
    reset
        resets following attributes to zero: x, s, p, and y
    """
    def __init__(self,
                 input_size,
                 weights_low=-0.5,
                 weights_high=-0.5
                 ):
        """__init__ for a Bernoulli Logistic Unit

        Parameters
        ----------
        input_size : tuple, list, int
            If tuple or list, should be 2 elements, (batch size, length), where "length" is the length of input vectors.
            If int, should be length of input vectors, and batch size defaults to be one.
        weights_low : float
            Lower boundary of output interval for uniform distribution from which weights are drawn
        weights_high : float
            Upper boundary of output interval for uniform distribution from which weights are drawn
        """
        super().__init__(input_size, weights_low, weights_high)

    @staticmethod
    def f(s):
        return logistic(s)

    @staticmethod
    def sample(p):
        return np.ceil(p - np.random.uniform(size=p.shape)).astype(int)

    def forward(self, x=None):
        """forward pass through unit. Weights convert inputs to vector of
        parameters for Beroulli distributions, p, which are then sampled
        to produce output y.

        Parameters
        ----------
        x : numpy.ndarray
            input vector. Default is None, in which case an input of
            np.ones(self.input_size) is forwarded through the unit.
            This is useful for environments where the only "input signal" is reward,
            e.g., non-associative learning tasks like a multi-arm bandit.

        Returns
        -------
        y : numpy.ndarray
            output vector. Samples from Bernoulli distribution.
        """
        return super().forward(x)

    def e(self):
        """characteristic eligibility"""
        return (self.y - self.p) * self.x
