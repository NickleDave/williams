"""units for neural networks, as described in Williams 1992"""
import numpy as np
from scipy.special import expit as logistic


class BernoulliLogisticUnit:
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
    bernoulli_sample
        static method that accepts parameter p and returns samples from Bernoulli distribution
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
        if type(input_size) == tuple or type(input_size) == list:
            if len(input_size) != 2:
                raise ValueError("when input size is tuple or list, must be two elements: (batch size, length)")
        elif type(input_size) == int:
            if input_size < 1:
                raise ValueError("input_size must be greater than or "
                                 f"equal to 1; got {input_size}")
            input_size = (1, input_size,)
        else:
            raise TypeError("input size must be two-element tuple or list, or int")

        self.input_size = input_size
        self.w = np.random.uniform(low=weights_low,
                                   high=weights_high,
                                   size=(input_size[1],))
        self.f = logistic
        self.x = 0
        self.s = 0
        self.p = 0
        self.y = 0

    @staticmethod
    def bernoulli_sample(p):
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
        if x is None:
            x = np.ones(shape=self.input_size)

        if x.shape != self.input_size:
            raise ValueError(f"shape of x, {x.shape}, does not "
                             f"equal input size of unit, {self.input_size}")

        self.x = x
        self.s = np.dot(x, self.w)
        self.p = self.f(self.s)
        self.y = self.bernoulli_sample(self.p)
        return self.y

    def e(self):
        """characteristic eligibility"""
        return (self.y - self.p) * self.x

    def reset(self):
        self.x = 0
        self.s = 0
        self.p = 0
        self.y = 0
