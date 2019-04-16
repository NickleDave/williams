import numpy as np
from scipy.special import expit as logistic


class BernoulliLogisticUnit:
    def __init__(self,
                 input_size,
                 weights_low=-0.5,
                 weights_high=-0.5
                 ):
        if input_size < 1:
            raise ValueError("input_size must be greater than or "
                             f"equal to 1; got {input_size}")
        if type(input_size) == int:
            input_size = (input_size,)
        else:
            if not input_size.is_integer():
                raise ValueError("input size must be an integer but"
                                 f" value {input_size} is not.")
            else:
                input_size = tuple(int(input_size))

        self.input_size = input_size
        self.w = np.random.uniform(low=weights_low,
                                   high=weights_high,
                                   size=input_size)
        self.f = logistic
        self.x = 0
        self.s = 0
        self.p = 0
        self.y = 0

    @staticmethod
    def bernoulli_sample(p):
        return np.ceil(p - np.random.uniform(size=p.shape)).astype(int)

    def forward(self, x=None):
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
