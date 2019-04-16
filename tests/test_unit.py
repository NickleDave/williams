import unittest

import numpy as np

from williams.units.blu import BernoulliLogisticUnit


class TestBernoulliLogisticUnit(unittest.TestCase):
    def test_init(self):
        input_size = (10, 20)
        blu = BernoulliLogisticUnit(input_size=input_size)
        self.assertTrue(blu.input_size == input_size)
        self.assertTrue(blu.w.shape == (input_size[1],))

        input_size = [10, 20]
        blu = BernoulliLogisticUnit(input_size=input_size)
        self.assertTrue(blu.input_size == input_size)
        self.assertTrue(blu.w.shape == (input_size[1],))

        input_size = 20
        blu = BernoulliLogisticUnit(input_size=input_size)
        self.assertTrue(blu.input_size == (1, 20))
        self.assertTrue(blu.w.shape == (input_size,))

    def test_forward(self):
        input_size = (10, 20)
        blu = BernoulliLogisticUnit(input_size=input_size)
        y = blu.forward()
        self.assertTrue(type(y) == np.ndarray)
        self.assertTrue(y.shape == blu.input_size[:1])

        input_size = 20
        blu = BernoulliLogisticUnit(input_size=input_size)
        y = blu.forward()
        self.assertTrue(type(y) == np.ndarray)
        self.assertTrue(y.shape == blu.input_size[:1])


if __name__ == '__main__':
    unittest.main()
