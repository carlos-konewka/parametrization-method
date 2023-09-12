import unittest

import numpy as np

from jet import Jet
from taylor_series import TaylorSeries


class TestTaylorSeries(unittest.TestCase):
    def test_one_variable(self):
        # given
        variables = 1
        deg = 5
        x = 0.2
        point = Jet.create_variable(x, variables, 0, deg)
        y = self._one_variable_function(point)
        taylor = TaylorSeries(y)
        taylor_args = [-0.9]
        # when
        approximation = taylor(taylor_args)
        # then
        point_plus_t = Jet.create_variable(x + taylor_args[0], variables, 0, deg)
        expected = self._one_variable_function(point_plus_t).get_value()
        self.assertAlmostEqual(approximation, expected)

    def test_multi_variables(self):
        # given
        deg = 8
        x = np.asarray([0.2, 3])
        points = Jet.create_variables(x, deg)
        y = self._multi_variables_function(*points)
        taylor = TaylorSeries(y)
        taylor_args = np.asarray([0.1, 0.14])
        # when
        approximation = taylor(taylor_args)
        # then
        points_plus_t = Jet.create_variables(x + taylor_args, deg)
        expected = self._multi_variables_function(*points_plus_t).get_value()
        self.assertAlmostEqual(approximation, expected)

    def _one_variable_function(self, x: Jet) -> Jet:
        return (x * x - 0.1) * x * x * x + 3 * x

    def _multi_variables_function(self, x: Jet, y: Jet) -> Jet:
        return x / y - y*y + x*y*(x-y) / (x*x+y*y / 0.2 + 0.2)


if __name__ == '__main__':
    unittest.main()
