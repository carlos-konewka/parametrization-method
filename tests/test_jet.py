import unittest
from typing import Dict

from jet import Jet
from multiindex import Multiindex


class TestJet(unittest.TestCase):
    def test_constructor_from_flat_data(self):
        # given
        data_sequence = [2, 3, 4, 3]
        variables = 3
        deg = 1
        # when
        jet = Jet.create_from_flat_data(data_sequence, variables)
        # then
        self.assertEqual(jet.deg, deg)
        self.assertEqual(jet.variables, variables)
        derivatives_dict = jet.derivatives
        self.assertIsInstance(derivatives_dict, Dict)
        expected_dict = {
            Multiindex([0, 0, 0]): 2, Multiindex([1, 0, 0]): 3, Multiindex([0, 1, 0]): 4, Multiindex([0, 0, 1]): 3
        }
        self.assertDictEqual(derivatives_dict, expected_dict)

    def test_variables_num(self):
        # given
        variables_num = 3
        jet = Jet.create_from_flat_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], variables_num)
        # when
        result = jet.variables
        # then
        self.assertEqual(result, variables_num)

    def test_deg(self):
        # given
        variables_num = 3
        deg = 2
        jet = Jet.create_from_flat_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], variables_num)
        # when
        result = jet.deg
        # then
        self.assertEqual(result, deg)

    def test_constructor_from_derivatives(self):
        # given
        taylor_sequence = {
            Multiindex([0, 0]): 1,
            Multiindex([1, 0]): -3, Multiindex([0, 1]): 4,
            Multiindex([2, 0]): 1, Multiindex([1, 1]): 2, Multiindex([0, 2]): 9
        }
        # when
        jet = Jet(taylor_sequence)
        # then
        data = jet._derivatives
        self.assertDictEqual(data, taylor_sequence)

    def test_addition(self):
        # given
        variables = 2
        deg = 3
        u = Jet.create_from_flat_data([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], variables)
        v = Jet.create_from_flat_data([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], variables)
        # when
        result = u + v
        # then
        result_data = result.derivatives
        expected = Jet.create_from_flat_data([10] * 10, variables).derivatives
        self.assertDictEqual(result_data, expected)

    def test_subtraction(self):
        # given
        variables = 3
        deg = 2
        u = Jet.create_from_flat_data([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], variables)
        v = Jet.create_from_flat_data([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], variables)
        # when
        result = u - v
        # then
        result_data = result.derivatives
        expected = Jet.create_from_flat_data([0] * 10, variables).derivatives
        self.assertDictEqual(result_data, expected)

    def test_multiplication(self):
        # given
        variables = 2
        deg = 2
        u = Jet.create_from_flat_data([2., -1, 3, -1, 4, -3], variables)
        v = Jet.create_from_flat_data([0., 1, 0, -1, 1, 9], variables)
        # when
        result = u * v
        # then
        result = result.derivatives
        expected = Jet.create_from_flat_data([0., 2, 0, -3, 5, 18], variables).derivatives
        self.assertDictEqual(result, expected)

    def test_multiplication_one_variable(self):
        # given
        variables = 1
        u = Jet.create_from_flat_data([2., -1, 3, -1, 4, -3], variables)
        v = Jet.create_from_flat_data([0., 1, 0, -1, 1, 9], variables)
        # when
        result = u * v
        # then
        result = result.derivatives
        expected = Jet.create_from_flat_data([0., 2, -1, 1, 2, 18], variables).derivatives
        self.assertDictEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
