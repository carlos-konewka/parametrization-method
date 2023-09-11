import unittest
from typing import Dict, Collection

import numpy as np

from jet import Jet
from multiindex import Multiindex


class TestJet(unittest.TestCase):
    def test_constructor_from_list(self):
        # given
        data_sequence = [2, 3, 4, 3]
        # when
        jet = Jet(data_sequence)
        # then
        data = jet._derivatives
        self.assertIsInstance(data, Dict)
        result = set(list(data.values()))
        expected = set(data_sequence)
        self.assertSetEqual(result, expected)

    def test_constructor_from_numpy(self):
        # given
        data_sequence = np.asarray([2, 3, 4, 3])
        # when
        jet = Jet(data_sequence)
        # then
        data = jet._derivatives
        self.assertIsInstance(data, Dict)
        result = set(list(data.values()))
        expected = set(data_sequence)
        self.assertSetEqual(result, expected)

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
        u = self._create_jet_from_flat_data([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 2)
        v = self._create_jet_from_flat_data([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 3, 2)
        # when
        result = u + v
        # then
        result_data = result.derivatives
        expected = self._create_jet_from_flat_data([10] * 10, 3, 2).derivatives
        print(result_data)
        print(expected)
        self.assertDictEqual(result_data, expected)

    def test_subtraction(self):
        # given
        variables = 2
        deg = 3
        u = self._create_jet_from_flat_data([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 2)
        v = self._create_jet_from_flat_data([1., 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 2)
        # when
        result = u - v
        # then
        result_data = result.derivatives
        expected = self._create_jet_from_flat_data([0] * 10, 3, 2).derivatives
        print(result_data)
        print(expected)
        self.assertDictEqual(result_data, expected)

    def _create_jet_from_flat_data(self, array: Collection, variables: int, deg: int) -> Jet:
        indices = Multiindex.get_range(variables, deg)
        derivatives_dict = dict([(i, a) for i, a in zip(indices, array)])
        return Jet(derivatives_dict)
