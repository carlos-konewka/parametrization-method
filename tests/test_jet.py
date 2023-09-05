import unittest

import numpy as np

from jet import Jet


class TestJet(unittest.TestCase):
    def test_constructor_from_list(self):
        # given
        data_sequence = [2, 3, 4, 3]
        # when
        jet = Jet(data_sequence)
        # then
        data = jet.data
        self.assertIsInstance(data, np.ndarray)
        self.assertEquals(data.dtype, 'float64')
        self.assertListEqual(list(data), data_sequence)

    def test_constructor_from_numpy(self):
        # given
        data_sequence = np.asarray([2, 3, 4, 3])
        # when
        index = Jet(data_sequence)
        # then
        data = index.data
        self.assertIsInstance(data, np.ndarray)
        self.assertEquals(data.dtype, 'float64')
        self.assertListEqual(list(data), list(data_sequence))
