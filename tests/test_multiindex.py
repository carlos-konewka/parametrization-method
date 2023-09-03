import unittest

import numpy as np

from multiindex import Multiindex


class TestMultiindex(unittest.TestCase):
    def test_constructor_from_list(self):
        # given
        data_sequence = [2, 3, 4, 3]
        # when
        index = Multiindex(data_sequence)
        # then
        data = index.data
        self.assertIsInstance(data, np.ndarray)
        self.assertEquals(data.dtype, 'int64')
        self.assertListEqual(list(data), data_sequence)

    def test_constructor_from_numpy(self):
        # given
        data_sequence = np.asarray([2, 3, 4, 3])
        # when
        index = Multiindex(data_sequence)
        # then
        data = index.data
        self.assertIsInstance(data, np.ndarray)
        self.assertEquals(data.dtype, 'int64')
        self.assertListEqual(list(data), list(data_sequence))

    def test_equals_when_indices_are_equal(self):
        # given
        u_data_sequence = np.asarray([1, 2, 3, 4])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertTrue(equal)

    def test_equals_when_indices_are_not_equal(self):
        # given
        u_data_sequence = np.asarray([1, 2, 2, 4])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertFalse(equal)

    def test_equals_when_indices_are_equal_and_lengths_are_different(self):
        # given
        u_data_sequence = np.asarray([1, 2, 3, 4, 0])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertTrue(equal)

    def test_equals_when_indices_are_not_equal_and_lengths_are_different(self):
        # given
        u_data_sequence = np.asarray([1, 2, 3, 4, 8])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertFalse(equal)


if __name__ == '__main__':
    unittest.main()
