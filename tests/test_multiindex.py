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

    def test_equality(self):
        # given
        u_data_sequence = np.asarray([1, 2, 3, 4])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertTrue(equal)

    def test_inequality(self):
        # given
        u_data_sequence = np.asarray([1, 2, 2, 4])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertFalse(equal)

    def test_equality_when_different_lengths(self):
        # given
        u_data_sequence = np.asarray([1, 2, 3, 4, 0])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertTrue(equal)

    def test_inequality_when_different_lengths(self):
        # given
        u_data_sequence = np.asarray([1, 2, 3, 4, 8])
        v_data_sequence = np.asarray([1, 2, 3, 4])
        u = Multiindex(u_data_sequence)
        v = Multiindex(v_data_sequence)
        # when
        equal = u == v
        # then
        self.assertFalse(equal)

    def test_leq(self):
        # given
        u = Multiindex([1, 3, 4, 2, 1])
        v = Multiindex([2, 3, 4, 5, 6])
        # when
        leq = u <= v
        # then
        self.assertTrue(leq)

    def test_non_leq_when_all_greater(self):
        # given
        u = Multiindex([2, 4, 5, 5, 6])
        v = Multiindex([1, 3, 4, 2, 1])
        # when
        leq = u <= v
        # then
        self.assertFalse(leq)

    def test_non_leq_when_at_least_one_greater(self):
        # given
        u = Multiindex([1, 3, 4, 2, 1])
        v = Multiindex([1, 4, 5, 1, 1])
        # when
        leq = u <= v
        # then
        self.assertFalse(leq)

    def test_leq_different_lengths(self):
        # given
        u = Multiindex([1, 2, 3, 0, 0])
        v = Multiindex([3, 3, 3])
        # when
        leq = u <= v
        # then
        self.assertTrue(leq)

    def test_non_leq_different_lengths(self):
        # given
        u = Multiindex([1, 2, 3, 0, 1])
        v = Multiindex([3, 3, 3])
        # when
        leq = u <= v
        # then
        self.assertFalse(leq)


if __name__ == '__main__':
    unittest.main()
