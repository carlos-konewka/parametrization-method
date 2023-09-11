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

    def test_less(self):
        # given
        u = Multiindex([1, 2, 3])
        v = Multiindex([3, 3, 3])
        # when
        le = u < v
        # then
        self.assertTrue(le)

    def test_non_less_eq(self):
        # given
        u = Multiindex([3, 0, 4])
        v = Multiindex([3, 0, 4])
        # when
        le = u < v
        # then
        self.assertFalse(le)

    def test_non_less_ge(self):
        # given
        u = Multiindex([3, 1, 4])
        v = Multiindex([3, 0, 4])
        # when
        le = u < v
        # then
        self.assertFalse(le)

    def test_not_equal_if_true(self):
        # given
        u = Multiindex([3, 1, 4])
        v = Multiindex([3, 0, 4])
        # when
        neq = u != v
        # then
        self.assertTrue(neq)

    def test_ge_if_eq(self):
        # given
        u = Multiindex([3, 0, 3])
        v = Multiindex([3, 0, 3])
        # when
        ge = u >= v
        # then
        self.assertTrue(ge)

    def test_ge_if_greater(self):
        # given
        u = Multiindex([3, 2, 3])
        v = Multiindex([3, 0, 3])
        # when
        ge = u >= v
        # then
        self.assertTrue(ge)

    def test_ge_if_false(self):
        # given
        u = Multiindex([2, 0, 3])
        v = Multiindex([3, 0, 3])
        # when
        ge = u >= v
        # then
        self.assertFalse(ge)

    def test_range(self):
        # given
        variables = 3
        deg = 2
        # when
        multiindices = Multiindex.get_range(variables, deg)
        # then
        multiindices_data = [tuple(index.data) for index in multiindices]
        expected = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1),
                    (0, 0, 2)]
        self.assertListEqual(multiindices_data, expected)


if __name__ == '__main__':
    unittest.main()
