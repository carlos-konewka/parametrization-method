from __future__ import annotations

from functools import cmp_to_key
from itertools import product
from typing import Iterable, Tuple

import numpy as np


class Multiindex:
    def __init__(self, data: Iterable) -> None:
        self._data = np.asarray(data, dtype=np.int64)

    @staticmethod
    def zero_index(variables: int) -> Multiindex:
        data = np.zeros(variables, dtype=np.int64)
        return Multiindex(data)

    @staticmethod
    def one_hot_index(variables: int, idx: int) -> Multiindex:
        data = np.zeros(variables, dtype=np.int64)
        data[idx] = 1
        return Multiindex(data)

    @property
    def data(self) -> np.ndarray:
        return np.copy(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: Multiindex) -> bool:
        return np.array_equiv(self._data, other._data)

    def __le__(self, other: Multiindex) -> bool:
        return (self._data <= other._data).prod() != 0

    def __lt__(self, other: Multiindex) -> bool:
        if self <= other and not self == other:
            return True
        return False

    def __ne__(self, other: Multiindex) -> bool:
        return not self == other

    def __ge__(self, other: Multiindex) -> bool:
        return other <= self

    def __gt__(self, other: Multiindex) -> bool:
        return other < self

    def __sub__(self, other: Multiindex) -> Multiindex:
        if not other <= self:
            raise Exception("Subtraction - result has negative elements!")
        data = self._data - other._data
        return Multiindex(data)

    def __hash__(self):
        data_list = self._data.tolist()
        return hash(tuple(data_list))

    @staticmethod
    def get_range(variables: int, deg: int) -> Iterable[Multiindex]:
        all_tuples = filter(lambda x: sum(x) <= deg, product(range(deg + 1), repeat=variables))

        def compare_tuples(u: Tuple[int], v: Tuple[int]) -> int:
            s_u, s_v = sum(u), sum(v)
            if s_u != s_v:
                return int(s_u > s_v) - int(s_u < s_v)
            for el_u, el_v in zip(u, v):
                if el_u > el_v:
                    return -1
                elif el_u < el_v:
                    return 1
            return 0

        sorted_tuples = sorted(all_tuples, key=cmp_to_key(compare_tuples))
        result = [Multiindex(idx) for idx in sorted_tuples]
        return result
