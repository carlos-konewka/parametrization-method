from __future__ import annotations

from typing import Iterable

import numpy as np


class Multiindex:
    def __init__(self, data: Iterable) -> None:
        self._data = np.asarray(data, dtype=np.int64)

    @property
    def data(self) -> np.ndarray:
        return np.copy(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: Multiindex) -> bool:
        if len(self) == len(other):
            return np.array_equiv(self._data, other._data)
        d1, d2 = (self._data, other._data) if len(self._data) < len(other._data) else (other._data, self._data)
        short_len = len(d1)
        first_part_cmp = np.array_equiv(d1, d2[:short_len])
        second_part_cmp = not d2[short_len:].any()
        return first_part_cmp and second_part_cmp

    def __le__(self, other: Multiindex) -> bool:
        for i, j in zip(self._data, other._data):
            if i > j:
                return False
        if len(self._data) <= len(other._data):
            return True
        if not self._data[len(other._data):].any():
            return True
        return False
