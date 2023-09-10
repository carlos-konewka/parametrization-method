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

    def __hash__(self):
        data_list = self._data.tolist()
        return hash(tuple(data_list))
