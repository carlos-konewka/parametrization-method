from __future__ import annotations
from typing import Iterable, Tuple

import numpy as np


class Jet:
    def __init__(self, data: Iterable) -> None:
        self._data = np.asarray(data, dtype=np.float64)

    @property
    def data(self) -> np.ndarray:
        return np.copy(self._data)

    def __add__(self, other: Jet) -> Jet:
        data1, data2 = Jet._align_with_zeros(self._data, other._data)
        result = data1 + data2
        return Jet(result)

    @staticmethod
    def _align_with_zeros(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_len = max(len(array1), len(array2))
        a1 = np.append(array1, np.zeros(max_len - len(array1), dtype=np.float64))
        a2 = np.append(array2, np.zeros(max_len - len(array2), dtype=np.float64))
        return a1, a2
