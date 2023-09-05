from typing import Iterable

import numpy as np


class Jet:
    def __init__(self, data: Iterable) -> None:
        self._data = np.asarray(data, dtype=np.float64)

    @property
    def data(self) -> np.ndarray:
        return np.copy(self._data)
