from typing import Collection

import numpy as np

from jet import Jet
from multiindex import Multiindex


class TaylorSeries:
    def __init__(self, coefficients: Jet) -> None:
        self._jet = coefficients
        self._deg = coefficients.deg
        self._variables = coefficients.variables

    def __call__(self, t: Collection) -> float:
        t = np.asarray(t, dtype=np.float64)
        indices = Multiindex.get_range(self._variables, self._deg)
        result = self._jet.get_value()
        derivatives = self._jet.derivatives
        for idx in indices[1:]:
            derivative = derivatives[idx]
            t_pow = t ** idx.data
            result += (derivative * t_pow.prod()).item()
        return result
