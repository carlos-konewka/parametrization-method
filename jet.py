from __future__ import annotations

import copy
from typing import Dict, List, Collection

import numpy as np

from multiindex import Multiindex


class Jet:
    def __init__(self, data: Collection | Dict) -> None:
        self._derivatives: Dict[Multiindex, float] = {}
        if isinstance(data, Dict):
            self._derivatives = copy.deepcopy(data)
        else:
            var_indices = Jet._create_var_multiindices(len(data))
            for idx, cof in zip(var_indices, data):
                self._derivatives[idx] = float(cof)

    @property
    def derivatives(self) -> Dict[Multiindex, float]:
        return copy.deepcopy(self._derivatives)

    def __len__(self) -> int:
        return len(self._derivatives)

    def __add__(self, other: Jet) -> Jet:
        short_dict, long_dict = self._derivatives, other._derivatives
        if len(self) > len(other):
            short_dict, long_dict = long_dict, short_dict
        result = copy.deepcopy(long_dict)
        for k in long_dict.keys():
            result[k] += short_dict[k]
        return Jet(result)

    def __sub__(self, other: Jet) -> Jet:
        if len(self._derivatives) >= len(other._derivatives):
            result = copy.deepcopy(self._derivatives)
            for k, v in other._derivatives.items():
                result[k] -= v
            return Jet(result)
        result = copy.deepcopy(other._derivatives)
        for k in result.keys():
            result[k] = - result[k]
        for k in other._derivatives.keys():
            result[k] -= other._derivatives[k]
        return Jet(result)

    @staticmethod
    def _create_var_multiindices(length: int) -> List[Multiindex]:
        result = []
        for i in range(length):
            data = np.zeros(length, dtype=np.int64)
            data[i] = 1
            result.append(Multiindex(data))
        return result
