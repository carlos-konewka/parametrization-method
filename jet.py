from __future__ import annotations

import copy
from typing import Dict, List, Collection

import numpy as np

from multiindex import Multiindex


class Jet:
    def __init__(self, data: Dict) -> None:
        self._derivatives = copy.deepcopy(data)

    @property
    def derivatives(self) -> Dict[Multiindex, float]:
        return copy.deepcopy(self._derivatives)

    @property
    def variables(self) -> int:
        any_key = next(iter(self._derivatives.keys()))
        return len(any_key)

    @staticmethod
    def create_from_flat_data(data: Collection, variables: int) -> Jet:
        deg = Jet._deg(len(data), variables)
        indices = Multiindex.get_range(variables, deg)
        derivatives_dict = dict([(i, a) for i, a in zip(indices, data)])
        return Jet(derivatives_dict)

    @staticmethod
    def create_variable(value: float, variables: int, var_idx: int, deg: int) -> Jet:
        indices = Multiindex.get_range(variables, deg)
        result_dict = dict.fromkeys(indices, 0.)
        zero_idx = Multiindex.zero_index(variables)
        result_dict[zero_idx] = value
        one_hot = Multiindex.one_hot_index(variables, var_idx)
        result_dict[one_hot] = 1.
        return Jet(result_dict)

    @staticmethod
    def create_variables(values: Collection, deg: int) -> List[Jet]:
        variables = len(values)
        indices = Multiindex.get_range(variables, deg)
        result_dicts = []
        for i, value in enumerate(values):
            i_dict = dict.fromkeys(indices, 0.)
            zero_idx = Multiindex.zero_index(variables)
            i_dict[zero_idx] = value
            one_hot = Multiindex.one_hot_index(variables, i)
            i_dict[one_hot] = 1.
            result_dicts.append(i_dict)
        result = [Jet(result_dict) for result_dict in result_dicts]
        return result

    @staticmethod
    def create_constant(value: float, variables: int, deg: int) -> Jet:
        indices = Multiindex.get_range(variables, deg)
        result_dict = dict.fromkeys(indices, 0.)
        zero_idx = Multiindex.zero_index(variables)
        result_dict[zero_idx] = value
        return Jet(result_dict)

    @property
    def deg(self) -> int:
        data_length = len(self._derivatives)
        variables = self.variables
        return self._deg(data_length, variables)

    @staticmethod
    def _deg(data_len: int, variables_num: int) -> int:
        d = 0
        length = 1
        while length < data_len:
            d += 1
            length *= (variables_num + d)
            length //= d
        if length != data_len:
            raise Exception("Jet has incorrect length!")
        return d

    def copy(self) -> Jet:
        return Jet(self._derivatives)

    def get_value(self) -> float:
        zero_idx = Multiindex.zero_index(self.variables)
        return self._derivatives[zero_idx]

    def __len__(self) -> int:
        return len(self._derivatives)

    def __add__(self, other: Jet | float) -> Jet:
        if not isinstance(other, Jet):
            result = self.copy()
            zero_idx = Multiindex.zero_index(result.variables)
            result._derivatives[zero_idx] += other
            return result
        short_dict, long_dict = self._derivatives, other._derivatives
        if len(self) > len(other):
            short_dict, long_dict = long_dict, short_dict
        result = copy.deepcopy(long_dict)
        for k in long_dict.keys():
            result[k] += short_dict[k]
        return Jet(result)

    def __radd__(self, other: float) -> Jet:
        return self + other

    def __sub__(self, other: Jet | float) -> Jet:
        if not isinstance(other, Jet):
            result = self.copy()
            zero_idx = Multiindex.zero_index(result.variables)
            result._derivatives[zero_idx] -= other
            return result
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

    def __neg__(self) -> Jet:
        result = self.copy()
        for k, v in result._derivatives.items():
            result._derivatives[k] = -v
        return result

    def __rsub__(self, other: float) -> Jet:
        return -self + other

    def __mul__(self, other: Jet | float) -> Jet:
        if not isinstance(other, Jet):
            result = self.copy()
            for key, value in result._derivatives.items():
                result._derivatives[key] = other * value
            return result
        if len(self._derivatives) != len(other._derivatives):
            raise Exception("Multiplication - jets have different length!")
        variables = self.variables
        deg = self.deg
        indices = Multiindex.get_range(variables, deg)
        result = {}
        for k in indices:
            k_sum = 0.
            for v in filter(lambda x: x <= k, indices):
                k_sum += self._derivatives[v] * other._derivatives[k - v]
            result[k] = k_sum
        return Jet(result)

    def __rmul__(self, other: Jet | float) -> Jet:
        return self * other

    def __truediv__(self, other: Jet | float) -> Jet:
        if not isinstance(other, Jet):
            result = self.copy()
            for k, v in self._derivatives.items():
                result._derivatives[k] = v / other
            return result
        if len(self._derivatives) != len(other._derivatives):
            raise Exception("Division - jets have different length!")
        variables = self.variables
        deg = self.deg
        zero_idx = Multiindex.zero_index(variables)
        result = {
            zero_idx: self._derivatives[zero_idx] / other._derivatives[zero_idx]
        }
        indices = Multiindex.get_range(variables, deg)
        for k in indices[1:]:
            k_sum = 0.
            for v in filter(lambda x: x < k, indices):
                h_v = result[v] * other._derivatives[k - v]
                k_sum += h_v
            k_value = (self._derivatives[k] - k_sum) / other._derivatives[zero_idx]
            result[k] = k_value
        return Jet(result)

    def __rtruediv__(self, other: float) -> Jet:
        deg = self.deg
        variables = self.variables
        other = Jet.create_constant(other, variables, deg)
        return other / self

    @staticmethod
    def _create_var_multiindices(length: int) -> List[Multiindex]:
        result = []
        for i in range(length):
            data = np.zeros(length, dtype=np.int64)
            data[i] = 1
            result.append(Multiindex(data))
        return result
