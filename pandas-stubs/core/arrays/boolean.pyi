from __future__ import annotations

import numpy as np

from pandas._typing import Scalar

from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype

from .masked import BaseMaskedArray as BaseMaskedArray

_type = type
_type_BooleanArray = type[BooleanArray]

class BooleanDtype(ExtensionDtype):
    name: str = ...
    @property
    def na_value(self) -> Scalar: ...
    @property
    def type(self) -> _type: ...
    @property
    def kind(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> _type_BooleanArray: ...
    def __from_arrow__(self, array): ...

def coerce_to_array(values, mask=..., copy: bool = ...): ...

class BooleanArray(BaseMaskedArray):
    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = ...
    ) -> None: ...
    @property
    def dtype(self): ...
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...
    def __setitem__(self, key, value) -> None: ...
    def astype(self, dtype, copy: bool = ...): ...
    def any(self, skipna: bool = ..., **kwargs): ...
    def all(self, skipna: bool = ..., **kwargs): ...
