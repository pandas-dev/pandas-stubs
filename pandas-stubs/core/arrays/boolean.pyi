from collections.abc import Sequence

import numpy as np
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.missing import NAType
from pandas._typing import (
    np_ndarray_bool,
    type_t,
)

from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype

class BooleanDtype(ExtensionDtype):
    @property
    def na_value(self) -> NAType: ...
    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]: ...

class BooleanArray(BaseMaskedArray):
    def __init__(
        self,
        values: (
            Sequence[bool | np.bool]
            | np_ndarray_bool
            | Index[bool]
            | Series[bool]
            | Self
        ),
        mask: np_ndarray_bool,
        copy: bool = False,
    ) -> None: ...
    @property
    def dtype(self) -> BooleanDtype: ...
    def __and__(
        self,
        other: (
            bool
            | np.bool
            | NAType
            | Sequence[bool | np.bool]
            | np_ndarray_bool
            | IntegerArray
            | Self
        ),
    ) -> Self: ...
    def __rand__(
        self,
        other: (
            bool
            | np.bool
            | NAType
            | Sequence[bool | np.bool]
            | np_ndarray_bool
            | IntegerArray
            | Self
        ),
    ) -> Self: ...
