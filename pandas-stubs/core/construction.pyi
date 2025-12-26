from collections.abc import Sequence
from typing import overload

import numpy as np
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.indexes.range import RangeIndex

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    BuiltinDtypeArg,
    NumpyNotTimeDtypeArg,
    PandasBooleanDtypeArg,
    PandasFloatDtypeArg,
    PandasIntDtypeArg,
    PandasUIntDtypeArg,
    SequenceNotStr,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_float,
)

@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[NAType | NaTType | None],
    dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[bool | np.bool | NAType | None] | np_ndarray_bool | BooleanArray,
    dtype: PandasBooleanDtypeArg | None = None,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: Sequence[int | np.integer | NAType | None] | np_ndarray_anyint | IntegerArray,
    dtype: PandasIntDtypeArg | PandasUIntDtypeArg | None = None,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(
    data: (
        Sequence[float | np.floating | NAType | None] | np_ndarray_float | FloatingArray
    ),
    dtype: PandasFloatDtypeArg | None = None,
    copy: bool = True,
) -> FloatingArray: ...
@overload
def array(
    data: SequenceNotStr[object] | np_ndarray | NumpyExtensionArray | RangeIndex,
    dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
