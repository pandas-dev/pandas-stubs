from collections.abc import Sequence
from typing import (
    Any,
    overload,
)

import numpy as np
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringArray,
)
from pandas.core.indexes.range import RangeIndex

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    BuiltinDtypeArg,
    Just,
    NumpyNotTimeDtypeArg,
    PandasBooleanDtypeArg,
    PandasFloatDtypeArg,
    PandasIntDtypeArg,
    PandasStrDtypeArg,
    PandasUIntDtypeArg,
    SequenceNotStr,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_float,
    np_ndarray_str,
)

@overload
def array(  # empty data, [float("nan")]
    data: Sequence[Just[float]],
    dtype: PandasFloatDtypeArg | None = None,
    copy: bool = True,
) -> FloatingArray: ...
@overload
def array(
    data: SequenceNotStr[Any],
    dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[NAType | NaTType | None],
    dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(
    data: Sequence[bool | np.bool | Just[float] | NAType | None],
    dtype: PandasBooleanDtypeArg,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: Sequence[bool | np.bool | NAType | None],
    dtype: None = None,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(  # pyright: ignore[reportOverlappingOverload]
    data: np_ndarray_bool | BooleanArray,
    dtype: PandasBooleanDtypeArg | None = None,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(
    data: Sequence[float | np.integer | NAType | None],
    dtype: PandasIntDtypeArg | PandasUIntDtypeArg,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: Sequence[int | np.integer | NAType | None],
    dtype: None = None,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(
    data: np_ndarray_anyint | IntegerArray,
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
    data: (
        SequenceNotStr[str | np.str_ | float | NAType | None]
        | np_ndarray
        | BaseStringArray
    ),
    dtype: PandasStrDtypeArg,
    copy: bool = True,
) -> StringArray: ...
@overload
def array(
    data: (
        SequenceNotStr[str | np.str_ | NAType | None] | np_ndarray_str | BaseStringArray
    ),
    dtype: None = None,
    copy: bool = True,
) -> BaseStringArray: ...
@overload
def array(
    data: SequenceNotStr[Any],
    dtype: None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(
    data: np_ndarray | NumpyExtensionArray | RangeIndex,
    dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
