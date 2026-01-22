from collections.abc import (
    MutableSequence,
    Sequence,
)
from datetime import datetime
from typing import (
    Any,
    Never,
    TypeAlias,
    overload,
)

import numpy as np
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringArray,
    StringDtype,
)
from pandas.core.arrays.string_arrow import ArrowStringArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import (
    BuiltinNotStrDtypeArg,
    Just,
    NumpyNotTimeDtypeArg,
    NumpyTimestampDtypeArg,
    PandasBaseStrDtypeArg,
    PandasBooleanDtypeArg,
    PandasFloatDtypeArg,
    PandasIntDtypeArg,
    PandasStrDtypeArg,
    PandasTimestampDtypeArg,
    PandasUIntDtypeArg,
    PyArrowStrDtypeArg,
    TimedeltaDtypeArg,
    np_1darray_td,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_str,
)

from pandas.core.dtypes.dtypes import DatetimeTZDtype

_NAStrElement: TypeAlias = str | np.str_ | NAType | None
_NaNStrElement: TypeAlias = Just[float] | _NAStrElement
_NaNStrData: TypeAlias = (
    tuple[_NaNStrElement, ...]
    | MutableSequence[_NaNStrElement]
    | np_ndarray
    | BaseStringArray
)
_NaTDatetimeElement: TypeAlias = (
    Just[float] | str | datetime | np.datetime64 | NaTType | None
)

@overload
def array(  # empty data, [float("nan")]
    data: Sequence[Just[float]],
    dtype: PandasFloatDtypeArg | None = None,
    copy: bool = True,
) -> FloatingArray: ...
@overload
def array(
    data: tuple[Any, ...] | MutableSequence[Any],
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[NAType | None],
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # pyright: ignore[reportOverlappingOverload]
    data: (
        Sequence[Timedelta]
        | Series[Timedelta]
        | TimedeltaArray
        | TimedeltaIndex
        | np_1darray_td
    ),
    dtype: TimedeltaDtypeArg | None = None,
    copy: bool = True,
) -> TimedeltaArray: ...
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
def array(
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
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[float | np.floating | NAType | None] | np_ndarray_float | FloatingArray
    ),
    dtype: PandasFloatDtypeArg | None = None,
    copy: bool = True,
) -> FloatingArray: ...
@overload
def array(
    data: (
        tuple[_NaTDatetimeElement, ...]
        | MutableSequence[_NaTDatetimeElement]
        | np_ndarray
        | DatetimeArray
    ),
    dtype: (
        DatetimeTZDtype
        | PandasTimestampDtypeArg
        | np.dtype[np.datetime64]
        | NumpyTimestampDtypeArg
    ),
    copy: bool = True,
) -> DatetimeArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (  # TODO: merge the two Sequence's after 3.0 pandas-dev/pandas#57064
        Sequence[datetime | NaTType | None]
        | Sequence[np.datetime64 | NaTType | None]
        | np_ndarray_dt
        | DatetimeArray
    ),
    dtype: None = None,
    copy: bool = True,
) -> DatetimeArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: _NaNStrData, dtype: StringDtype[Never], copy: bool = True
) -> BaseStringArray: ...
@overload
def array(
    data: _NaNStrData, dtype: PyArrowStrDtypeArg, copy: bool = True
) -> ArrowStringArray: ...
@overload
def array(
    data: _NaNStrData, dtype: PandasStrDtypeArg, copy: bool = True
) -> StringArray: ...

# TODO: pandas-dev/pandas#54466 add BuiltinStrDtypeArg after Pandas 3.0
@overload
def array(
    data: _NaNStrData, dtype: PandasBaseStrDtypeArg, copy: bool = True
) -> BaseStringArray: ...
@overload
def array(  # pyright: ignore[reportOverlappingOverload]
    data: (
        tuple[_NAStrElement, ...]
        | MutableSequence[_NAStrElement]
        | np_ndarray_str
        | BaseStringArray
    ),
    dtype: None = None,
    copy: bool = True,
) -> BaseStringArray: ...
@overload
def array(
    data: tuple[Any, ...] | MutableSequence[Any], dtype: None = None, copy: bool = True
) -> NumpyExtensionArray: ...
@overload
def array(
    data: np_ndarray | NumpyExtensionArray | RangeIndex,
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
