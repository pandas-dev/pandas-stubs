from collections.abc import Sequence
from typing import (
    Any,
    overload,
)

import numpy as np
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    PandasBooleanDtypeArg,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_td,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[bool | np.bool | NAType | None] | np_ndarray_bool | BooleanArray,
    dtype: PandasBooleanDtypeArg | None = None,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[NAType | None],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[bool | np.bool | NAType | None]
        | np_ndarray_bool
        | BooleanArray
        | Index[bool]
        | Series[int]
    ),
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[int | np.integer | NAType | None]
        | np_ndarray_anyint
        | IntegerArray
        | Index[int]
        | RangeIndex
        | Series[int]
    ),
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[float | np.floating | NAType | None]
        | np_ndarray_float
        | FloatingArray
        | Index[float]
        | Series[float]
    ),
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> FloatingArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[Timestamp | np.datetime64 | NaTType | None]
        | np_ndarray_dt
        | DatetimeArray
        | DatetimeIndex
        | Series[Timestamp]
    ),
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> DatetimeArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[Timedelta | np.timedelta64 | NaTType | None]
        | np_ndarray_td
        | TimedeltaArray
        | TimedeltaIndex
        | Series[Timedelta]
    ),
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> TimedeltaArray: ...
@overload
def array(
    data: Sequence[Any] | np_ndarray | ExtensionArray | Index | Series,
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
