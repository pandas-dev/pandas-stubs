from collections.abc import Sequence
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Any,
    Never,
    TypeAlias,
    overload,
)

import numpy as np
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.period import PeriodArray
from pandas.core.arrays.sparse.array import SparseArray
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringArray,
    StringDtype,
)
from pandas.core.arrays.string_arrow import ArrowStringArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.base import Index
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series

from pandas._libs.interval import Interval
from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.period import Period
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    BuiltinNotStrObjDtypeArg,
    BuiltinObjectDtypeArg,
    BuiltinStrDtypeArg,
    CategoryDtypeArg,
    IntervalT,
    Just,
    NumpyNotTimeDtypeArg,
    NumpyTimedeltaDtypeArg,
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
    np_ndarray_td,
)

from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
    SparseDtype,
)

_NAStrElement: TypeAlias = str | np.str_ | NAType | None
_NaNStrElement: TypeAlias = Just[float] | _NAStrElement
_NaNStrData: TypeAlias = Sequence[_NaNStrElement] | np_ndarray | BaseStringArray
_NaTDatetimeElement: TypeAlias = (
    Just[float] | str | datetime | np.datetime64 | NaTType | None
)

@overload
def array(data: str, dtype: Any = None, copy: bool = True) -> Never: ...
@overload
def array(  # empty data, [float("nan")] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[Just[float]],
    dtype: PandasFloatDtypeArg | None = None,
    copy: bool = True,
) -> FloatingArray: ...
@overload
def array(
    data: Sequence[Any],
    dtype: BuiltinNotStrObjDtypeArg | NumpyNotTimeDtypeArg,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[NAType | None],
    dtype: BuiltinNotStrObjDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(
    data: Sequence[Any] | np_ndarray | ExtensionArray | Index | Series,
    dtype: CategoryDtypeArg,
    copy: bool = True,
) -> Categorical: ...
@overload
def array(
    # TODO: Categorical Series pandas-dev/pandas-stubs#1415
    data: Categorical | CategoricalIndex,
    dtype: CategoryDtypeArg | None = None,
    copy: bool = True,
) -> Categorical: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[Period | NaTType | None] | PeriodArray | PeriodIndex | Series[Period]
    ),
    dtype: PeriodDtype | None = None,
    copy: bool = True,
) -> PeriodArray: ...
@overload
def array(
    data: (
        Sequence[IntervalT | None | float]
        | IntervalArray
        | IntervalIndex
        | Series[Interval]
    ),
    dtype: IntervalDtype,
    copy: bool = True,
) -> IntervalArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: Sequence[IntervalT | None] | IntervalArray | IntervalIndex | Series[Interval],
    dtype: None = None,
    copy: bool = True,
) -> IntervalArray: ...
@overload
def array(
    data: Sequence[Any] | np_ndarray | ExtensionArray | Index | Series,
    dtype: SparseDtype,
    copy: bool = True,
) -> SparseArray: ...
@overload
def array(data: SparseArray, dtype: None = None, copy: bool = True) -> SparseArray: ...
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
    data: Sequence[_NaTDatetimeElement] | np_ndarray | DatetimeArray,
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
    data: (
        Sequence[datetime | np.datetime64 | NaTType | None]
        | np_ndarray_dt
        | DatetimeArray
    ),
    dtype: None = None,
    copy: bool = True,
) -> DatetimeArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[datetime | np.datetime64 | NaTType | None]
        | np_ndarray_dt
        | DatetimeArray
        | DatetimeIndex
        | Series[Timestamp]
    ),
    dtype: PandasTimestampDtypeArg | NumpyTimestampDtypeArg | None = None,
    copy: bool = True,
) -> DatetimeArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: (
        Sequence[timedelta | np.timedelta64 | NaTType | None]
        | np_ndarray_td
        | TimedeltaArray
        | TimedeltaIndex
        | Series[Timedelta]
    ),
    dtype: NumpyTimedeltaDtypeArg | None = None,
    copy: bool = True,
) -> TimedeltaArray: ...
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
@overload
def array(  # type: ignore[overload-overlap]
    data: _NaNStrData,
    dtype: BuiltinStrDtypeArg | PandasBaseStrDtypeArg,
    copy: bool = True,
) -> BaseStringArray: ...
@overload
def array(
    data: Sequence[_NAStrElement] | np_ndarray_str | BaseStringArray,
    dtype: None = None,
    copy: bool = True,
) -> BaseStringArray: ...
@overload
def array(
    data: Sequence[Any], dtype: BuiltinObjectDtypeArg | None = None, copy: bool = True
) -> NumpyExtensionArray: ...
@overload
def array(
    data: np_ndarray | NumpyExtensionArray | RangeIndex,
    dtype: BuiltinNotStrObjDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
