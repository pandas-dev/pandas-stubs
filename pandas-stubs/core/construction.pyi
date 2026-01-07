from collections.abc import Sequence
from typing import (
    Any,
    TypeAlias,
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
    StringDtype,
)
from pandas.core.arrays.string_arrow import ArrowStringArray
from pandas.core.indexes.range import RangeIndex
from typing_extensions import Never

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    BuiltinNotStrDtypeArg,
    Just,
    NumpyNotTimeDtypeArg,
    PandasBaseStrDtypeArg,
    PandasBooleanDtypeArg,
    PandasFloatDtypeArg,
    PandasIntDtypeArg,
    PandasStrDtypeArg,
    PandasUIntDtypeArg,
    PyArrowStrDtypeArg,
    SequenceNotStr,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_float,
    np_ndarray_str,
)

_NaNNullableStrData: TypeAlias = (
    SequenceNotStr[str | np.str_ | float | NAType | None] | np_ndarray | BaseStringArray
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
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg,
    copy: bool = True,
) -> NumpyExtensionArray: ...
@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[NAType | NaTType | None],
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg | None = None,
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
def array(  # type: ignore[overload-overlap]
    data: _NaNNullableStrData, dtype: StringDtype[Never], copy: bool = True
) -> BaseStringArray: ...
@overload
def array(
    data: _NaNNullableStrData, dtype: PyArrowStrDtypeArg, copy: bool = True
) -> ArrowStringArray: ...
@overload
def array(
    data: _NaNNullableStrData, dtype: PandasStrDtypeArg, copy: bool = True
) -> StringArray: ...

# TODO: pandas-dev/pandas#54466 add BuiltinStrDtypeArg after Pandas 3.0
# Also PyArrow will become required in Pandas 3.0, so "string" will give
# ArrowStringArray.
# StringDtype[None] means unknown, so it will still give BaseStringArray
@overload
def array(
    data: _NaNNullableStrData, dtype: PandasBaseStrDtypeArg, copy: bool = True
) -> BaseStringArray: ...
@overload
def array(
    data: (
        SequenceNotStr[str | np.str_ | NAType | None] | np_ndarray_str | BaseStringArray
    ),
    dtype: None = None,
    copy: bool = True,
    # TODO: pandas-dev/pandas#54466
    # PyArrow will become required in Pandas 3.0, so no dtype will give
    # ArrowStringArray.
) -> BaseStringArray: ...
@overload
def array(
    data: SequenceNotStr[Any], dtype: None = None, copy: bool = True
) -> NumpyExtensionArray: ...
@overload
def array(
    data: np_ndarray | NumpyExtensionArray | RangeIndex,
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg | None = None,
    copy: bool = True,
) -> NumpyExtensionArray: ...
