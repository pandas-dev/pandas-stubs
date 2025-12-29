from collections.abc import Sequence
import sys
from typing import (
    Any,
    overload,
)

import numpy as np
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringArray,
)
from pandas.core.arrays.string_arrow import ArrowStringArray

from pandas._libs.missing import NAType
from pandas._typing import (
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

from pandas.core.dtypes.base import ExtensionDtype

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
    data: (
        SequenceNotStr[str | np.str_ | float | NAType | None]
        | np_ndarray
        | BaseStringArray
    ),
    dtype: PyArrowStrDtypeArg,
    copy: bool = True,
) -> ArrowStringArray: ...
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
        SequenceNotStr[str | np.str_ | float | NAType | None]
        | np_ndarray
        | BaseStringArray
    ),
    dtype: PandasBaseStrDtypeArg,
    copy: bool = True,
) -> BaseStringArray: ...
@overload
def array(
    data: (
        SequenceNotStr[str | np.str_ | NAType | None] | np_ndarray_str | BaseStringArray
    ),
    dtype: PandasBaseStrDtypeArg | None = None,
    copy: bool = True,
) -> BaseStringArray: ...

if sys.version_info >= (3, 11):
    @overload
    def array(
        data: Sequence[object],
        dtype: str | np.dtype | ExtensionDtype | None = None,
        copy: bool = True,
    ) -> ExtensionArray: ...

else:
    @overload
    def array(
        data: Sequence[object],
        dtype: str | np.dtype[Any] | ExtensionDtype | None = None,
        copy: bool = True,
    ) -> ExtensionArray: ...
