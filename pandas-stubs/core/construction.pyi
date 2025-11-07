from collections.abc import Sequence
from datetime import (
    datetime,
    timedelta,
)
from typing import overload

import numpy as np
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.timedeltas import TimedeltaArray

from pandas._libs.missing import NAType
from pandas._typing import (
    IntDtypeArg,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    UIntDtypeArg,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[timedelta | NAType],
    dtype: TimedeltaDtypeArg | None = None,
    copy: bool = True,
) -> TimedeltaArray: ...
@overload
def array(  # type: ignore[overload-overlap]
    data: Sequence[datetime | NAType],
    dtype: TimestampDtypeArg | None = None,
    copy: bool = True,
) -> DatetimeArray: ...
@overload
def array(
    data: Sequence[int | NAType],
    dtype: IntDtypeArg | UIntDtypeArg | None = None,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(
    data: Sequence[object],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> ExtensionArray: ...
