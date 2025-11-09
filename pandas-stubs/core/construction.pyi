from collections.abc import Sequence
from typing import overload

import numpy as np
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.integer import IntegerArray

from pandas._libs.missing import NAType
from pandas._typing import (
    BooleanDtypeArg,
    IntDtypeArg,
    UIntDtypeArg,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

@overload
def array(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    data: Sequence[bool | NAType | None],
    dtype: BooleanDtypeArg | None = None,
    copy: bool = True,
) -> BooleanArray: ...
@overload
def array(
    data: Sequence[int | NAType | None],
    dtype: IntDtypeArg | UIntDtypeArg | None = None,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(
    data: Sequence[object],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> ExtensionArray: ...
