from collections.abc import Sequence
from typing import overload

import numpy as np
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray

from pandas._libs.missing import NAType

from pandas.core.dtypes.dtypes import ExtensionDtype

@overload
def array(
    data: Sequence[int] | Sequence[int | NAType],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> IntegerArray: ...
@overload
def array(
    data: Sequence[object],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> ExtensionArray: ...
