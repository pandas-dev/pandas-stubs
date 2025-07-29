from collections.abc import Sequence

import numpy as np
from pandas.core.arrays.base import ExtensionArray

from pandas.core.dtypes.dtypes import ExtensionDtype

def array(
    data: Sequence[object],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> ExtensionArray: ...
