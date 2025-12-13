import sys
from typing import Any

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

class PandasDtype(ExtensionDtype):
    if sys.version_info >= (3, 11):
        @property
        def numpy_dtype(self) -> np.dtype: ...
    else:
        @property
        def numpy_dtype(self) -> np.dtype[Any]: ...

    @property
    def itemsize(self) -> int: ...

class PandasArray(ExtensionArray, ExtensionOpsMixin, NDArrayOperatorsMixin): ...
