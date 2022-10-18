import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
)

from pandas._typing import npt

from pandas.core.dtypes.dtypes import ExtensionDtype

class PandasDtype(ExtensionDtype):
    def __init__(self, dtype: npt.DTypeLike) -> None: ...
    @property
    def numpy_dtype(self) -> np.dtype: ...
    @property
    def itemsize(self) -> int: ...

class PandasArray(ExtensionArray, ExtensionOpsMixin, NDArrayOperatorsMixin):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...
