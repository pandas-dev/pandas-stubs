import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
)
from pandas.core.strings.object_array import ObjectStringArrayMixin

from pandas.core.dtypes.dtypes import ExtensionDtype

class PandasDtype(ExtensionDtype):
    @property
    def numpy_dtype(self) -> np.dtype: ...
    @property
    def itemsize(self) -> int: ...

class PandasArray(ExtensionArray, ExtensionOpsMixin, NDArrayOperatorsMixin): ...
class NumpyExtensionArray(
    OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin
): ...
