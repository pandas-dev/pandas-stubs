from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

class PandasDtype(ExtensionDtype):
    def __init__(self, dtype) -> None: ...
    @property
    def numpy_dtype(self): ...
    @property
    def name(self): ...
    @property
    def type(self): ...
    @classmethod
    def construct_from_string(cls, string): ...
    @classmethod
    def construct_array_type(cls): ...
    @property
    def kind(self): ...
    @property
    def itemsize(self): ...

class PandasArray(ExtensionArray, ExtensionOpsMixin, NDArrayOperatorsMixin):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...
