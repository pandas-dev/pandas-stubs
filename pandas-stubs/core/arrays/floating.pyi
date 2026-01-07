from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)

from pandas._typing import (
    np_ndarray_bool,
    np_ndarray_float,
)

class FloatingDtype(NumericDtype):
    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]: ...

class FloatingArray(NumericArray):
    @property
    def dtype(self) -> FloatingDtype: ...
    def __init__(
        self, values: np_ndarray_float, mask: np_ndarray_bool, copy: bool = False
    ) -> None: ...

class Float32Dtype(FloatingDtype): ...
class Float64Dtype(FloatingDtype): ...
