from collections.abc import Iterator
from typing import (
    Any,
    Literal,
)

from pandas import Index
from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)
from typing_extensions import override

from pandas._typing import (
    InterpolateOptions,
    np_ndarray_bool,
    np_ndarray_float,
)

class FloatingDtype(NumericDtype):
    @override
    def construct_array_type(self) -> type[FloatingArray]: ...

class FloatingArray(NumericArray):
    @property
    @override
    def dtype(self) -> FloatingDtype: ...
    def __init__(
        self, values: np_ndarray_float, mask: np_ndarray_bool, copy: bool = False
    ) -> None: ...
    @override
    def __iter__(self) -> Iterator[float]: ...
    @override
    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit: int | None,
        limit_direction: Literal["forward", "backward", "both"],
        limit_area: Literal["inside", "outside"] | None,
        copy: bool,
        **kwargs: Any,
    ) -> FloatingArray: ...

class Float32Dtype(FloatingDtype): ...
class Float64Dtype(FloatingDtype): ...
