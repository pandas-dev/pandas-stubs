from collections.abc import Sequence
from typing import Any

from typing_extensions import Self

from pandas._typing import (
    AnyArrayLikeInt,
    AxisInt,
    DtypeObj,
    Shape,
    np_ndarray,
)

class NDArrayBacked:
    _dtype: DtypeObj
    _ndarray: np_ndarray
    def __setstate__(self, state: Any) -> None: ...
    def __len__(self) -> int: ...
    @property
    def shape(self) -> Shape: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def repeat(
        self,
        repeats: int | Sequence[int] | AnyArrayLikeInt,
        axis: AxisInt | None = None,
    ) -> Self: ...
    def reshape(self, *args: Any, **kwargs: Any) -> Self: ...
    @property
    def T(self) -> Self: ...
