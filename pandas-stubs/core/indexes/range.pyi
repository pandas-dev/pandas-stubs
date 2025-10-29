from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    Any,
    final,
    overload,
)

import numpy as np
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.base import IndexOpsMixin
from pandas.core.indexes.base import (
    Index,
    _IndexSubclassBase,
)
from typing_extensions import Self

from pandas._typing import (
    AnyArrayLike,
    Dtype,
    HashableT,
    MaskType,
    Scalar,
    np_1darray,
    np_ndarray_anyint,
    np_ndarray_bool,
)

class RangeIndex(_IndexSubclassBase[int, np.int64]):
    def __new__(
        cls,
        start: int | RangeIndex | range | None = None,
        stop: int | None = None,
        step: int | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ): ...
    @classmethod
    def from_range(
        cls, data: range, name: Hashable | None = None, dtype: Dtype | None = None
    ) -> Self: ...
    @property
    def start(self) -> int: ...
    @property
    def stop(self) -> int: ...
    @property
    def step(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def has_duplicates(self) -> bool: ...
    def __contains__(self, key: int | np.integer) -> bool: ...
    def factorize(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> tuple[np_1darray[np.intp], RangeIndex]: ...
    @property
    def size(self) -> int: ...
    # Base class returns `Self`, but for `RangeIndex` that's not true.
    def __floordiv__(  # type: ignore[override]
        self, other: float | Sequence[float] | Index[int] | Index[float]
    ) -> Index[int]: ...
    def all(self, *args: Any, **kwargs: Any) -> bool: ...
    def any(self, *args: Any, **kwargs: Any) -> bool: ...
    @final
    def union(  # type: ignore[override]
        self, other: list[HashableT] | Index, sort: bool | None = None
    ) -> Index | Index[int] | RangeIndex: ...
    @overload  # type: ignore[override]
    # pyrefly: ignore  # bad-override
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Index: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, idx: int
    ) -> int: ...
    def where(  # type: ignore[override]
        self,
        cond: Sequence[bool] | np_ndarray_bool | BooleanArray | IndexOpsMixin[bool],
        other: Scalar | AnyArrayLike | None = None,
    ) -> Index: ...
