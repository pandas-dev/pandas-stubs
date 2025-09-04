from collections.abc import (
    Hashable,
    Iterator,
    Sequence,
)
from typing import (
    Any,
    Generic,
    Literal,
    TypeAlias,
    final,
    overload,
)

import numpy as np
from pandas import (
    Index,
    Series,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from typing_extensions import Self

from pandas._typing import (
    S1,
    ArrayLike,
    AxisIndex,
    DropKeep,
    DTypeLike,
    GenericT,
    GenericT_co,
    NDFrameT,
    Scalar,
    SequenceNotStr,
    SupportsDType,
    np_1darray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_float,
)
from pandas.util._decorators import cache_readonly

_ListLike: TypeAlias = ArrayLike | dict[str, np.ndarray] | SequenceNotStr[S1]

class NoNewAttributesMixin:
    def __setattr__(self, key: str, value: Any) -> None: ...

class SelectionMixin(Generic[NDFrameT]):
    obj: NDFrameT
    exclusions: frozenset[Hashable]
    @final
    @cache_readonly
    def ndim(self) -> int: ...
    def __getitem__(self, key): ...
    def aggregate(self, func, *args, **kwargs): ...

class IndexOpsMixin(OpsMixin, Generic[S1, GenericT_co]):
    __array_priority__: int = ...
    @property
    def T(self) -> Self: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    def item(self) -> S1: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def array(self) -> ExtensionArray: ...
    @overload
    def to_numpy(
        self,
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT_co]: ...
    @overload
    def to_numpy(
        self,
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @property
    def empty(self) -> bool: ...
    def max(
        self, axis: AxisIndex | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> S1: ...
    def min(
        self, axis: AxisIndex | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> S1: ...
    def argmax(
        self,
        axis: AxisIndex | None = ...,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def argmin(
        self,
        axis: AxisIndex | None = ...,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def tolist(self) -> list[S1]: ...
    def to_list(self) -> list[S1]: ...
    def __iter__(self) -> Iterator[S1]: ...
    @property
    def hasnans(self) -> bool: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = ...,
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | None = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | None = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    def nunique(self, dropna: bool = True) -> int: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    def factorize(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> tuple[np_1darray, np_1darray | Index | Categorical]: ...
    @overload
    def searchsorted(
        self,
        value: _ListLike,
        side: Literal["left", "right"] = ...,
        sorter: _ListLike | None = ...,
    ) -> np_1darray[np.intp]: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Literal["left", "right"] = ...,
        sorter: _ListLike | None = ...,
    ) -> np.intp: ...
    def drop_duplicates(self, *, keep: DropKeep = ...) -> Self: ...

NumListLike: TypeAlias = (
    ExtensionArray
    | np_ndarray_bool
    | np_ndarray_anyint
    | np_ndarray_float
    | np_ndarray_complex
    | dict[str, np.ndarray]
    | Sequence[complex]
    | IndexOpsMixin[complex]
)
