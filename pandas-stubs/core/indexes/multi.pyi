from collections.abc import (
    Callable,
    Hashable,
    Sequence,
)
from typing import (
    Any,
    Literal,
    overload,
)

from _typing import SequenceNotStr
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
from typing_extensions import Self

from pandas._typing import (
    Dtype,
    DtypeArg,
    HashableT,
    MaskType,
    np_ndarray_anyint,
    np_ndarray_bool,
)

class MultiIndex(Index[Any]):
    def __init__(
        self,
        levels=...,
        codes=...,
        sortorder=...,
        names=...,
        dtype=...,
        copy=...,
        name: SequenceNotStr[Hashable] = ...,
        verify_integrity: bool = ...,
        _set_identity: bool = ...,
    ) -> None: ...
    @classmethod
    def from_arrays(cls, arrays, sortorder=..., names=...) -> Self: ...
    @classmethod
    def from_tuples(cls, tuples, sortorder=..., names=...) -> Self: ...
    @classmethod
    def from_product(cls, iterables, sortorder=..., names=...) -> Self: ...
    @classmethod
    def from_frame(cls, df, sortorder=..., names=...) -> Self: ...
    @property
    def shape(self): ...
    @property  # Should be read-only
    def levels(self) -> list[Index]: ...
    def set_levels(self, levels, *, level=..., verify_integrity: bool = ...): ...
    @property
    def codes(self): ...
    def set_codes(self, codes, *, level=..., verify_integrity: bool = ...): ...
    def copy(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, names=..., deep: bool = ...
    ) -> Self: ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def view(self, cls=...): ...
    def __contains__(self, key) -> bool: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def dtypes(self) -> pd.Series[Dtype]: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def format(
        self,
        name: bool | None = ...,
        formatter: Callable | None = ...,
        na_rep: str | None = ...,
        names: bool = ...,
        space: int = ...,
        sparsify: bool | None = ...,
        adjoin: bool = ...,
    ) -> list: ...
    def __len__(self) -> int: ...
    @property
    def values(self): ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def duplicated(self, keep: Literal["first", "last", False] = ...): ...
    def fillna(self, value=..., downcast=...) -> None: ...
    def dropna(self, how: Literal["any", "all"] = ...) -> Self: ...
    def get_level_values(self, level: str | int) -> Index: ...
    def unique(self, level=...): ...
    def to_frame(
        self,
        index: bool = ...,
        name: list[HashableT] = ...,
        allow_duplicates: bool = ...,
    ) -> pd.DataFrame: ...
    def to_flat_index(self): ...
    def remove_unused_levels(self): ...
    @property
    def nlevels(self) -> int: ...
    @property
    def levshape(self): ...
    def __reduce__(self): ...
    @overload  # type: ignore[override]
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: int
    ) -> tuple: ...
    def take(
        self, indices, axis: int = ..., allow_fill: bool = ..., fill_value=..., **kwargs
    ): ...
    def append(self, other): ...
    def argsort(self, *args, **kwargs): ...
    def repeat(self, repeats, axis=...): ...
    def where(self, cond, other=...) -> None: ...
    def drop(self, codes, level=..., errors: str = ...) -> Self: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def swaplevel(self, i: int = ..., j: int = ...): ...
    def reorder_levels(self, order): ...
    def sortlevel(
        self, level: int = ..., ascending: bool = ..., sort_remaining: bool = ...
    ): ...
    def get_indexer(self, target, method=..., limit=..., tolerance=...): ...
    def get_indexer_non_unique(self, target): ...
    def reindex(self, target, method=..., level=..., limit=..., tolerance=...): ...
    def get_slice_bound(
        self, label: Hashable | Sequence[Hashable], side: str
    ) -> int: ...
    def slice_locs(self, start=..., end=..., step=...): ...
    def get_loc_level(self, key, level=..., drop_level: bool = ...): ...
    def get_locs(self, seq): ...
    def truncate(self, before=..., after=...): ...
    def equals(self, other) -> bool: ...
    def equal_levels(self, other): ...
    def union(self, other, sort=...): ...
    def intersection(self, other: list | Self, sort: bool = ...): ...
    def difference(self, other, sort=...): ...
    def astype(self, dtype: DtypeArg, copy: bool = ...) -> Self: ...
    def insert(self, loc, item): ...
    def delete(self, loc): ...
    def isin(self, values, level=...) -> np_ndarray_bool: ...

def maybe_droplevels(index, key): ...
