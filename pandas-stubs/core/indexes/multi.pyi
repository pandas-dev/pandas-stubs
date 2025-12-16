from collections.abc import (
    Collection,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import sys
from typing import (
    Any,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
from typing_extensions import Self

from pandas._typing import (
    AnyAll,
    Axes,
    Dtype,
    HashableT,
    IndexLabel,
    Label,
    Level,
    MaskType,
    NaPosition,
    NumpyNotTimeDtypeArg,
    NumpyTimedeltaDtypeArg,
    NumpyTimestampDtypeArg,
    SequenceNotStr,
    Shape,
    np_1darray_bool,
    np_1darray_int8,
    np_1darray_intp,
    np_ndarray,
    np_ndarray_anyint,
)

class MultiIndex(Index):
    def __new__(
        cls,
        levels: Sequence[SequenceNotStr[Hashable]] = ...,
        codes: Sequence[Sequence[int]] = ...,
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
        copy: bool = ...,
        name: SequenceNotStr[Hashable] = ...,
        verify_integrity: bool = ...,
    ) -> Self: ...
    @classmethod
    def from_arrays(
        cls,
        arrays: Sequence[Axes],
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @classmethod
    def from_tuples(
        cls,
        tuples: Iterable[tuple[Hashable, ...]],
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @classmethod
    def from_product(
        cls,
        iterables: Sequence[SequenceNotStr[Hashable] | pd.Series | pd.Index | range],
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        sortorder: int | None = ...,
        names: SequenceNotStr[Hashable] = ...,
    ) -> Self: ...
    @property  # Should be read-only
    def levels(self) -> list[Index]: ...
    @overload
    def set_levels(
        self,
        levels: Sequence[SequenceNotStr[Hashable]],
        *,
        level: Sequence[Level] | None = None,
        verify_integrity: bool = True,
    ) -> MultiIndex: ...
    @overload
    def set_levels(
        self,
        levels: SequenceNotStr[Hashable],
        *,
        level: Level,
        verify_integrity: bool = True,
    ) -> MultiIndex: ...
    @property
    def codes(self) -> list[np_1darray_int8]: ...
    @overload
    def set_codes(
        self,
        codes: Sequence[Sequence[int]],
        *,
        level: Sequence[Level] | None = None,
        verify_integrity: bool = True,
    ) -> MultiIndex: ...
    @overload
    def set_codes(
        self,
        codes: Sequence[int],
        *,
        level: Level,
        verify_integrity: bool = True,
    ) -> MultiIndex: ...
    def copy(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore
        self, names: SequenceNotStr[Hashable] = ..., deep: bool = False
    ) -> Self: ...
    def view(self, cls: NumpyNotTimeDtypeArg | NumpyTimedeltaDtypeArg | NumpyTimestampDtypeArg | type[np_ndarray] | None = None) -> MultiIndex: ...  # type: ignore[override] # pyrefly: ignore[bad-override] # pyright: ignore[reportIncompatibleMethodOverride]
    if sys.version_info >= (3, 11):
        @property
        def dtype(self) -> np.dtype: ...
    else:
        @property
        def dtype(self) -> np.dtype[Any]: ...

    @property
    def dtypes(self) -> pd.Series[Dtype]: ...
    def memory_usage(self, deep: bool = False) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __len__(self) -> int: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def dropna(self, how: AnyAll = "any") -> Self: ...
    def droplevel(self, level: Level | Sequence[Level] = 0) -> MultiIndex | Index: ...  # type: ignore[override]
    def get_level_values(self, level: str | int) -> Index: ...
    @overload  # type: ignore[override]
    def unique(  # pyrefly: ignore[bad-override]
        self, level: None = None
    ) -> MultiIndex: ...
    @overload
    def unique(  # ty: ignore[invalid-method-override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, level: Level
    ) -> Index: ...
    def to_frame(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        index: bool = True,
        name: list[HashableT] = ...,
        allow_duplicates: bool = False,
    ) -> pd.DataFrame: ...
    def to_flat_index(self) -> Index: ...
    def remove_unused_levels(self) -> MultiIndex: ...
    @property
    def nlevels(self) -> int: ...
    @property
    def levshape(self) -> Shape: ...
    @overload  # type: ignore[override]
    # pyrefly: ignore  # bad-override
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, key: int
    ) -> tuple[Hashable, ...]: ...
    @overload  # type: ignore[override]
    def append(self, other: MultiIndex | Sequence[MultiIndex]) -> MultiIndex: ...
    @overload
    def append(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Index | Sequence[Index]
    ) -> Index: ...
    def drop(self, codes: Level | Sequence[Level], level: Level | None = None, errors: str = "raise") -> MultiIndex: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def swaplevel(self, i: int = -2, j: int = -1) -> Self: ...
    def reorder_levels(self, order: Sequence[Level]) -> MultiIndex: ...
    def sortlevel(
        self,
        level: Level | Sequence[Level] = 0,
        ascending: bool = True,
        sort_remaining: bool = True,
        na_position: NaPosition = "first",
    ) -> tuple[MultiIndex, np_1darray_intp]: ...
    def get_loc_level(
        self,
        key: Label | Sequence[Label],
        level: Level | Sequence[Level] | None = None,
        drop_level: bool = True,
    ) -> tuple[int | slice | np_1darray_bool, Index]: ...
    def get_locs(self, seq: Level | Sequence[Level]) -> np_1darray_intp: ...
    def truncate(
        self, before: IndexLabel | None = None, after: IndexLabel | None = None
    ) -> MultiIndex: ...
    @overload  # type: ignore[override]
    def isin(  # pyrefly: ignore[bad-override]
        self, values: Iterable[Any], level: Level
    ) -> np_1darray_bool: ...
    @overload
    def isin(  # ty: ignore[invalid-method-override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, values: Collection[Iterable[Any]], level: None = None
    ) -> np_1darray_bool: ...
    def set_names(
        self,
        names: Hashable | Sequence[Hashable] | Mapping[Any, Hashable],
        *,
        level: Level | Sequence[Level] | None = None,
        inplace: bool = False,
    ) -> Self: ...
