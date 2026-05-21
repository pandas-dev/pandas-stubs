from collections.abc import (
    Callable,
    Hashable,
    Sequence,
)
from typing import (
    Any,
    Generic,
    Literal,
    Self,
    overload,
)

import numpy as np
from pandas import Series
from pandas.core.accessor import PandasDelegate as PandasDelegate
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.base import NoNewAttributesMixin as NoNewAttributesMixin
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index

from pandas._libs.missing import NAType
from pandas._typing import (
    AnyArrayLike,
    ListLike,
    NaPosition,
    NpDtype,
    Ordered,
    PositionalIndexerTuple,
    Renamer,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    SequenceNotStr,
    np_1darray,
    np_1darray_bool,
)

from pandas.core.dtypes.dtypes import (
    CategoricalDtype as CategoricalDtype,
    CategoricalValueT,
)

class Categorical(NDArrayBackedExtensionArray):
    __array_priority__: int = ...
    def __init__(
        self,
        values: SequenceNotStr[Hashable] | AnyArrayLike,
        categories: SequenceNotStr[Hashable] | AnyArrayLike | None = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype[Any] | None = None,
        copy: bool = True,
    ) -> None: ...
    @property
    def categories(self) -> Index: ...
    @property
    def ordered(self) -> Ordered: ...
    @property
    def dtype(self) -> CategoricalDtype[object]: ...
    def tolist(self) -> list[Scalar]: ...
    @classmethod
    def from_codes(
        cls,
        codes: AnyArrayLike | Sequence[int],
        categories: Index | None = ...,
        ordered: bool | None = ...,
        dtype: CategoricalDtype[Any] | None = ...,
        validate: bool = True,
    ) -> Categorical: ...
    @property
    def codes(self) -> np_1darray[np.signedinteger]: ...
    def set_ordered(self, value: bool) -> Self: ...
    def as_ordered(self) -> Categorical: ...
    def as_unordered(self) -> Categorical: ...
    def set_categories(
        self,
        new_categories: AnyArrayLike | SequenceNotStr[Hashable],
        ordered: bool | None = False,
        rename: bool = False,
    ) -> Self: ...
    def rename_categories(self, new_categories: Renamer) -> Categorical: ...
    def reorder_categories(
        self,
        new_categories: SequenceNotStr[Hashable] | AnyArrayLike,
        ordered: bool | None = None,
    ) -> Categorical: ...
    def add_categories(
        self, new_categories: AnyArrayLike | SequenceNotStr[Hashable]
    ) -> Categorical: ...
    def remove_categories(
        self, removals: Hashable | SequenceNotStr[Hashable] | AnyArrayLike
    ) -> Categorical: ...
    def remove_unused_categories(self) -> Categorical: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @property
    def nbytes(self) -> int: ...
    def memory_usage(self, deep: bool = False) -> int: ...
    def isna(self) -> np_1darray_bool: ...
    def isnull(self) -> np_1darray_bool: ...
    def notna(self) -> np_1darray_bool: ...
    def notnull(self) -> np_1darray_bool: ...
    @overload
    def sort_values(
        self,
        *,
        inplace: Literal[False] = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
    ) -> Self: ...
    @overload
    def sort_values(
        self,
        *,
        inplace: Literal[True],
        ascending: bool = True,
        na_position: NaPosition = "last",
    ) -> None: ...
    def __contains__(self, item: Hashable) -> bool: ...
    @overload
    def __getitem__(  # pyrefly: ignore[bad-override]
        self, key: ScalarIndexer
    ) -> Any: ...
    @overload
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, key: SequenceIndexer | PositionalIndexerTuple
    ) -> Self: ...
    def min(self, *, skipna: bool = True, **kwargs: Any) -> Scalar | NAType: ...
    def max(self, *, skipna: bool = True, **kwargs: Any) -> Scalar | NAType: ...
    def equals(self, other: Any) -> bool: ...
    def describe(self) -> DataFrame: ...
    def isin(
        self, values: AnyArrayLike | SequenceNotStr[Hashable]
    ) -> np_1darray_bool: ...

class CategoricalAccessor(
    PandasDelegate, NoNewAttributesMixin, Generic[CategoricalValueT]
):
    @property
    def codes(self) -> Series[int]: ...
    @property
    def categories(self) -> Index: ...
    @property
    def ordered(self) -> bool | None: ...
    def rename_categories(
        self,
        new_categories: (
            Sequence[CategoricalValueT]
            | dict[Any, CategoricalValueT]
            | Callable[[CategoricalValueT], CategoricalValueT]
        ),
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def reorder_categories(
        self,
        new_categories: ListLike,
        ordered: bool = ...,
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def add_categories(
        self, new_categories: Scalar | ListLike
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def remove_categories(
        self, removals: Scalar | ListLike
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def remove_unused_categories(
        self,
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def set_categories(
        self,
        new_categories: Sequence[CategoricalValueT] | ListLike,
        ordered: bool | None = False,
        rename: bool = False,
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def as_ordered(self) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    def as_unordered(self) -> Series[CategoricalDtype[CategoricalValueT]]: ...
