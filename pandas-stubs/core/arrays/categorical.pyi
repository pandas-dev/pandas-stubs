from collections.abc import (
    Callable,
    Hashable,
    Sequence,
)
from typing import (
    Any,
    Generic,
    Literal,
    Never,
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
    np_1darray_str,
    np_ndarray_anyint,
    np_ndarray_float,
)

from pandas.core.dtypes.dtypes import (
    CategoricalDtype as CategoricalDtype,
    CategoricalValueT,
    CategoricalValueT1,
)

class Categorical(NDArrayBackedExtensionArray, Generic[CategoricalValueT]):
    __array_priority__: int = ...
    @overload
    def __new__(  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
        cls,
        values: Sequence[Never],
        categories: SequenceNotStr[Hashable] | AnyArrayLike | None = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype | None = None,
        copy: bool = True,
    ) -> Categorical: ...
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]
        cls,
        values: (
            list[str] | np_1darray_str | SequenceNotStr[str] | Series[str] | Index[str]
        ),
        categories: (
            SequenceNotStr[str] | Series[str] | Index[str] | np_1darray_str | None
        ) = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype | None = None,
        copy: bool = True,
    ) -> Categorical[str]: ...
    @overload
    def __new__(
        cls,
        values: Sequence[int] | np_ndarray_anyint | Series[int] | Index[int],
        categories: (
            SequenceNotStr[int] | Series[int] | Index[int] | np_ndarray_anyint | None
        ) = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype | None = None,
        copy: bool = True,
    ) -> Categorical[int]: ...
    @overload
    def __new__(
        cls,
        values: Sequence[float] | np_ndarray_float | Series[float] | Index[float],
        categories: (
            SequenceNotStr[float]
            | Series[float]
            | Index[float]
            | np_ndarray_float
            | None
        ) = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype | None = None,
        copy: bool = True,
    ) -> Categorical[float]: ...
    @overload
    def __new__(
        cls,
        values: Categorical[CategoricalValueT1],
        categories: (
            SequenceNotStr[CategoricalValueT1]
            | Series[CategoricalValueT1]
            | Index[CategoricalValueT1]
            | np_1darray_str
            | None
        ) = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype | None = None,
        copy: bool = True,
    ) -> Categorical[CategoricalValueT1]: ...
    @overload
    def __new__(
        cls,
        values: SequenceNotStr[Hashable] | AnyArrayLike,
        categories: SequenceNotStr[Hashable] | AnyArrayLike | None = None,
        ordered: bool | None = None,
        dtype: CategoricalDtype | None = None,
        copy: bool = True,
    ) -> Categorical: ...
    @property
    def categories(self) -> Index: ...
    @property
    def ordered(self) -> Ordered: ...
    @property
    def dtype(self) -> CategoricalDtype[CategoricalValueT]: ...
    def tolist(self) -> list[CategoricalValueT]: ...
    @overload
    @classmethod
    def from_codes(  # pyright: ignore[reportOverlappingOverload]
        cls,
        codes: Series[int] | Index[int] | np_ndarray_anyint | Sequence[int],
        categories: Index[CategoricalValueT1],
        ordered: bool | None = ...,
        dtype: CategoricalDtype[CategoricalValueT1] | None = ...,
        validate: bool = True,
    ) -> Categorical[CategoricalValueT1]: ...
    @overload
    @classmethod
    def from_codes(
        cls,
        codes: Series[int] | Index[int] | np_ndarray_anyint | Sequence[int],
        categories: Index | None = ...,
        ordered: bool | None = ...,
        dtype: CategoricalDtype[int] | None = ...,
        validate: bool = True,
    ) -> Categorical[int]: ...
    @property
    def codes(self) -> np_1darray[np.signedinteger]: ...
    def set_ordered(self, value: bool) -> Self: ...
    def as_ordered(self) -> Self: ...
    def as_unordered(self) -> Self: ...
    def set_categories(
        self,
        new_categories: AnyArrayLike | SequenceNotStr[Hashable],
        ordered: bool | None = False,
        rename: bool = False,
    ) -> Self: ...
    def rename_categories(self, new_categories: Renamer) -> Self: ...
    def reorder_categories(
        self,
        new_categories: SequenceNotStr[Hashable] | AnyArrayLike,
        ordered: bool | None = None,
    ) -> Self: ...
    def add_categories(
        self, new_categories: AnyArrayLike | SequenceNotStr[Hashable]
    ) -> Self: ...
    def remove_categories(
        self, removals: Hashable | SequenceNotStr[Hashable] | AnyArrayLike
    ) -> Self: ...
    def remove_unused_categories(self) -> Self: ...
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
    ) -> CategoricalValueT | NAType: ...
    @overload
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, key: SequenceIndexer | PositionalIndexerTuple
    ) -> Self: ...
    def min(
        self, *, skipna: bool = True, **kwargs: Any
    ) -> CategoricalValueT | NAType: ...
    def max(
        self, *, skipna: bool = True, **kwargs: Any
    ) -> CategoricalValueT | NAType: ...
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
        new_categories: Sequence[CategoricalValueT] | AnyArrayLike,
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
