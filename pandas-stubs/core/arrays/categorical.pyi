from collections.abc import (
    Callable,
    Hashable,
    Sequence,
)
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Never,
    Self,
    overload,
)

import numpy as np
from pandas.core.accessor import PandasDelegate as PandasDelegate
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import override

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
    np_ndarray_anyint,
    np_ndarray_float,
    np_ndarray_str,
)

from pandas.core.dtypes.dtypes import (
    CategoricalDtype as CategoricalDtype,
    CategoricalValueT,
    CategoricalValueT1,
)

class Categorical(NDArrayBackedExtensionArray, Generic[CategoricalValueT]):
    __array_priority__: int = ...
    __hash__: ClassVar[None]  # type: ignore[assignment] # pyright: ignore[reportIncompatibleMethodOverride]
    @overload
    def __new__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
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
            list[str] | np_ndarray_str | SequenceNotStr[str] | Series[str] | Index[str]
        ),
        categories: (
            SequenceNotStr[str] | Series[str] | Index[str] | np_ndarray_str | None
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
            Sequence[int] | Series[int] | Index[int] | np_ndarray_anyint | None
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
            Sequence[float] | Series[float] | Index[float] | np_ndarray_float | None
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
            | np_ndarray_str
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
    @override
    def dtype(self) -> CategoricalDtype[CategoricalValueT]: ...
    @override
    def tolist(self) -> list[CategoricalValueT]: ...
    @overload
    @classmethod
    def from_codes(
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
        categories: None = None,
        ordered: bool | None = ...,
        dtype: CategoricalDtype[int] | None = ...,
        validate: bool = True,
    ) -> Categorical[int]: ...
    @overload
    @classmethod
    def from_codes(
        cls,
        codes: Series[int] | Index[int] | np_ndarray_anyint | Sequence[int],
        categories: None = ...,
        ordered: bool | None = ...,
        *,
        dtype: CategoricalDtype[CategoricalValueT],
        validate: bool = True,
    ) -> Categorical[CategoricalValueT]: ...
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
    @overload
    def reorder_categories(
        self: Categorical[CategoricalValueT1],
        new_categories: (
            SequenceNotStr[CategoricalValueT1]
            | Index[CategoricalValueT1]
            | Series[CategoricalValueT1]
        ),
        ordered: bool | None = None,
    ) -> Categorical[CategoricalValueT1]: ...
    @overload
    def reorder_categories(
        self,
        new_categories: AnyArrayLike,
        ordered: bool | None = None,
    ) -> Categorical: ...
    def add_categories(
        self, new_categories: AnyArrayLike | SequenceNotStr[Hashable]
    ) -> Self: ...
    def remove_categories(
        self, removals: Hashable | SequenceNotStr[Hashable] | AnyArrayLike
    ) -> Self: ...
    @overload  # type: ignore[override]
    @override
    def __eq__(self, other: Series) -> Series[bool]: ...  # type: ignore[overload-overlap] # pyrefly: ignore[bad-override]
    @overload
    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride,reportOverlappingOverload] # ty: ignore[invalid-method-override]
        self, other: object
    ) -> np_1darray_bool: ...
    @overload  # type: ignore[override]
    @override
    def __ne__(self, other: Series) -> Series[bool]: ...  # type: ignore[overload-overlap] # pyrefly: ignore[bad-override]
    @overload
    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride,reportOverlappingOverload] # ty: ignore[invalid-method-override]
        self, other: object
    ) -> np_1darray_bool: ...
    @overload
    def __lt__(self, other: Self) -> np_1darray_bool: ...
    @overload
    def __lt__(  # type: ignore[overload-overlap]
        self: Categorical[CategoricalValueT1], other: Series[CategoricalValueT1]
    ) -> Series[bool]: ...
    @overload
    def __lt__(self, other: CategoricalValueT) -> np_1darray_bool: ...
    @overload
    def __le__(self, other: Self) -> np_1darray_bool: ...
    @overload
    def __le__(  # type: ignore[overload-overlap]
        self: Categorical[CategoricalValueT1], other: Series[CategoricalValueT1]
    ) -> Series[bool]: ...
    @overload
    def __le__(self, other: CategoricalValueT) -> np_1darray_bool: ...
    @overload
    def __gt__(self, other: Self) -> np_1darray_bool: ...
    @overload
    def __gt__(  # type: ignore[overload-overlap]
        self: Categorical[CategoricalValueT1], other: Series[CategoricalValueT1]
    ) -> Series[bool]: ...
    @overload
    def __gt__(self, other: CategoricalValueT) -> np_1darray_bool: ...
    @overload
    def __ge__(self, other: Self) -> np_1darray_bool: ...
    @overload
    def __ge__(  # type: ignore[overload-overlap]
        self: Categorical[CategoricalValueT1], other: Series[CategoricalValueT1]
    ) -> Series[bool]: ...
    @overload
    def __ge__(self, other: CategoricalValueT) -> np_1darray_bool: ...
    def remove_unused_categories(self) -> Self: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    def memory_usage(self, deep: bool = False) -> int: ...
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
    @override
    def __contains__(self, item: Hashable) -> bool: ...
    @overload
    @override
    # pyrefly: ignore[bad-override]
    def __getitem__(self, key: ScalarIndexer) -> CategoricalValueT | NAType: ...
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

class CategoricalAccessor(PandasDelegate, Generic[CategoricalValueT]):
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
    @overload
    def reorder_categories(
        self,
        new_categories: Sequence[CategoricalValueT],
        ordered: bool = ...,
    ) -> Series[CategoricalDtype[CategoricalValueT]]: ...
    @overload
    def reorder_categories(
        self,
        new_categories: AnyArrayLike,
        ordered: bool = ...,
    ) -> Series[CategoricalDtype]: ...
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
