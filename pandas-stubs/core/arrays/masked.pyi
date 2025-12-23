from collections.abc import Iterator
from typing import (
    Any,
    overload,
)

from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.series import Series
from typing_extensions import Self

from pandas._typing import (
    DtypeArg,
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    np_1darray,
    np_1darray_bool,
    npt,
)

class BaseMaskedArray(ExtensionArray):
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __invert__(self) -> Self: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = False,
        na_value: Scalar = ...,
    ) -> np_1darray: ...
    __array_priority__: int = ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> Any: ...
    def copy(self) -> Self: ...
    def value_counts(self, dropna: bool = True) -> Series[int]: ...
    def isna(self) -> np_1darray_bool: ...
    @property
    def nbytes(self) -> int: ...
