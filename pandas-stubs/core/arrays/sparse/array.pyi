from enum import Enum
import sys
from typing import (
    Any,
    Literal,
    Protocol,
    TypeAlias,
    final,
    overload,
    type_check_only,
)

import numpy as np
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.sparse import SparseIndex
from pandas._typing import (
    AnyArrayLike,
    Axis,
    AxisInt,
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    np_1darray,
    np_1darray_intp,
    np_ndarray,
)

from pandas.core.dtypes.dtypes import SparseDtype

SparseIndexKind: TypeAlias = Literal["integer", "block"]

@type_check_only
class _SparseMatrixLike(Protocol):
    @property
    def shape(self, /) -> tuple[int, int]: ...

@final
class ellipsis(Enum):
    Ellipsis = "..."

class SparseArray(OpsMixin, ExtensionArray):
    if sys.version_info >= (3, 11):
        def __new__(
            cls,
            data: AnyArrayLike | Scalar,
            sparse_index: SparseIndex | None = None,
            fill_value: Scalar | None = None,
            kind: SparseIndexKind = "integer",
            dtype: np.dtype | SparseDtype | None = None,
            copy: bool = False,
        ) -> Self: ...
    else:
        def __new__(
            cls,
            data: AnyArrayLike | Scalar,
            sparse_index: SparseIndex | None = None,
            fill_value: Scalar | None = None,
            kind: SparseIndexKind = "integer",
            dtype: np.dtype[Any] | SparseDtype | None = None,
            copy: bool = False,
        ) -> Self: ...

    @classmethod
    def from_spmatrix(cls, data: _SparseMatrixLike) -> Self: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @property
    def sp_index(self) -> SparseIndex: ...
    @property
    def sp_values(self) -> np_ndarray: ...
    @property
    def dtype(self) -> SparseDtype: ...
    @property
    def fill_value(self) -> Any: ...
    @fill_value.setter
    def fill_value(self, value: Any) -> None: ...
    @property
    def kind(self) -> SparseIndexKind: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def density(self) -> float: ...
    @property
    def npoints(self) -> int: ...
    def shift(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        self, periods: int = 1, fill_value: Scalar | None = None
    ) -> Self: ...
    def unique(self) -> Self: ...
    def value_counts(self, dropna: bool = True) -> Series[int]: ...
    @overload
    def __getitem__(  # pyrefly: ignore[bad-param-name-override]
        self, key: ScalarIndexer
    ) -> Any: ...
    @overload
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, key: SequenceIndexer | tuple[int | ellipsis, ...]
    ) -> Self: ...
    def copy(self) -> Self: ...
    def to_dense(self) -> np_1darray: ...
    def nonzero(self) -> tuple[np_1darray_intp]: ...
    def all(self, axis: Axis | None = None, *args: Any, **kwargs: Any) -> bool: ...
    def any(self, axis: AxisInt = 0, *args: Any, **kwargs: Any) -> bool: ...
    def sum(
        self,
        axis: AxisInt = 0,
        min_count: int = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Scalar: ...
    def cumsum(self, axis: AxisInt = 0, *args: Any, **kwargs: Any) -> Self: ...
    def mean(self, axis: AxisInt = 0, *args: Any, **kwargs: Any) -> Self: ...
    @property
    def T(self) -> Self: ...
    def __abs__(self) -> Self: ...
