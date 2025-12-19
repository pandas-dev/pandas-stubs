from enum import Enum
import sys
from typing import (
    Any,
    final,
    overload,
)

import numpy as np
from pandas.core.arrays import (
    ExtensionArray,
    ExtensionOpsMixin,
)
from typing_extensions import Self

from pandas._libs.sparse import SparseIndex
from pandas._typing import (
    AnyArrayLike,
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    np_1darray,
)

from pandas.core.dtypes.dtypes import SparseDtype

@final
class ellipsis(Enum):
    Ellipsis = "..."

class SparseArray(ExtensionArray, ExtensionOpsMixin):
    if sys.version_info >= (3, 11):
        def __new__(
            cls,
            data: AnyArrayLike | Scalar,
            sparse_index: SparseIndex | None = None,
            fill_value: Scalar | None = None,
            kind: str = "integer",
            dtype: np.dtype | SparseDtype | None = ...,
            copy: bool = ...,
        ) -> Self: ...
    else:
        def __new__(
            cls,
            data: AnyArrayLike | Scalar,
            sparse_index: SparseIndex | None = None,
            fill_value: Scalar | None = None,
            kind: str = "integer",
            dtype: np.dtype[Any] | SparseDtype | None = ...,
            copy: bool = ...,
        ) -> Self: ...

    @classmethod
    def from_spmatrix(cls, data: Any) -> Self: ...
    @property
    def sp_index(self) -> SparseIndex: ...
    @property
    def sp_values(self) -> np.ndarray: ...
    @property
    def dtype(self) -> SparseDtype: ...
    @property
    def fill_value(self) -> Any: ...
    @fill_value.setter
    def fill_value(self, value: Any) -> None: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @property
    def kind(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def density(self) -> float: ...
    @property
    def npoints(self) -> int: ...
    @overload
    def __getitem__(  # pyrefly: ignore[bad-override,bad-param-name-override]
        self, key: ScalarIndexer
    ) -> Any: ...
    @overload
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, key: SequenceIndexer | tuple[int | ellipsis, ...]
    ) -> Self: ...
