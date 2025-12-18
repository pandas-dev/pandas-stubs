from enum import Enum
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
    def __init__(
        self,
        data: AnyArrayLike | Scalar,
        sparse_index: SparseIndex | None = None,
        fill_value: Scalar | None = None,
        kind: str = "integer",
        dtype: np.dtype | SparseDtype | None = ...,
        copy: bool = ...,
    ) -> None: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @property
    def kind(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
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
