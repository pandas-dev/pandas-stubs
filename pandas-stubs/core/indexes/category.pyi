from collections.abc import (
    Hashable,
    Iterable,
)

import numpy as np
from pandas.core import accessor
from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.base import Index
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import Self

from pandas._typing import (
    S1,
    Dtype,
    ListLike,
)

class CategoricalIndex(ExtensionIndex[S1], accessor.PandasDelegate):
    codes: np.ndarray = ...
    categories: Index = ...
    @property
    def array(self) -> Categorical: ...  # type: ignore[override] # pyrefly: ignore[bad-override]
    def __new__(
        cls,
        data: Iterable[S1] = ...,
        categories: ListLike | None = None,
        ordered: bool | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self: ...
    @property
    def inferred_type(self) -> str: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    # `item` might be `S1` but not one of the categories, thus changing
    # the return type from `CategoricalIndex` to `Index`.
    def insert(self, loc: int, item: object) -> Index: ...  # type: ignore[override]
