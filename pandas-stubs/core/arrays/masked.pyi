import numpy as np
from pandas.core.arrays import (
    ExtensionArray as ExtensionArray,
    ExtensionOpsMixin,
)

from pandas._typing import Scalar

class BaseMaskedArray(ExtensionArray, ExtensionOpsMixin):
    def __getitem__(self, item): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __invert__(self): ...
    def to_numpy(self, dtype=..., copy=..., na_value: Scalar = ...): ...
    __array_priority__: int = ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __arrow_array__(self, type=...): ...
    def isna(self): ...
    @property
    def nbytes(self) -> int: ...
    def take(self, indexer, allow_fill: bool = ..., fill_value=...): ...
    def copy(self): ...
    def value_counts(self, dropna: bool = ...): ...
