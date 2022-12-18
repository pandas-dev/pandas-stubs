from typing import Sequence

import numpy as np

from pandas._typing import (
    ArrayLike,
    Scalar,
    npt,
)

from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.generic import ABCExtensionArray

class ExtensionArray:
    def __getitem__(self, item) -> None: ...
    def __setitem__(self, key: int | slice | np.ndarray, value) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = ...,
        na_value: Scalar = ...,
    ) -> np.ndarray: ...
    @property
    def dtype(self) -> ExtensionDtype: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def astype(self, dtype, copy: bool = ...): ...
    def isna(self) -> ArrayLike: ...
    def argsort(
        self, *, ascending: bool = ..., kind: str = ..., **kwargs
    ) -> np.ndarray: ...
    def fillna(self, value=..., method=..., limit=...): ...
    def dropna(self): ...
    def shift(
        self, periods: int = ..., fill_value: object = ...
    ) -> ABCExtensionArray: ...
    def unique(self): ...
    def searchsorted(self, value, side: str = ..., sorter=...): ...
    def factorize(
        self, use_na_sentinel: bool = ...
    ) -> tuple[np.ndarray, ABCExtensionArray]: ...
    def repeat(self, repeats, axis=...): ...
    def take(
        self, indices: Sequence[int], *, allow_fill: bool = ..., fill_value=...
    ) -> ABCExtensionArray: ...
    def copy(self) -> ABCExtensionArray: ...
    def view(self, dtype=...) -> ABCExtensionArray | np.ndarray: ...
    def ravel(self, order=...) -> ABCExtensionArray: ...

class ExtensionOpsMixin: ...
class ExtensionScalarOpsMixin(ExtensionOpsMixin): ...
