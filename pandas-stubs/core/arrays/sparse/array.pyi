import numpy as np
from pandas.core.arrays import (
    ExtensionArray,
    ExtensionOpsMixin,
)
from pandas.core.base import PandasObject as PandasObject

class SparseArray(PandasObject, ExtensionArray, ExtensionOpsMixin):
    def __init__(
        self,
        data,
        sparse_index=...,
        fill_value=...,
        kind: str = ...,
        dtype=...,
        copy: bool = ...,
    ) -> None: ...
    @classmethod
    def from_spmatrix(cls, data): ...
    def __array__(self, dtype=..., copy=...) -> np.ndarray: ...
    def __setitem__(self, key, value) -> None: ...
    @property
    def sp_index(self): ...
    @property
    def sp_values(self): ...
    @property
    def dtype(self): ...
    @property
    def fill_value(self): ...
    @fill_value.setter
    def fill_value(self, value) -> None: ...
    @property
    def kind(self) -> str: ...
    def __len__(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def density(self): ...
    @property
    def npoints(self) -> int: ...
    def isna(self): ...
    def fillna(self, value=..., method=..., limit=...): ...
    def shift(self, periods: int = ..., fill_value=...): ...
    def unique(self): ...
    def factorize(
        self,
        na_sentinel: int = ...,
        # Not actually positional-only, used to handle deprecations in 1.5.0
        *,
        use_na_sentinel: bool = ...,
    ): ...
    def value_counts(self, dropna: bool = ...): ...
    def __getitem__(self, key): ...
    def take(self, indices, allow_fill: bool = ..., fill_value=...): ...
    def searchsorted(self, v, side: str = ..., sorter=...): ...
    def copy(self): ...
    def astype(self, dtype=..., copy: bool = ...): ...
    def map(self, mapper): ...
    def to_dense(self): ...
    def nonzero(self): ...
    def all(self, axis=..., *args, **kwargs): ...
    def any(self, axis: int = ..., *args, **kwargs): ...
    def sum(self, axis: int = ..., *args, **kwargs): ...
    def cumsum(self, axis: int = ..., *args, **kwargs): ...
    def mean(self, axis: int = ..., *args, **kwargs): ...
    def transpose(self, *axes): ...
    @property
    def T(self): ...
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...
    def __abs__(self): ...

def make_sparse(arr, kind: str = ..., fill_value=..., dtype=..., copy: bool = ...): ...
