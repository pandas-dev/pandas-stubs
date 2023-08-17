from typing import (
    Literal,
    Sequence,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.arrays import (
    ExtensionArray,
    PandasArray,
)
from typing_extensions import TypeAlias

from pandas._libs.missing import NAType
from pandas._typing import npt

from pandas.core.dtypes.base import ExtensionDtype

class StringDtype(ExtensionDtype):
    def __init__(self, storage: Literal["python", "pyarrow"] | None = None) -> None: ...
    @property
    def na_value(self) -> NAType: ...

_ScalarType: TypeAlias = str | NAType | None
_ArrayKey: TypeAlias = Sequence[int] | npt.NDArray[np.integer] | slice

class StringArray(PandasArray):
    def __init__(
        self,
        values: npt.NDArray[np.str_]
        | npt.NDArray[np.string_]
        | npt.NDArray[np.object_],
        copy: bool = ...,
    ) -> None: ...
    @property
    def na_value(self) -> NAType: ...
    def __arrow_array__(self, type=...): ...
    @overload  # type: ignore[override]
    def __setitem__(self, key: int, value: _ScalarType) -> None: ...
    @overload
    def __setitem__(
        self,
        key: _ArrayKey,
        value: _ScalarType
        | Sequence[str | NAType | None]
        | npt.NDArray[np.str_]
        | npt.NDArray[np.string_]
        | StringArray,
    ) -> None: ...
    @overload
    def __getitem__(self, item: int) -> str | NAType: ...
    @overload
    def __getitem__(self, item: _ArrayKey) -> StringArray: ...
    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...
    def min(
        self, axis: int | None = ..., skipna: bool = ..., **kwargs
    ) -> str | NAType: ...
    def max(
        self, axis: int | None = ..., skipna: bool = ..., **kwargs
    ) -> str | NAType: ...
    def value_counts(self, dropna: bool = ...) -> pd.Series[int]: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
