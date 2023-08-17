from typing import (
    ClassVar,
    Sequence,
    overload,
)

import numpy as np
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from typing_extensions import TypeAlias

from pandas._libs.missing import NAType
from pandas._typing import (
    npt,
    type_t,
)

from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype

class BooleanDtype(ExtensionDtype):
    na_value: ClassVar[NAType]
    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]: ...

_ScalarType: TypeAlias = bool | np.bool_ | NAType | None
_ArrayKey: TypeAlias = Sequence[int] | npt.NDArray[np.integer] | slice

class BooleanArray(BaseMaskedArray):
    def __init__(
        self,
        values: npt.NDArray[np.bool_],
        mask: npt.NDArray[np.bool_],
        copy: bool = ...,
    ) -> None: ...
    # Ignore overrides since more specific than super type
    @overload  # type: ignore[override]
    def __setitem__(self, key: int, value: _ScalarType) -> None: ...
    @overload
    def __setitem__(
        self,
        key: _ArrayKey,
        value: _ScalarType
        | Sequence[bool | NAType | None]
        | npt.NDArray[np.bool_]
        | BooleanArray,
    ) -> None: ...
    @overload  # type: ignore[override]
    def __getitem__(self, item: int) -> bool | NAType: ...
    @overload
    def __getitem__(self, item: _ArrayKey) -> BooleanArray: ...
    @property
    def dtype(self) -> ExtensionDtype: ...
    def astype(
        self, dtype: npt.DTypeLike | ExtensionDtype, copy: bool = ...
    ) -> np.ndarray | ExtensionArray: ...
    def any(self, skipna: bool = ..., **kwargs) -> bool: ...
    def all(self, skipna: bool = ..., **kwargs) -> bool: ...
    def __setitem__(self, key, value) -> None: ...
