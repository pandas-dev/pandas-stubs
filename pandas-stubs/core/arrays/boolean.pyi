from typing import (
    Sequence,
    Union,
    overload,
)

import numpy as np
from pandas.core.arrays import ExtensionArray
from typing_extensions import TypeAlias

from pandas._libs.missing import NAType
from pandas._typing import (
    Scalar,
    npt,
    type_t,
)

from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype

from .masked import BaseMaskedArray as BaseMaskedArray

class BooleanDtype(ExtensionDtype):
    name: str = ...
    @property
    def na_value(self) -> Scalar: ...
    @property
    def type(self) -> type_t: ...
    @property
    def kind(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]: ...
    def __from_arrow__(self, array): ...

_ScalarType: TypeAlias = Union[bool, np.bool_, NAType, None]
_ArrayKey: TypeAlias = Union[Sequence[int], npt.NDArray[np.integer], slice]

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
        value: _ScalarType | Sequence[bool | NAType | None] | npt.NDArray[np.bool_],
    ) -> None: ...
    @overload  # type: ignore[override]
    def __getitem__(self, item: int) -> bool | NAType: ...
    @overload
    def __getitem__(self, item: _ArrayKey) -> BooleanArray: ...
    @property
    def dtype(self): ...
    def astype(
        self, dtype: str | np.dtype, copy: bool = ...
    ) -> np.ndarray | ExtensionArray: ...
    def any(self, skipna: bool = ..., **kwargs) -> bool: ...
    def all(self, skipna: bool = ..., **kwargs) -> bool: ...
