from typing import (
    Literal,
    Sequence,
    overload,
)

import numpy as np
import pandas as pd
from pandas.arrays import (
    BooleanArray,
    DatetimeArray,
    StringArray,
)
from pandas.core.arrays.masked import BaseMaskedArray

from pandas._libs.missing import NAType
from pandas._typing import npt

from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype

class _IntegerDtype(ExtensionDtype):
    base: None
    @property
    def na_value(self) -> NAType: ...
    @property
    def itemsize(self) -> int: ...
    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]: ...
    def __from_arrow__(self, array): ...

class IntegerArray(BaseMaskedArray):
    def dtype(self): ...
    def __init__(
        self,
        values: npt.NDArray[np.integer],
        mask: npt.NDArray[np.bool_] | Sequence[bool],
        copy: bool = ...,
    ) -> None: ...
    @overload  # type: ignore[override]
    def __setitem__(self, key: int, value: float | NAType) -> None: ...
    @overload
    def __setitem__(
        self,
        key: Sequence[int] | slice | npt.NDArray[np.integer],
        value: float | NAType | Sequence[float | NAType] | npt.NDArray[np.integer],
    ) -> None: ...
    @overload
    def __getitem__(self, item: int) -> int | NAType: ...
    @overload
    def __getitem__(
        self, item: slice | list[int] | npt.NDArray[np.integer]
    ) -> IntegerArray: ...
    # Note: the ignores are needed below due to types being subclasses,
    # e.g., float32 and float64 or bool, int, float, complex
    @overload
    def astype(  # type: ignore[misc]
        self, dtype: Literal["str"] | type[str] | np.str_
    ) -> npt.NDArray[np.str_]: ...
    @overload
    def astype(  # type: ignore[misc]
        self, dtype: type[bool | np.bool_] | Literal["bool"], copy: bool = ...
    ) -> npt.NDArray[np.bool_]: ...
    @overload
    def astype(
        self,
        dtype: Literal["i1", "i2", "i4", "i8", "int8", "int16", "int32", "int64"]
        | type[int | np.signedinteger],
        copy: bool = ...,
    ) -> npt.NDArray[np.signedinteger]: ...
    @overload
    def astype(
        self,
        dtype: Literal["u1", "u2", "u4", "u8", "uint8", "uint16", "uint32", "uint64"]
        | type[np.unsignedinteger],
        copy: bool = ...,
    ) -> npt.NDArray[np.unsignedinteger]: ...
    @overload
    def astype(  # type: ignore[misc]
        self, dtype: Literal["f4", "float32"] | type[np.float32]
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def astype(
        self, dtype: Literal["float", "float64", "f8"] | type[np.float64 | float]
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def astype(  # type: ignore[misc]
        self, dtype: Literal["complex64", "c8"] | type[np.complex64]
    ) -> npt.NDArray[np.complex64]: ...
    @overload
    def astype(
        self,
        dtype: Literal["complex", "complex128", "c16"] | type[np.complex128 | complex],
    ) -> npt.NDArray[np.complex128]: ...
    @overload
    def astype(self, dtype: Literal["boolean"] | pd.BooleanDtype) -> BooleanArray: ...
    @overload
    def astype(
        self,
        dtype: Literal[
            "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"
        ]
        | pd.Int8Dtype
        | pd.Int16Dtype
        | pd.Int32Dtype
        | pd.Int64Dtype
        | pd.UInt8Dtype
        | pd.UInt16Dtype
        | pd.UInt32Dtype
        | pd.UInt64Dtype,
    ) -> IntegerArray: ...
    @overload
    def astype(self, dtype: Literal["string"] | pd.StringDtype) -> StringArray: ...
    @overload
    def astype(
        self, dtype: type[np.datetime64] | Literal["M8[ns]"]
    ) -> npt.NDArray[np.datetime64]: ...
    @overload
    def astype(self, dtype: pd.DatetimeTZDtype) -> DatetimeArray: ...

class Int8Dtype(_IntegerDtype): ...
class Int16Dtype(_IntegerDtype): ...
class Int32Dtype(_IntegerDtype): ...
class Int64Dtype(_IntegerDtype): ...
class UInt8Dtype(_IntegerDtype): ...
class UInt16Dtype(_IntegerDtype): ...
class UInt32Dtype(_IntegerDtype): ...
class UInt64Dtype(_IntegerDtype): ...
