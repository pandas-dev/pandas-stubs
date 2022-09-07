from typing import overload

import numpy as np
from numpy import typing as npt
from pandas import (
    DataFrame,
    Index,
    Series,
)

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    ArrayLike,
    Scalar,
)

isposinf_scalar = ...
isneginf_scalar = ...

@overload
def isna(obj: DataFrame) -> DataFrame: ...
@overload
def isna(obj: Series) -> Series[bool]: ...
@overload
def isna(obj: Index | list | ArrayLike) -> npt.NDArray[np.bool_]: ...
@overload
def isna(obj: Scalar | NaTType | NAType) -> bool: ...

isnull = isna

@overload
def notna(obj: DataFrame) -> DataFrame: ...
@overload
def notna(obj: Series) -> Series[bool]: ...
@overload
def notna(obj: Index | list | ArrayLike) -> np.ndarray: ...
@overload
def notna(obj: Scalar | NaTType | NAType) -> bool: ...

notnull = notna

def array_equivalent(left, right, strict_nan: bool = ...) -> bool: ...
def na_value_for_dtype(dtype, compat: bool = ...): ...
def remove_na_arraylike(arr): ...
