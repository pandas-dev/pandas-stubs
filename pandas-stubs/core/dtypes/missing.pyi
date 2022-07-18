from __future__ import annotations

from typing import (
    List,
    Union,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)

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
def isna(obj: Index | list | ArrayLike) -> np.ndarray: ...
@overload
def isna(obj: Scalar) -> bool: ...

isnull = isna

@overload
def notna(obj: DataFrame) -> DataFrame: ...
@overload
def notna(obj: Series) -> Series[bool]: ...
@overload
def notna(obj: Index | list | ArrayLike) -> np.ndarray: ...
@overload
def notna(obj: Scalar) -> bool: ...

notnull = notna

def array_equivalent(left, right, strict_nan: bool = ...) -> bool: ...
def na_value_for_dtype(dtype, compat: bool = ...): ...
def remove_na_arraylike(arr): ...
def is_valid_nat_for_dtype(obj, dtype) -> bool: ...
