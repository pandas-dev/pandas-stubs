from typing import (
    Literal,
    overload,
)

import numpy as np
from numpy import typing as npt
from pandas import (
    DataFrame,
    Index,
    Series,
)

from pandas._libs.missing import NAType
from pandas._libs.tslibs import NaTType
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
def isna(obj: Scalar) -> bool: ...
@overload
def isna(obj: NaTType | NAType) -> Literal[True]: ...

isnull = isna

@overload
def notna(obj: DataFrame) -> DataFrame: ...
@overload
def notna(obj: Series) -> Series[bool]: ...
@overload
def notna(obj: Index | list | ArrayLike) -> npt.NDArray[np.bool_]: ...
@overload
def notna(obj: Scalar) -> bool: ...
@overload
def notna(obj: NaTType | NAType) -> Literal[False]: ...

notnull = notna
