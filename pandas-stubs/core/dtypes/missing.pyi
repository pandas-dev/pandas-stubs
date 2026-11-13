from typing import (
    Any,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.arrays import ExtensionArray
from typing_extensions import (
    TypeIs,
    TypeVar,
)

from pandas._libs.missing import NAType
from pandas._libs.tslibs import NaTType
from pandas._typing import (
    ExtScalarOrNA,
    ExtScalarT,
    ScalarT,
    ShapeT,
    np_1darray_bool,
    np_ndarray,
    np_ndarray_bool,
)

# Bind the input type to preserve mypy's ndarray overload selection with Any.
_ObjectT = TypeVar("_ObjectT", bound=object)

@overload
def isna(obj: DataFrame) -> DataFrame: ...
@overload
def isna(obj: Series) -> Series[bool]: ...
@overload
def isna(obj: Index | ExtensionArray | list[ScalarT]) -> np_1darray_bool: ...
@overload
def isna(obj: np_ndarray[ShapeT]) -> np_ndarray_bool[ShapeT]: ...
@overload
def isna(obj: list[Any]) -> np_ndarray_bool: ...
@overload
def isna(obj: ExtScalarOrNA) -> TypeIs[NaTType | NAType | None]: ...
@overload
def isna(
    obj: _ObjectT,  # pyright: ignore[reportInvalidTypeVarUse]
) -> bool | np.bool_ | np_ndarray_bool | Series[bool] | DataFrame: ...

isnull = isna

@overload
def notna(obj: DataFrame) -> DataFrame: ...
@overload
def notna(obj: Series) -> Series[bool]: ...
@overload
def notna(obj: Index | ExtensionArray | list[ScalarT]) -> np_1darray_bool: ...
@overload
def notna(obj: np_ndarray[ShapeT]) -> np_ndarray_bool[ShapeT]: ...
@overload
def notna(obj: list[Any]) -> np_ndarray_bool: ...
@overload
def notna(obj: ExtScalarT | NaTType | NAType | None) -> TypeIs[ExtScalarT]: ...
@overload
def notna(
    obj: _ObjectT,  # pyright: ignore[reportInvalidTypeVarUse]
) -> bool | np.bool_ | np_ndarray_bool | Series[bool] | DataFrame: ...

notnull = notna
