from typing import (
    Literal,
    overload,
)

import numpy as np
import pandas as pd

from pandas._typing import (
    IgnoreRaiseCoerce,
    Scalar,
    npt,
)

@overload
def to_numeric(
    arg: Scalar,
    errors: Literal["raise", "coerce"] = ...,
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = ...,
) -> int | float: ...
@overload
def to_numeric(
    arg: Scalar,
    errors: Literal["ignore"],
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = ...,
) -> Scalar: ...
@overload
def to_numeric(
    arg: list | tuple | npt.NDArray,
    errors: IgnoreRaiseCoerce = ...,
    *,
    downcast: Literal["integer", "signed", "unsigned", "float"],
) -> npt.NDArray: ...
@overload
def to_numeric(
    arg: list | tuple | npt.NDArray,
    errors: Literal["ignore"],
    downcast: None = ...,
) -> npt.NDArray: ...
@overload
def to_numeric(
    arg: list | tuple | npt.NDArray,
    errors: Literal["raise", "coerce"] = ...,
    downcast: None = ...,
) -> npt.NDArray[np.intp] | npt.NDArray[np.float_]: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: IgnoreRaiseCoerce = ...,
    *,
    downcast: Literal["integer", "signed", "unsigned", "float"],
) -> pd.Series: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: Literal["ignore"],
    downcast: None = ...,
) -> pd.Series: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: Literal["raise", "coerce"] = ...,
    downcast: None = ...,
) -> pd.Series[int] | pd.Series[float]: ...
