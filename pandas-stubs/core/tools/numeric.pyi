from typing import (
    Literal,
    overload,
)

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

from pandas._libs.lib import NoDefault
from pandas._typing import (
    DtypeBackend,
    IgnoreRaiseCoerce,
    Scalar,
    npt,
)

_Downcast: TypeAlias = Literal["integer", "signed", "unsigned", "float"] | None

@overload
def to_numeric(
    arg: Scalar,
    errors: Literal["raise", "coerce"] = ...,
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> float: ...
@overload
def to_numeric(
    arg: Scalar,
    errors: Literal["ignore"],
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> Scalar: ...
@overload
def to_numeric(
    arg: list | tuple | np.ndarray,
    errors: IgnoreRaiseCoerce = ...,
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> npt.NDArray: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: IgnoreRaiseCoerce = ...,
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> pd.Series: ...
