from typing import (
    Literal,
    overload,
)

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    RaiseCoerce,
    Scalar,
    npt,
)

_Downcast: TypeAlias = Literal["integer", "signed", "unsigned", "float"] | None

@overload
def to_numeric(
    arg: Scalar,
    errors: Literal["raise", "coerce"] = ...,
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> float: ...
@overload
def to_numeric(
    arg: list | tuple | np.ndarray,
    errors: RaiseCoerce = ...,
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> npt.NDArray: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: RaiseCoerce = ...,
    downcast: _Downcast = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> pd.Series: ...
