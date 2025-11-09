from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
import pandas as pd

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    RaiseCoerce,
    Scalar,
    SequenceNotStr,
    np_1darray,
    np_ndarray,
)

_Downcast: TypeAlias = Literal["integer", "signed", "unsigned", "float"] | None

@overload
def to_numeric(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    arg: Scalar,
    errors: RaiseCoerce = "raise",
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
) -> float: ...
@overload
def to_numeric(
    arg: Sequence[int],
    errors: RaiseCoerce = "raise",
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
) -> np_1darray[np.integer]: ...
@overload
def to_numeric(
    arg: SequenceNotStr[Any] | np_ndarray,
    errors: RaiseCoerce = "raise",
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
) -> np_1darray[np.floating]: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: RaiseCoerce = "raise",
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
) -> pd.Series: ...
