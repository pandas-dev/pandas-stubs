from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexes.api import Index
from pandas.core.series import Series

from pandas._typing import (
    ArrayLike,
    Dtype,
    npt,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

def array(
    # str is forbidden even though Sequence[object] allows "abc"
    data: npt.NDArray | Sequence[object] | pd.Index | pd.Series,
    dtype: str
    | np.dtype[np.generic]
    | ExtensionDtype
    | type[str | bool | float | int]
    | None = ...,
    copy: bool = ...,
) -> ExtensionArray: ...
def extract_array(obj, extract_numpy: bool = ...): ...
def sanitize_array(
    data, index, dtype=..., copy: bool = ..., raise_cast_failure: bool = ...
): ...
def is_empty_data(data) -> bool: ...
def create_series_with_explicit_dtype(
    data=...,
    index: ArrayLike | Index | None = ...,
    dtype: Dtype | None = ...,
    name: str | None = ...,
    copy: bool = ...,
    fastpath: bool = ...,
    dtype_if_empty: Dtype = ...,
) -> Series: ...
