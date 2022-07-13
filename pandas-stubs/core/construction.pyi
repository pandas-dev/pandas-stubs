from typing import (
    Optional,
    Sequence,
    Union,
)

import numpy as np
from pandas.core.indexes.api import Index
from pandas.core.series import Series

from pandas._typing import (
    ArrayLike,
    Dtype,
)

from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCExtensionArray

def array(
    data: Sequence[object],
    dtype: Optional[Union[str, np.dtype, ExtensionDtype]] = ...,
    copy: bool = ...,
) -> ABCExtensionArray: ...
def extract_array(obj, extract_numpy: bool = ...): ...
def sanitize_array(
    data, index, dtype=..., copy: bool = ..., raise_cast_failure: bool = ...
): ...
def is_empty_data(data) -> bool: ...
def create_series_with_explicit_dtype(
    data=...,
    index: Optional[Union[ArrayLike, Index]] = ...,
    dtype: Optional[Dtype] = ...,
    name: Optional[str] = ...,
    copy: bool = ...,
    fastpath: bool = ...,
    dtype_if_empty: Dtype = ...,
) -> Series: ...
