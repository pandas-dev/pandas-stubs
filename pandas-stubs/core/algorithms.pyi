from typing import (
    Any,
    overload,
)

import numpy as np
from pandas import (
    Categorical,
    Index,
    Series,
)
from pandas.api.extensions import ExtensionArray

@overload
def unique(values: Index) -> Index: ...
@overload
def unique(values: Categorical) -> Categorical: ...
@overload
def unique(values: Series) -> np.ndarray | ExtensionArray: ...
@overload
def unique(values: np.ndarray | list) -> np.ndarray: ...
@overload
def unique(values: ExtensionArray) -> ExtensionArray: ...
def factorize(
    values: Any,
    sort: bool = ...,
    na_sentinel: int | None = ...,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np.ndarray, np.ndarray | Index]: ...
