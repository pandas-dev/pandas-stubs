from typing import (
    Sequence,
    overload,
)

import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    PeriodIndex,
    RangeIndex,
    Series,
)
from pandas.api.extensions import ExtensionArray

from pandas._typing import (
    AnyArrayLike,
    npt,
)

@overload
def unique(values: DatetimeIndex) -> DatetimeIndex: ...
@overload
def unique(values: PeriodIndex) -> PeriodIndex: ...
@overload
def unique(values: CategoricalIndex) -> CategoricalIndex: ...
@overload
def unique(values: RangeIndex | pd.Float64Index) -> np.ndarray: ...
@overload
def unique(values: Index) -> Index | np.ndarray: ...
@overload
def unique(values: Categorical) -> Categorical: ...
@overload
def unique(values: Series) -> np.ndarray | ExtensionArray: ...
@overload
def unique(values: np.ndarray | list) -> np.ndarray: ...
@overload
def unique(values: ExtensionArray) -> ExtensionArray: ...
def factorize(
    values: Sequence | AnyArrayLike,
    sort: bool = ...,
    # Not actually positional-only, used to handle deprecations in 1.5.0
    *,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np.ndarray, np.ndarray | Index | Categorical]: ...
def value_counts(
    values: AnyArrayLike | list | tuple,
    sort: bool = ...,
    ascending: bool = ...,
    normalize: bool = ...,
    bins: int | None = ...,
    dropna: bool = ...,
) -> Series: ...
