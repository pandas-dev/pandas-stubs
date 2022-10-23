from typing import (
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    Interval,
    IntervalIndex,
    PeriodIndex,
    Series,
)
from pandas.api.extensions import ExtensionArray

from pandas._typing import AnyArrayLike

# These are type: ignored because the Index types overlap due to inheritance but indices
# with extension types return the same type while standard type return ndarray

_IntervalT = TypeVar(
    "_IntervalT",
    Interval[int],
    Interval[float],
    Interval[pd.Timestamp],
    Interval[pd.Timedelta],
)

@overload
def unique(values: PeriodIndex) -> PeriodIndex: ...  # type: ignore[misc]
@overload
def unique(values: CategoricalIndex) -> CategoricalIndex: ...  # type: ignore[misc]
@overload
def unique(values: IntervalIndex[_IntervalT]) -> IntervalIndex[_IntervalT]: ...  # type: ignore[misc]
@overload
def unique(values: Index) -> np.ndarray: ...
@overload
def unique(values: Categorical) -> Categorical: ...
@overload
def unique(values: Series) -> np.ndarray | ExtensionArray: ...
@overload
def unique(values: np.ndarray | list) -> np.ndarray: ...
@overload
def unique(values: ExtensionArray) -> ExtensionArray: ...
@overload
def factorize(
    values: Sequence,
    sort: bool = ...,
    # Not actually positional-only, used to handle deprecations in 1.5.0
    *,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def factorize(
    values: Index | Series,
    sort: bool = ...,
    # Not actually positional-only, used to handle deprecations in 1.5.0
    *,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np.ndarray, Index]: ...
@overload
def factorize(
    values: Categorical,
    sort: bool = ...,
    # Not actually positional-only, used to handle deprecations in 1.5.0
    *,
    use_na_sentinel: bool = ...,
    size_hint: int | None = ...,
) -> tuple[np.ndarray, Categorical]: ...
def value_counts(
    values: AnyArrayLike | list | tuple,
    sort: bool = ...,
    ascending: bool = ...,
    normalize: bool = ...,
    bins: int | None = ...,
    dropna: bool = ...,
) -> Series: ...
