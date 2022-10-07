from typing import (
    Literal,
    Sequence,
    overload,
)

import numpy as np
from pandas import (
    Categorical,
    Float64Index,
    Index,
    Int64Index,
    IntervalIndex,
    Series,
)

from pandas._typing import (
    Label,
    npt,
)

@overload
def cut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: int | Series | Int64Index | Float64Index | Sequence[int] | Sequence[float],
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[npt.NDArray[np.intp], npt.NDArray]: ...
@overload
def cut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: IntervalIndex,
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[npt.NDArray[np.intp], IntervalIndex]: ...
@overload
def cut(
    x: Series,
    bins: int | Series | Int64Index | Float64Index | Sequence[int] | Sequence[float],
    right: bool = ...,
    labels: Literal[False] | Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, npt.NDArray]: ...
@overload
def cut(
    x: Series,
    bins: IntervalIndex,
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, IntervalIndex]: ...
@overload
def cut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: int | Series | Int64Index | Float64Index | Sequence[int] | Sequence[float],
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Categorical, npt.NDArray]: ...
@overload
def cut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: IntervalIndex,
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Categorical, IntervalIndex]: ...
@overload
def cut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: int
    | Series
    | Int64Index
    | Float64Index
    | Sequence[int]
    | Sequence[float]
    | IntervalIndex,
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[False] = ...,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> npt.NDArray[np.intp]: ...
@overload
def cut(
    x: Series,
    bins: int
    | Series
    | Int64Index
    | Float64Index
    | Sequence[int]
    | Sequence[float]
    | IntervalIndex,
    right: bool = ...,
    labels: Literal[False] | Sequence[Label] | None = ...,
    retbins: Literal[False] = ...,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> Series: ...
@overload
def cut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: int
    | Series
    | Int64Index
    | Float64Index
    | Sequence[int]
    | Sequence[float]
    | IntervalIndex,
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    retbins: Literal[False] = ...,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> Categorical: ...
@overload
def qcut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Float64Index | npt.NDArray,
    *,
    labels: Literal[False],
    retbins: Literal[False] = ...,
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> npt.NDArray[np.intp]: ...
@overload
def qcut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Float64Index | npt.NDArray,
    labels: Sequence[Label] | None = ...,
    retbins: Literal[False] = ...,
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> Categorical: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | Series[float] | Float64Index | npt.NDArray,
    labels: Literal[False] | Sequence[Label] | None = ...,
    retbins: Literal[False] = ...,
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> Series: ...
@overload
def qcut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Float64Index | npt.NDArray,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float_]]: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | Series[float] | Float64Index | npt.NDArray,
    labels: Literal[False] | Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> tuple[Series, npt.NDArray[np.float_]]: ...
@overload
def qcut(
    x: Index | npt.NDArray | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Float64Index | npt.NDArray,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> tuple[Categorical, npt.NDArray[np.float_]]: ...
