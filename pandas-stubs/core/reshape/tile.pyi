from typing import (
    Literal,
    Sequence,
    overload,
)

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
    x: Series | Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: int | Series | Int64Index | Float64Index | Sequence[int] | Sequence[float],
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[npt.NDArray, npt.NDArray]: ...
@overload
def cut(
    x: Series | Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: IntervalIndex,
    right: bool = ...,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[npt.NDArray, IntervalIndex]: ...
@overload
def cut(
    x: Categorical,
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
    x: Categorical,
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
    x: Series | Index | npt.NDArray | Sequence[int] | Sequence[float],
    bins: int | Series | Int64Index | Float64Index | Sequence[int] | Sequence[float],
    right: bool = ...,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: Literal["raise", "drop"] = ...,
    ordered: bool = ...,
) -> tuple[Series, npt.NDArray]: ...
@overload
def cut(
    x: Series | Index | npt.NDArray | Sequence[int] | Sequence[float],
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
    x: Series | Index | npt.NDArray | Sequence[int] | Sequence[float],
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
) -> npt.NDArray: ...
@overload
def cut(
    x: Categorical,
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
def cut(
    x: Series | Index | npt.NDArray | Sequence[int] | Sequence[float],
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
) -> Series: ...
@overload
def qcut(
    x: npt.NDArray | Series,
    q: int | Sequence[float] | Series[float] | Float64Index,
    *,
    labels: Literal[False],
    retbins: Literal[False] = ...,
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> npt.NDArray: ...
@overload
def qcut(
    x: npt.NDArray,
    q: int | Sequence[float] | Series[float] | Float64Index,
    labels: Sequence[Label] | None = ...,
    retbins: Literal[False] = ...,
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> Categorical: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | Series[float] | Float64Index,
    labels: Sequence[Label] | None = ...,
    retbins: Literal[False] = ...,
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> Series: ...
@overload
def qcut(
    x: npt.NDArray | Series,
    q: int | Sequence[float] | Series[float] | Float64Index,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> tuple[npt.NDArray, npt.NDArray]: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | Series[float] | Float64Index,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> tuple[Series, npt.NDArray]: ...
@overload
def qcut(
    x: npt.NDArray,
    q: int | Sequence[float] | Series[float] | Float64Index,
    labels: Sequence[Label] | None = ...,
    *,
    retbins: Literal[True],
    precision: int = ...,
    duplicates: Literal["raise", "drop"] = ...,
) -> tuple[Categorical, npt.NDArray]: ...
