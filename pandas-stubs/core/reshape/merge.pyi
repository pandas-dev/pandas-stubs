from datetime import timedelta
from typing import (
    Literal,
    overload,
)

from pandas import (
    DataFrame,
    Series,
    Timedelta,
)

from pandas._typing import (
    AnyArrayLike,
    HashableT,
    JoinHow,
    Label,
    MergeHow,
    Suffixes,
    ValidationOptions,
)

def merge(
    left: DataFrame | Series,
    right: DataFrame | Series,
    how: MergeHow = "inner",
    on: Label | list[HashableT] | AnyArrayLike | None = None,
    left_on: Label | list[HashableT] | AnyArrayLike | None = None,
    right_on: Label | list[HashableT] | AnyArrayLike | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Suffixes = ...,
    indicator: bool | str = False,
    validate: ValidationOptions | None = None,
) -> DataFrame: ...
@overload
def merge_ordered(
    left: DataFrame,
    right: DataFrame,
    on: Label | list[HashableT] | None = None,
    left_on: Label | list[HashableT] | None = None,
    right_on: Label | list[HashableT] | None = None,
    left_by: Label | list[HashableT] | None = None,
    right_by: Label | list[HashableT] | None = None,
    fill_method: Literal["ffill"] | None = None,
    suffixes: Suffixes = ("_x", "_y"),
    how: JoinHow = "outer",
) -> DataFrame: ...
@overload
def merge_ordered(
    left: Series,
    right: DataFrame | Series,
    on: Label | list[HashableT] | None = None,
    left_on: Label | list[HashableT] | None = None,
    right_on: Label | list[HashableT] | None = None,
    left_by: None = None,
    right_by: None = None,
    fill_method: Literal["ffill"] | None = None,
    suffixes: (
        list[str | None] | tuple[str, str] | tuple[None, str] | tuple[str, None]
    ) = ("_x", "_y"),
    how: JoinHow = "outer",
) -> DataFrame: ...
@overload
def merge_ordered(
    left: DataFrame | Series,
    right: Series,
    on: Label | list[HashableT] | None = None,
    left_on: Label | list[HashableT] | None = None,
    right_on: Label | list[HashableT] | None = None,
    left_by: None = None,
    right_by: None = None,
    fill_method: Literal["ffill"] | None = None,
    suffixes: Suffixes = ("_x", "_y"),
    how: JoinHow = "outer",
) -> DataFrame: ...
def merge_asof(
    left: DataFrame | Series,
    right: DataFrame | Series,
    on: Label | None = None,
    left_on: Label | None = None,
    right_on: Label | None = None,
    left_index: bool = False,
    right_index: bool = False,
    by: Label | list[HashableT] | None = None,
    left_by: Label | list[HashableT] | None = None,
    right_by: Label | list[HashableT] | None = None,
    suffixes: Suffixes = ("_x", "_y"),
    tolerance: int | timedelta | Timedelta | None = None,
    allow_exact_matches: bool = True,
    direction: Literal["backward", "forward", "nearest"] = "backward",
) -> DataFrame: ...
