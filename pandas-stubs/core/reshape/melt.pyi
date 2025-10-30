from collections.abc import (
    Hashable,
    Sequence,
)
from typing import Any

from numpy import typing as npt
from pandas.core.frame import DataFrame

from pandas._typing import HashableT

def melt(
    frame: DataFrame,
    id_vars: Sequence[Hashable] | npt.NDArray[Any] | None = None,
    value_vars: Sequence[Hashable] | npt.NDArray[Any] | None = None,
    var_name: str | None = None,
    value_name: Hashable = "value",
    col_level: int | str | None = None,
    ignore_index: bool = True,
) -> DataFrame: ...
def lreshape(
    data: DataFrame,
    groups: dict[HashableT, list[HashableT]],
    dropna: bool = True,
) -> DataFrame: ...
def wide_to_long(
    df: DataFrame,
    stubnames: str | list[str],
    i: str | list[str],
    j: str,
    sep: str = "",
    suffix: str = "\\d+",
) -> DataFrame: ...
