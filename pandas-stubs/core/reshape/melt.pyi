import numpy as np
from pandas.core.frame import DataFrame

def melt(
    frame: DataFrame,
    id_vars: tuple | list | np.ndarray | None = ...,
    value_vars: tuple | list | np.ndarray | None = ...,
    var_name: str | None = ...,
    value_name: str = ...,
    col_level: int | str | None = ...,
    ignore_index: bool = ...,
) -> DataFrame: ...
def lreshape(data: DataFrame, groups, dropna: bool = ..., label=...) -> DataFrame: ...
def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = ..., suffix: str = ...
) -> DataFrame: ...
