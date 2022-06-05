import numpy as np
from pandas.core.frame import DataFrame as DataFrame
from typing import List, Optional, Tuple, Union

def melt(
    frame: DataFrame,
    id_vars: Optional[Union[Tuple, List, np.ndarray]] = ...,
    value_vars: Optional[Union[Tuple, List, np.ndarray]] = ...,
    var_name: Optional[str] = ...,
    value_name: str = ...,
    col_level: Optional[Union[int, str]] = ...,
    ignore_index: bool = ...,
) -> DataFrame: ...
def lreshape(data: DataFrame, groups, dropna: bool = ..., label=...) -> DataFrame: ...
def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = ..., suffix: str = ...
) -> DataFrame: ...
