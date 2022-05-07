from pandas.core.series import Series
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas._typing import Scalar
from typing import Callable, Optional, Sequence, Union

def pivot_table(
    data: DataFrame,
    values: Optional[str] = ...,
    index: Optional[Union[str, Sequence, Grouper]] = ...,
    columns: Optional[Union[str, Sequence, Grouper]] = ...,
    aggfunc=...,
    fill_value: Optional[Scalar] = ...,
    margins: bool = ...,
    dropna: bool = ...,
    margins_name: str = ...,
    observed: bool = ...,
) -> DataFrame: ...
def pivot(
    data: DataFrame,
    index: Optional[str] = ...,
    columns: Optional[str] = ...,
    values: Optional[Union[str, Sequence[str]]] = ...,
) -> DataFrame: ...
def crosstab(
    index: Union[Sequence, Series],
    columns: Union[Sequence, Series],
    values: Optional[Sequence] = ...,
    rownames: Optional[Sequence] = ...,
    colnames: Optional[Sequence] = ...,
    aggfunc: Optional[Callable] = ...,
    margins: bool = ...,
    margins_name: str = ...,
    dropna: bool = ...,
    normalize: bool = ...,
) -> DataFrame: ...
