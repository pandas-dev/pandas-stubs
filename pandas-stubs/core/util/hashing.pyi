from pandas import (
    DataFrame,
    Index,
    Series,
)

from pandas._typing import ArrayLike

def hash_pandas_object(
    obj: Index | Series | DataFrame,
    index: bool = ...,
    encoding: str = ...,
    hash_key: str | None = ...,
    categorize: bool = ...,
): ...
def hash_array(
    vals: ArrayLike, encoding: str = ..., hash_key: str = ..., categorize: bool = ...
): ...
