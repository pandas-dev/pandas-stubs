from typing import overload

from pandas._typing import (
    AnyArrayLike,
    ListLike,
    np_ndarray_int,
)

@overload
def check_array_indexer(array: AnyArrayLike, indexer: int) -> int: ...
@overload
def check_array_indexer(array: AnyArrayLike, indexer: slice) -> slice: ...
@overload
def check_array_indexer(array: AnyArrayLike, indexer: ListLike) -> np_ndarray_int: ...
