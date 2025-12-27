from typing import overload

from pandas._typing import (
    AnyArrayLike,
    ListLike,
    ScalarIndexer,
    np_1darray_bool,
)

@overload
def check_array_indexer(array: AnyArrayLike, indexer: ScalarIndexer) -> int: ...
@overload
def check_array_indexer(array: AnyArrayLike, indexer: slice) -> slice: ...
@overload
def check_array_indexer(array: AnyArrayLike, indexer: ListLike) -> np_1darray_bool: ...
