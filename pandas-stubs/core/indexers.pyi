from typing import Any

from numpy import typing as npt

from pandas._typing import (
    AnyArrayLike,
    np_1darray_bool,
    np_1darray_intp,
    np_ndarray_intp,
)

def check_array_indexer(
    arrayArrayLike: AnyArrayLike, indexer: AnyArrayLike
) -> np_1darray_bool: ...

class BaseIndexer:
    def __init__(
        self,
        index_array: np_ndarray_intp | None = ...,
        window_size: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np_1darray_intp, np_1darray_intp]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np_ndarray_intp | None = ...,
        window_size: int = ...,
        index=...,
        offset=...,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
