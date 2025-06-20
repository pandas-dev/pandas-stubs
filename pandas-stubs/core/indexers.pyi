import numpy as np

def check_array_indexer(arrayArrayLike, indexer): ...

class BaseIndexer:
    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int = ...,
        **kwargs,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int = ...,
        index=...,
        offset=...,
        **kwargs,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...
