from typing import Any

from numpy import typing as npt
from pandas.core.indexes.datetimes import DatetimeIndex

from pandas._libs.tslibs import BaseOffset

class BaseIndexer:
    def __init__(
        self,
        index_array: npt.NDArray[Any] | None = None,
        window_size: int = 0,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int,
        min_periods: int | None,
        center: bool | None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

class FixedForwardWindowIndexer(BaseIndexer): ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: npt.NDArray[Any] | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs: Any,
    ) -> None: ...
