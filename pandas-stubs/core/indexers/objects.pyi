from typing import Any

import numpy as np
from pandas import DatetimeIndex

from pandas._libs.tslibs import BaseOffset

class BaseIndexer:
    def __init__(
        self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs: Any
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int,
        min_periods: int | None,
        center: bool | None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer): ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs: Any,
    ) -> None: ...
