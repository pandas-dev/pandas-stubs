from typing import Any

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.window.rolling import BaseWindow

from pandas._typing import (
    Axis,
    TimedeltaConvertibleTypes,
)

class ExponentialMovingWindow(BaseWindow):
    com: Any = ...  # Incomplete
    span: Any = ...  # Incomplete
    halflife: Any = ...  # Incomplete
    alpha: Any = ...  # Incomplete
    adjust: Any = ...  # Incomplete
    ignore_na: Any = ...  # Incomplete
    times: Any = ...  # Incomplete
    def __init__(
        self,
        obj: NDFrame,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: str | np.ndarray | NDFrame | None = ...,
        method: str = ...,
        *,
        selection: Any | None = ...,
    ) -> None: ...
    def online(self, engine: str = ..., engine_kwargs: Any | None = ...): ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = aggregate
    def mean(
        self,
        *args,
        engine: Any | None = ...,
        engine_kwargs: Any | None = ...,
        **kwargs,
    ) -> Series | DataFrame: ...
    def sum(
        self,
        *args,
        engine: Any | None = ...,
        engine_kwargs: Any | None = ...,
        **kwargs,
    ) -> Series | DataFrame: ...
    def std(self, bias: bool = ..., *args, **kwargs) -> Series | DataFrame: ...
    def vol(self, bias: bool = ..., *args, **kwargs) -> Series | DataFrame: ...
    def var(self, bias: bool = ..., *args, **kwargs) -> Series | DataFrame: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        **kwargs,
    ) -> Series | DataFrame: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        **kwargs,
    ) -> Series | DataFrame: ...
