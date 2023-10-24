from typing import Generic

import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.window.rolling import BaseWindow

from pandas._typing import (
    Axis,
    CalculationMethod,
    NDFrameT,
    TimedeltaConvertibleTypes,
    WindowingEngine,
    WindowingEngineKwargs,
)

class ExponentialMovingWindow(BaseWindow[NDFrameT], Generic[NDFrameT]):
    def __init__(
        self,
        obj: NDFrameT,
        com: float | None = ...,
        span: float | None = ...,
        halflife: TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: str | np.ndarray | Series | None | np.timedelta64 = ...,
        method: CalculationMethod = ...,
    ) -> None: ...
    def mean(
        self,
        numeric_only: bool = ...,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def sum(
        self,
        numeric_only: bool = ...,
        *,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def std(self, bias: bool = ..., numeric_only: bool = ...) -> NDFrameT: ...
    def var(self, bias: bool = ..., numeric_only: bool = ...) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = ...,
    ) -> NDFrameT: ...
