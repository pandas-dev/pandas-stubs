import numpy as np
from pandas import (
    DataFrame,
    Series,
)
from pandas.core.window.rolling import (
    BaseWindow,
    BaseWindowGroupby,
)

from pandas._typing import (
    Axis,
    CalculationMethod,
    IndexLabel,
    NDFrameT,
    TimedeltaConvertibleTypes,
    WindowingEngine,
    WindowingEngineKwargs,
)

class ExponentialMovingWindow(BaseWindow[NDFrameT]):
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
        *,
        selection: IndexLabel | None = ...,
    ) -> None: ...
    def online(
        self,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> OnlineExponentialMovingWindow[NDFrameT]: ...
    def mean(
        self,
        numeric_only: bool = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
    ) -> NDFrameT: ...
    def sum(
        self,
        numeric_only: bool = ...,
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

class ExponentialMovingWindowGroupby(
    BaseWindowGroupby[NDFrameT], ExponentialMovingWindow[NDFrameT]
):
    def __init__(self, obj, *args, _grouper=..., **kwargs) -> None: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow[NDFrameT]):
    def __init__(
        self,
        obj: NDFrameT,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: np.ndarray | NDFrameT | None = ...,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        *,
        selection: IndexLabel | None = ...,
    ) -> None: ...
    def reset(self) -> None: ...
    def aggregate(self, func, *args, **kwargs): ...
    def std(self, bias: bool = ..., *args, **kwargs): ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = ...,
    ): ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        numeric_only: bool = ...,
    ): ...
    def var(self, bias: bool = ..., numeric_only: bool = ...): ...
    def mean(
        self, *args, update: NDFrameT | None = ..., update_times: None = ..., **kwargs
    ) -> NDFrameT: ...
