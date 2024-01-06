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
        com: float | None = None,
        span: float | None = None,
        halflife: TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = 0,
        times: str | np.ndarray | Series | np.timedelta64 | None = None,
        method: CalculationMethod = "single",
        *,
        selection: IndexLabel | None = None,
    ) -> None: ...
    def online(
        self,
        engine: WindowingEngine = "numba",
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> OnlineExponentialMovingWindow[NDFrameT]: ...
    def mean(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def sum(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def std(self, bias: bool = False, numeric_only: bool = False) -> NDFrameT: ...
    def var(self, bias: bool = False, numeric_only: bool = False) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
    ) -> NDFrameT: ...

class ExponentialMovingWindowGroupby(
    BaseWindowGroupby[NDFrameT], ExponentialMovingWindow[NDFrameT]
):
    def __init__(self, obj, *args, _grouper=None, **kwargs) -> None: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow[NDFrameT]):
    def __init__(
        self,
        obj: NDFrameT,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = 0,
        times: np.ndarray | NDFrameT | None = None,
        engine: WindowingEngine = "numba",
        engine_kwargs: WindowingEngineKwargs = None,
        *,
        selection: IndexLabel | None = None,
    ) -> None: ...
    def reset(self) -> None: ...
    def aggregate(self, func, *args, **kwargs): ...
    def std(self, bias: bool = False, *args, **kwargs): ...
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
    ): ...
    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ): ...
    def var(self, bias: bool = False, numeric_only: bool = False): ...
    def mean(
        self, *args, update: NDFrameT | None = None, update_times: None = None, **kwargs
    ) -> NDFrameT: ...
