from pandas import (
    DataFrame,
    Series,
)
from pandas.core.window.rolling import (
    BaseWindow,
    BaseWindowGroupby,
)

from pandas._typing import (
    NDFrameT,
    WindowingEngine,
    WindowingEngineKwargs,
)

class ExponentialMovingWindow(BaseWindow[NDFrameT]):
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
): ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow[NDFrameT]):
    def reset(self) -> None: ...
    def aggregate(self, func, *args, **kwargs): ...
    def std(self, bias: bool = False, *args, **kwargs): ...  # pyrefly: ignore
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
    def mean(  # pyrefly: ignore
        self, *args, update: NDFrameT | None = ..., update_times: None = ..., **kwargs
    ) -> NDFrameT: ...
