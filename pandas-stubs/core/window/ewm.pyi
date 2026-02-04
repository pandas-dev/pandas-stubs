from typing import (
    Any,
    overload,
)

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
    @overload  # type: ignore[override]
    def aggregate(  # pyrefly: ignore[bad-override]
        self: BaseWindow[Series],
        func: str,
        *args: Any,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def aggregate(  # ty: ignore[invalid-method-override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self: BaseWindow[DataFrame],
        func: str,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    agg = aggregate  # type: ignore[assignment]  # ty: ignore[invalid-method-override]  # pyrefly: ignore[bad-override]

class ExponentialMovingWindowGroupby(
    BaseWindowGroupby[NDFrameT], ExponentialMovingWindow[NDFrameT]
): ...
