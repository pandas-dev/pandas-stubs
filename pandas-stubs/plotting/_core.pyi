from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Sequence,
)
from typing import (
    Any,
    Literal,
    NamedTuple,
    TypeAlias,
    overload,
    type_check_only,
)

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.stats import gaussian_kde

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    HashableT,
    HashableT1,
    HashableT2,
    HashableT3,
    ListLikeHashable,
    np_ndarray_float,
    np_ndarray_object,
)

@type_check_only
class BoxPlotT(NamedTuple):
    ax: Axes
    lines: dict[str, list[Line2D]]

_SingleColor: TypeAlias = (
    str | list[float] | tuple[float, float, float] | tuple[float, float, float, float]
)
_PlotAccessorColor: TypeAlias = str | list[_SingleColor] | dict[Any, _SingleColor]

# Keep in sync with `DataFrame.boxplot`
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | ListLikeHashable,
    by: None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: Literal["axes"] | None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> Axes: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | ListLikeHashable,
    by: None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: Literal["dict"],
    backend: str | None = None,
    **kwargs: Any,
) -> dict[str, Axes]: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | ListLikeHashable,
    by: None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: Literal["both"],
    backend: str | None = None,
    **kwargs: Any,
) -> BoxPlotT: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | ListLikeHashable,
    by: Hashable | ListLikeHashable,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: None = None,
    backend: str | None = None,
    **kwargs: Any,
) -> Axes: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | ListLikeHashable,
    by: Hashable | ListLikeHashable,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: Literal["axes", "dict", "both"],
    backend: str | None = None,
    **kwargs: Any,
) -> Series: ...

class PlotAccessor:
    def __init__(self, data: Series | DataFrame) -> None: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = None,
        x: Hashable = None,
        y: Hashable | Sequence[Hashable] = None,
        kind: Literal["pie"],
        ax: Axes | None = None,
        subplots: Literal[False] | None = False,
        sharex: bool | None = None,
        sharey: bool = False,
        layout: tuple[int, int] | None = None,
        figsize: tuple[float, float] | None = None,
        use_index: bool = True,
        title: Sequence[str] | None = None,
        grid: bool | None = None,
        legend: bool | Literal["reverse"] = False,
        style: str | list[str] | dict[HashableT1, str] | None = None,
        logx: bool | Literal["sym"] = False,
        logy: bool | Literal["sym"] = False,
        loglog: bool | Literal["sym"] = False,
        xticks: Sequence[float] | None = None,
        yticks: Sequence[float] | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        rot: float | None = None,
        fontsize: float | None = None,
        colormap: str | Colormap | None = None,
        colorbar: bool = False,
        position: float = 0.5,
        table: bool | Series | DataFrame = False,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        stacked: bool = False,
        secondary_y: bool | list[HashableT2] | tuple[HashableT2, ...] = False,
        mark_right: bool = True,
        include_bool: bool = False,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = None,
        x: Hashable = None,
        y: Hashable | Sequence[Hashable] = None,
        kind: Literal["hist"],
        ax: Axes | None = None,
        subplots: Literal[False] | None = False,
        sharex: bool | None = None,
        sharey: bool = False,
        layout: tuple[int, int] | None = None,
        figsize: tuple[float, float] | None = None,
        use_index: bool = True,
        title: Sequence[str] | None = None,
        grid: bool | None = None,
        legend: bool | Literal["reverse"] = False,
        style: str | list[str] | dict[HashableT1, str] | None = None,
        logx: bool | Literal["sym"] = False,
        logy: bool | Literal["sym"] = False,
        loglog: bool | Literal["sym"] = False,
        xticks: Sequence[float] | None = None,
        yticks: Sequence[float] | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        rot: float | None = None,
        fontsize: float | None = None,
        colormap: str | Colormap | None = None,
        colorbar: bool = False,
        position: float = 0.5,
        table: bool | Series | DataFrame = False,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        stacked: bool = False,
        secondary_y: bool | list[HashableT2] | tuple[HashableT2, ...] = False,
        mark_right: bool = True,
        include_bool: bool = False,
        backend: str | None = None,
        weights: AnyArrayLike | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = ...,
        x: Hashable = None,
        y: Hashable | Sequence[Hashable] = None,
        kind: Literal[
            "line",
            "bar",
            "barh",
            "hist",
            "box",
            "kde",
            "density",
            "area",
            "pie",
            "scatter",
            "hexbin",
        ] = "line",
        ax: Axes | None = None,
        subplots: Literal[False] | None = False,
        sharex: bool | None = None,
        sharey: bool = False,
        layout: tuple[int, int] | None = None,
        figsize: tuple[float, float] | None = None,
        use_index: bool = True,
        title: Sequence[str] | None = None,
        grid: bool | None = None,
        legend: bool | Literal["reverse"] = False,
        style: str | list[str] | dict[HashableT1, str] | None = None,
        logx: bool | Literal["sym"] = False,
        logy: bool | Literal["sym"] = False,
        loglog: bool | Literal["sym"] = False,
        xticks: Sequence[float] | None = None,
        yticks: Sequence[float] | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        rot: float | None = None,
        fontsize: float | None = None,
        colormap: str | Colormap | None = None,
        colorbar: bool = False,
        position: float = 0.5,
        table: bool | Series | DataFrame = False,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        stacked: bool = False,
        secondary_y: bool | list[HashableT2] | tuple[HashableT2, ...] = False,
        mark_right: bool = True,
        include_bool: bool = False,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = ...,
        x: Hashable = None,
        y: Hashable | Sequence[Hashable] = None,
        kind: Literal[
            "line",
            "bar",
            "barh",
            "hist",
            "kde",
            "density",
            "area",
            "pie",
            "scatter",
            "hexbin",
        ] = "line",
        ax: Axes | None = None,
        subplots: Literal[True] | Sequence[Iterable[HashableT1]],
        sharex: bool | None = None,
        sharey: bool = False,
        layout: tuple[int, int] | None = None,
        figsize: tuple[float, float] | None = None,
        use_index: bool = True,
        title: Sequence[str] | None = None,
        grid: bool | None = None,
        legend: bool | Literal["reverse"] = False,
        style: str | list[str] | dict[HashableT2, str] | None = None,
        logx: bool | Literal["sym"] = False,
        logy: bool | Literal["sym"] = False,
        loglog: bool | Literal["sym"] = False,
        xticks: Sequence[float] | None = None,
        yticks: Sequence[float] | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        rot: float | None = None,
        fontsize: float | None = None,
        colormap: str | Colormap | None = None,
        colorbar: bool = False,
        position: float = 0.5,
        table: bool | Series | DataFrame = False,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        stacked: bool = False,
        secondary_y: bool | list[HashableT3] | tuple[HashableT3, ...] = False,
        mark_right: bool = True,
        include_bool: bool = False,
        backend: str | None = None,
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = ...,
        x: Hashable = None,
        y: Hashable | Sequence[Hashable] = None,
        kind: Literal["box"],
        ax: Axes | None = None,
        subplots: Literal[True] | Sequence[Iterable[HashableT1]],
        sharex: bool | None = None,
        sharey: bool = False,
        layout: tuple[int, int] | None = None,
        figsize: tuple[float, float] | None = None,
        use_index: bool = True,
        title: Sequence[str] | None = None,
        grid: bool | None = None,
        legend: bool | Literal["reverse"] = False,
        style: str | list[str] | dict[HashableT2, str] | None = None,
        logx: bool | Literal["sym"] = False,
        logy: bool | Literal["sym"] = False,
        loglog: bool | Literal["sym"] = False,
        xticks: Sequence[float] | None = None,
        yticks: Sequence[float] | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        rot: float | None = None,
        fontsize: float | None = None,
        colormap: str | Colormap | None = None,
        colorbar: bool = False,
        position: float = 0.5,
        table: bool | Series | DataFrame = False,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str | None = None,
        stacked: bool = False,
        secondary_y: bool | list[HashableT3] | tuple[HashableT3, ...] = False,
        mark_right: bool = True,
        include_bool: bool = False,
        backend: str | None = None,
        **kwargs: Any,
    ) -> pd.Series: ...
    @overload
    def line(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor | None = None,
        *,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def line(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor | None = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def bar(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor | None = None,
        *,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def bar(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor | None = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def barh(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor | None = None,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def barh(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor | None = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def box(
        self,
        by: Hashable | list[HashableT] = None,
        *,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def box(
        self,
        by: Hashable | list[HashableT] = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def hist(
        self,
        by: Hashable | list[HashableT] | None = None,
        bins: int = 10,
        *,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def hist(
        self,
        by: Hashable | list[HashableT] | None = None,
        bins: int = 10,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def kde(
        self,
        bw_method: (
            Literal["scott", "silverman"]
            | float
            | Callable[[gaussian_kde], float]
            | None
        ) = None,
        weights: np_ndarray_float | Series[float] | None = None,
        ind: np_ndarray_float | int | None = None,
        *,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def kde(
        self,
        bw_method: (
            Literal["scott", "silverman"]
            | float
            | Callable[[gaussian_kde], float]
            | None
        ) = None,
        weights: np_ndarray_float | Series[float] | None = None,
        ind: np_ndarray_float | int | None = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        *,
        subplots: Literal[False] | None = False,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def pie(
        self, y: Hashable, *, subplots: Literal[False] | None = None, **kwargs: Any
    ) -> Axes: ...
    @overload
    def pie(
        self, y: Hashable, *, subplots: Literal[True], **kwargs: Any
    ) -> np_ndarray_object: ...
    @overload
    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[float] | None = None,
        c: Hashable | list[str] = None,
        *,
        subplots: Literal[False] | None = None,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[float] | None = None,
        c: Hashable | list[str] = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...
    @overload
    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable[[list[Any]], float] | None = None,
        gridsize: int | tuple[int, int] | None = None,
        *,
        subplots: Literal[False] | None = False,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable[[list[Any]], float] | None = None,
        gridsize: int | tuple[int, int] | None = None,
        *,
        subplots: Literal[True],
        **kwargs: Any,
    ) -> np_ndarray_object: ...

    density = kde
