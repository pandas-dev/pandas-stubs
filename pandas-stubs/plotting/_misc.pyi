from typing import (
    Any,
    Sequence,
)

from matplotlib.axes import Axes as PlotAxes
from matplotlib.figure import Figure
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series

def table(
    ax,
    data,
    rowLabels=...,
    colLabels=...,
): ...
def register() -> None: ...
def deregister() -> None: ...
def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: tuple[float, float] | None = ...,
    ax: PlotAxes | None = ...,
    grid: bool = ...,
    diagonal: str = ...,
    marker: str = ...,
    density_kwds=...,
    hist_kwds=...,
    range_padding: float = ...,
) -> np.ndarray: ...
def radviz(
    frame: DataFrame,
    class_column: str,
    ax: PlotAxes | None = ...,
    color: list[str] | tuple[str] | None = ...,
    colormap=...,
) -> PlotAxes: ...
def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: PlotAxes | None = ...,
    samples: int = ...,
    color: list[str] | tuple[str] | None = ...,
    colormap=...,
) -> PlotAxes: ...
def bootstrap_plot(
    series: Series,
    fig: Figure | None = ...,
    size: int = ...,
    samples: int = ...,
) -> Figure: ...
def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[str] | None = ...,
    ax: PlotAxes | None = ...,
    color: list[str] | tuple[str] | None = ...,
    use_columns: bool = ...,
    xticks: Sequence | tuple | None = ...,
    colormap=...,
    axvlines: bool = ...,
    axvlines_kwds=...,
    sort_labels: bool = ...,
) -> PlotAxes: ...
def lag_plot(
    series: Series,
    lag: int = ...,
    ax: PlotAxes | None = ...,
) -> PlotAxes: ...
def autocorrelation_plot(
    series: Series,
    ax: PlotAxes | None = ...,
) -> PlotAxes: ...

class _Options(dict):
    def __init__(self, deprecated: bool = ...) -> None: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def __delitem__(self, key): ...
    def __contains__(self, key) -> bool: ...
    def reset(self) -> None: ...
    def use(self, key, value) -> None: ...

plot_params: dict[str, Any]
