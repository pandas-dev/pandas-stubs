import io
from typing import (
    Any,
    Dict,
)

import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check

from pandas.plotting import (
    deregister_matplotlib_converters,
    register_matplotlib_converters,
)


@pytest.fixture
def close_figures():
    plt.close("all")


IRIS = """SepalLength,SepalWidth,PetalLength,PetalWidth,Name
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3.0,5.8,2.2,Iris-virginica
7.6,3.0,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3.0,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6.0,2.2,5.0,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3.0,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2.0,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3.0,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6.0,3.0,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3.0,5.2,2.3,Iris-virginica
6.3,2.5,5.0,1.9,Iris-virginica
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica"""

IRIS_DF = pd.read_csv(io.StringIO(IRIS))


def test_andrews_curves(close_figures) -> None:
    check(assert_type(pd.plotting.andrews_curves(IRIS_DF, "Name"), plt.Axes), plt.Axes)


def test_autocorrelation_plot(close_figures) -> None:
    spacing = np.linspace(-9 * np.pi, 9 * np.pi, num=1000)
    s = pd.Series(0.7 * np.random.rand(1000) + 0.3 * np.sin(spacing))
    check(assert_type(pd.plotting.autocorrelation_plot(s), plt.Axes), plt.Axes)


def test_bootstrap_plot(close_figures) -> None:
    s = pd.Series(np.random.uniform(size=100))
    check(assert_type(pd.plotting.bootstrap_plot(s), plt.Figure), plt.Figure)


def test_boxplot(close_figures) -> None:
    np.random.seed(1234)
    df = pd.DataFrame(np.random.randn(10, 4), columns=["Col1", "Col2", "Col3", "Col4"])
    check(
        assert_type(pd.plotting.boxplot(df, column=["Col1", "Col2", "Col3"]), plt.Axes),
        plt.Axes,
    )


def test_reg_dereg(close_figures) -> None:
    check(assert_type(register_matplotlib_converters(), None), type(None))
    check(assert_type(deregister_matplotlib_converters(), None), type(None))


def test_lag_plot(close_figures) -> None:
    np.random.seed(5)
    x = np.cumsum(np.random.normal(loc=1, scale=5, size=50))
    s = pd.Series(x)
    check(assert_type(pd.plotting.lag_plot(s, lag=1), plt.Axes), plt.Axes)


def test_plot_parallel_coordinates(close_figures) -> None:
    check(
        assert_type(
            pd.plotting.parallel_coordinates(
                IRIS_DF, "Name", color=("#556270", "#4ECDC4", "#C7F464")
            ),
            plt.Axes,
        ),
        plt.Axes,
    )


def test_plot_params(close_figures) -> None:
    check(assert_type(pd.plotting.plot_params, Dict[str, Any]), dict)


def test_radviz(close_figures) -> None:
    df = pd.DataFrame(
        {
            "SepalLength": [6.5, 7.7, 5.1, 5.8, 7.6, 5.0, 5.4, 4.6, 6.7, 4.6],
            "SepalWidth": [3.0, 3.8, 3.8, 2.7, 3.0, 2.3, 3.0, 3.2, 3.3, 3.6],
            "PetalLength": [5.5, 6.7, 1.9, 5.1, 6.6, 3.3, 4.5, 1.4, 5.7, 1.0],
            "PetalWidth": [1.8, 2.2, 0.4, 1.9, 2.1, 1.0, 1.5, 0.2, 2.1, 0.2],
            "Category": [
                "virginica",
                "virginica",
                "setosa",
                "virginica",
                "virginica",
                "versicolor",
                "versicolor",
                "setosa",
                "virginica",
                "setosa",
            ],
        }
    )
    check(assert_type(pd.plotting.radviz(df, "Category"), plt.Axes), plt.Axes)


def test_scatter_matrix(close_figures) -> None:
    df = pd.DataFrame(np.random.randn(1000, 4), columns=["A", "B", "C", "D"])
    check(
        assert_type(
            pd.plotting.scatter_matrix(df, alpha=0.2),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_table(close_figures) -> None:
    df = pd.DataFrame(np.random.randn(1000, 4), columns=["A", "B", "C", "D"])
    _, ax = plt.subplots(1, 1)
    check(assert_type(pd.plotting.table(ax, df), Table), Table)


def test_plot_line():
    check(assert_type(IRIS_DF.plot(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot.line(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="line"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.line(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="line", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_area(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.area(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="area"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.area(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="area", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_bar(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.bar(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="bar"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.bar(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="bar", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_barh(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.barh(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="barh"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.barh(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="barh", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_box(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.box(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="box"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.box(subplots=True),
            pd.Series,
        ),
        pd.Series,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="box", subplots=True),
            pd.Series,
        ),
        pd.Series,
    )


def test_plot_density(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.density(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="density"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.density(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="density", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_hexbin(close_figures) -> None:
    check(
        assert_type(IRIS_DF.plot.hexbin(x="SepalLength", y="SepalWidth"), plt.Axes),
        plt.Axes,
    )
    check(
        assert_type(
            IRIS_DF.plot(x="SepalLength", y="SepalWidth", kind="hexbin"), plt.Axes
        ),
        plt.Axes,
    )
    check(
        assert_type(
            IRIS_DF.plot.hexbin(x="SepalLength", y="SepalWidth", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(kind="hexbin", x="SepalLength", y="SepalWidth", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_hist(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.hist(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="hist"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.hist(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(subplots=True, kind="hist"),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_kde(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.kde(), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="kde"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.kde(subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(subplots=True, kind="kde"),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_pie(close_figures) -> None:
    check(assert_type(IRIS_DF.plot.pie(y="SepalLength"), plt.Axes), plt.Axes)
    check(assert_type(IRIS_DF.plot(kind="pie", y="SepalLength"), plt.Axes), plt.Axes)
    check(
        assert_type(
            IRIS_DF.plot.pie(y="SepalLength", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )

    check(
        assert_type(
            IRIS_DF.plot(kind="pie", y="SepalLength", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_scatter(close_figures) -> None:
    check(
        assert_type(IRIS_DF.plot.scatter(x="SepalLength", y="SepalWidth"), plt.Axes),
        plt.Axes,
    )
    check(
        assert_type(
            IRIS_DF.plot(x="SepalLength", y="SepalWidth", kind="scatter"), plt.Axes
        ),
        plt.Axes,
    )
    check(
        assert_type(
            IRIS_DF.plot.scatter(x="SepalLength", y="SepalWidth", subplots=True),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            IRIS_DF.plot(
                x="SepalLength", y="SepalWidth", subplots=True, kind="scatter"
            ),
            npt.NDArray[np.object_],
        ),
        np.ndarray,
    )


def test_plot_keywords(close_figures):
    _, ax = plt.subplots(1, 1)
    df = IRIS_DF.iloc[:, :3].abs()
    check(
        assert_type(
            df.plot(
                x="SepalLength",
                y="SepalWidth",
                kind="line",
                ax=ax,
                subplots=False,
                sharex=False,
                sharey=False,
                layout=(8, 8),
                figsize=(16, 16),
                use_index=False,
                title="Some title",
                grid=True,
                legend="reverse",
                style="r:",
                logx=True,
                logy=True,
                loglog=True,
                xticks=[0, 3, 5, 10],
                yticks=[0, 0.2, 0.8],
                xlim=(1e-5, 10.0),
                ylim=(0.001, 1.0),
                xlabel="XX",
                ylabel="YY",
                rot=0.0,
                fontsize=12,
                colormap="jet",
                table=True,
                stacked=True,
                secondary_y=True,
                mark_right=True,
                include_bool=True,
                backend="matplotlib",
            ),
            plt.Axes,
        ),
        plt.Axes,
    )

    df = pd.DataFrame(np.random.rand(50, 4), columns=["a", "b", "c", "d"])
    df["species"] = pd.Categorical(
        ["setosa"] * 20 + ["versicolor"] * 20 + ["virginica"] * 10
    )
    check(
        assert_type(
            df.plot(
                kind="scatter",
                x="a",
                y="b",
                c="species",
                cmap="viridis",
                s=50,
                colorbar=False,
            ),
            plt.Axes,
        ),
        plt.Axes,
    )

    df = pd.DataFrame(np.random.rand(10, 5), columns=["A", "B", "C", "D", "E"])
    check(
        assert_type(
            df.plot(kind="box", vert=False, positions=[1, 4, 5, 6, 8]), plt.Axes
        ),
        plt.Axes,
    )


def test_plot_subplot_changes_150() -> None:
    df = pd.DataFrame(np.random.standard_normal((25, 4)), columns=["a", "b", "c", "d"])
    check(
        assert_type(
            df.plot(subplots=[("a", "b"), ("c", "d")]), npt.NDArray[np.object_]
        ),
        np.ndarray,
    )
