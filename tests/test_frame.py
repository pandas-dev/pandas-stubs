from __future__ import annotations

import datetime
import io
from pathlib import Path
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Tuple,
)

import numpy as np
import pandas as pd
from pandas._testing import getSeriesData
import pytest
from typing_extensions import assert_type

from pandas._typing import Scalar

from tests import check

from pandas.io.parsers import TextFileReader


def test_types_init() -> None:
    pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}, index=[2, 1])
    pd.DataFrame(data=[1, 2, 3, 4], dtype=np.int8)
    pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        columns=["a", "b", "c"],
        dtype=np.int8,
        copy=True,
    )


def test_types_append() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})
    with pytest.warns(FutureWarning, match="The frame.append"):
        res1: pd.DataFrame = df.append(df2)
    with pytest.warns(FutureWarning, match="The frame.append"):
        res2: pd.DataFrame = df.append([1, 2, 3])
    with pytest.warns(FutureWarning, match="The frame.append"):
        res3: pd.DataFrame = df.append([[1, 2, 3]])
    with pytest.warns(FutureWarning, match="The frame.append"):
        res4: pd.DataFrame = df.append(
            {("a", 1): [1, 2, 3], "b": df2}, ignore_index=True
        )
    with pytest.warns(FutureWarning, match="The frame.append"):
        res5: pd.DataFrame = df.append({1: [1, 2, 3]}, ignore_index=True)
    with pytest.warns(FutureWarning, match="The frame.append"):
        res6: pd.DataFrame = df.append(
            {1: [1, 2, 3], "col2": [1, 2, 3]}, ignore_index=True
        )
    with pytest.warns(FutureWarning, match="The frame.append"):
        res7: pd.DataFrame = df.append(pd.Series([5, 6]), ignore_index=True)
    with pytest.warns(FutureWarning, match="The frame.append"):
        res8: pd.DataFrame = df.append(
            pd.Series([5, 6], index=["col1", "col2"]), ignore_index=True
        )


def test_types_to_csv() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    csv_df: str = df.to_csv()

    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_csv(file.name)
        file.close()
        df2: pd.DataFrame = pd.read_csv(file.name)

    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_csv(Path(file.name))
        file.close()
        df3: pd.DataFrame = pd.read_csv(Path(file.name))

    # This keyword was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_csv(file.name, errors="replace")
        file.close()
        df4: pd.DataFrame = pd.read_csv(file.name)

    # Testing support for binary file handles, added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.to_csv(io.BytesIO(), encoding="utf-8", compression="gzip")


def test_types_to_csv_when_path_passed() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    path: Path = Path("./dummy_path.txt")
    try:
        assert not path.exists()
        df.to_csv(path)
        df5: pd.DataFrame = pd.read_csv(path)
    finally:
        path.unlink()


def test_types_copy() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df2: pd.DataFrame = df.copy()


def test_types_getitem() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    i = pd.Index(["col1", "col2"])
    s = pd.Series(["col1", "col2"])
    select_df = pd.DataFrame({"col1": [True, True], "col2": [False, True]})
    a = np.array(["col1", "col2"])
    df["col1"]
    df[5]
    df[["col1", "col2"]]
    df[1:]
    df[s]
    df[a]
    df[select_df]
    df[i]


def test_slice_setitem() -> None:
    # Due to the bug in pandas 1.2.3(https://github.com/pandas-dev/pandas/issues/40440), this is in separate test case
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    df[1:] = ["a", "b", "c"]


def test_types_setitem() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    i = pd.Index(["col1", "col2"])
    s = pd.Series(["col1", "col2"])
    a = np.array(["col1", "col2"])
    df["col1"] = [1, 2]
    df[5] = [5, 6]
    df[["col1", "col2"]] = [[1, 2], [3, 4]]
    df[s] = [5, 6]
    df[a] = [[1, 2], [3, 4]]
    df[i] = [8, 9]


def test_types_setitem_mask() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    select_df = pd.DataFrame({"col1": [True, True], "col2": [False, True]})
    df[select_df] = [1, 2, 3]


def test_types_iloc_iat() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.iloc[1, 1]
    df.iloc[[1], [1]]
    df.iat[0, 0]


def test_types_loc_at() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.loc[[0], "col1"]
    df.at[0, "col1"]
    df.loc[0, "col1"]


def test_types_boolean_indexing() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df[df > 1]
    df[~(df > 1.0)]


def test_types_df_to_df_comparison() -> None:
    df = pd.DataFrame(data={"col1": [1, 2]})
    df2 = pd.DataFrame(data={"col1": [3, 2]})
    res_gt: pd.DataFrame = df > df2
    res_ge: pd.DataFrame = df >= df2
    res_lt: pd.DataFrame = df < df2
    res_le: pd.DataFrame = df <= df2
    res_e: pd.DataFrame = df == df2


def test_types_head_tail() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.head(1)
    df.tail(1)


def test_types_assign() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.assign(col3=lambda frame: frame.sum(axis=1))
    df["col3"] = df.sum(axis=1)


def test_types_sample() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    # GH 67
    check(assert_type(df.sample(frac=0.5), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.sample(n=1), pd.DataFrame), pd.DataFrame)


def test_types_nlargest_nsmallest() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.nlargest(1, "col1")
    df.nsmallest(1, "col2")


def test_types_filter() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.filter(items=["col1"])
    df.filter(regex="co.*")
    df.filter(like="1")


def test_types_setting() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df["col1"] = 1
    df[df == 1] = 7


def test_types_drop() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    res: pd.DataFrame = df.drop("col1", axis=1)
    res2: pd.DataFrame = df.drop(columns=["col1"])
    res3: pd.DataFrame = df.drop([0])
    res4: pd.DataFrame = df.drop(index=[0])
    res5: pd.DataFrame = df.drop(columns=["col1"])
    res6: pd.DataFrame = df.drop(index=1)
    res7: pd.DataFrame = df.drop(labels=0)
    res8: None = df.drop([0, 0], inplace=True)
    to_drop: list[str] = ["col1"]
    res9: pd.DataFrame = df.drop(columns=to_drop)


def test_types_dropna() -> None:
    df = pd.DataFrame(data={"col1": [np.nan, np.nan], "col2": [3, np.nan]})
    res: pd.DataFrame = df.dropna()
    res2: pd.DataFrame = df.dropna(axis=1, thresh=1)
    res3: None = df.dropna(axis=0, how="all", subset=["col1"], inplace=True)


def test_types_fillna() -> None:
    df = pd.DataFrame(data={"col1": [np.nan, np.nan], "col2": [3, np.nan]})
    res: pd.DataFrame = df.fillna(0)
    res2: None = df.fillna(method="pad", axis=1, inplace=True)


def test_types_sort_index() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=[5, 1, 3, 2])
    df2 = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
    res: pd.DataFrame = df.sort_index()
    level1 = (1, 2)
    res2: pd.DataFrame = df.sort_index(ascending=False, level=level1)
    level2: list[str] = ["a", "b", "c"]
    res3: pd.DataFrame = df2.sort_index(level=level2)
    res4: pd.DataFrame = df.sort_index(ascending=False, level=3)
    res5: None = df.sort_index(kind="mergesort", inplace=True)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_index_with_key() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=["a", "b", "C", "d"])
    res: pd.DataFrame = df.sort_index(key=lambda k: k.str.lower())


def test_types_set_index() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}, index=[5, 1, 3, 2]
    )
    res: pd.DataFrame = df.set_index("col1")
    res2: pd.DataFrame = df.set_index("col1", drop=False)
    res3: pd.DataFrame = df.set_index("col1", append=True)
    res4: pd.DataFrame = df.set_index("col1", verify_integrity=True)
    res5: pd.DataFrame = df.set_index(["col1", "col2"])
    res6: None = df.set_index("col1", inplace=True)
    # GH 140
    res7: pd.DataFrame = df.set_index(pd.Index(["w", "x", "y", "z"]))


def test_types_query() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4], "col2": [3, 0, 1, 7]})
    res: pd.DataFrame = df.query("col1 > col2")
    res2: None = df.query("col1 % col2 == 0", inplace=True)


def test_types_eval() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4], "col2": [3, 0, 1, 7]})
    df.eval("col1 > col2")
    res: None = df.eval("C = col1 % col2 == 0", inplace=True)


def test_types_sort_values() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    res: pd.DataFrame = df.sort_values("col1")
    res2: None = df.sort_values("col1", ascending=False, inplace=True)
    res3: pd.DataFrame = df.sort_values(by=["col1", "col2"], ascending=[True, False])


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_values_with_key() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    res: pd.DataFrame = df.sort_values(by="col1", key=lambda k: -k)


def test_types_shift() -> None:
    df = pd.DataFrame(data={"col1": [1, 1], "col2": [3, 4]})
    df.shift()
    df.shift(1)
    df.shift(-1)


def test_types_rank() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.rank(axis=0, na_option="bottom")
    df.rank(method="min", pct=True)
    df.rank(method="dense", ascending=True)
    df.rank(method="first", numeric_only=True)


def test_types_mean() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    s1: pd.Series = df.mean()
    s2: pd.Series = df.mean(axis=0)
    with pytest.warns(FutureWarning, match="Using the level"):
        df2: pd.DataFrame = df.mean(level=0)
    with pytest.warns(FutureWarning, match="Using the level"):
        df3: pd.DataFrame = df.mean(axis=1, level=0)
    with pytest.warns(FutureWarning, match="Using the level"):
        df4: pd.DataFrame = df.mean(1, True, level=0)
    s3: pd.Series = df.mean(axis=1, skipna=True, numeric_only=False)


def test_types_median() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    s1: pd.Series = df.median()
    s2: pd.Series = df.median(axis=0)
    with pytest.warns(FutureWarning, match="Using the level keyword"):
        df2: pd.DataFrame = df.median(level=0)
    with pytest.warns(FutureWarning, match="Using the level keyword"):
        df3: pd.DataFrame = df.median(axis=1, level=0)
    with pytest.warns(FutureWarning, match="Using the level keyword"):
        df4: pd.DataFrame = df.median(1, True, level=0)
    s3: pd.Series = df.median(axis=1, skipna=True, numeric_only=False)


def test_types_itertuples() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    res1: Iterable[tuple[Any, ...]] = df.itertuples()
    res2: Iterable[tuple[Any, ...]] = df.itertuples(index=False, name="Foobar")
    res3: Iterable[tuple[Any, ...]] = df.itertuples(index=False, name=None)


def test_types_sum() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.sum()
    df.sum(axis=1)


def test_types_cumsum() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.cumsum()
    df.sum(axis=0)


def test_types_min() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.min()
    df.min(axis=0)


def test_types_max() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.max()
    df.max(axis=0)


def test_types_quantile() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.quantile([0.25, 0.5])
    df.quantile(0.75)
    df.quantile()
    # GH 81
    df.quantile(np.array([0.25, 0.75]))


def test_types_clip() -> None:
    df = pd.DataFrame(data={"col1": [20, 12], "col2": [3, 14]})
    df.clip(lower=5, upper=15)


def test_types_abs() -> None:
    df = pd.DataFrame(data={"col1": [-5, 1], "col2": [3, -14]})
    df.abs()


def test_types_var() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [1, 4]})
    df.var()
    df.var(axis=1, ddof=1)
    df.var(skipna=True, numeric_only=False)


def test_types_std() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [1, 4]})
    df.std()
    df.std(axis=1, ddof=1)
    df.std(skipna=True, numeric_only=False)


def test_types_idxmin() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.idxmin()
    df.idxmin(axis=0)


def test_types_idxmax() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.idxmax()
    df.idxmax(axis=0)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_value_counts() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [1, 4]})
    df.value_counts()


def test_types_unique() -> None:
    # This is really more for of a Series test
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [1, 4]})
    df["col1"].unique()


def test_types_apply() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.apply(lambda x: x**2)
    df.apply(np.exp)
    df.apply(str)


def test_types_applymap() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.applymap(lambda x: x**2)
    df.applymap(np.exp)
    df.applymap(str)
    # na_action parameter was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.applymap(np.exp, na_action="ignore")
    df.applymap(str, na_action=None)


def test_types_element_wise_arithmetic() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"col1": [10, 20], "col3": [3, 4]})

    res_add1: pd.DataFrame = df + df2
    res_add2: pd.DataFrame = df.add(df2, fill_value=0)

    res_sub: pd.DataFrame = df - df2
    res_sub2: pd.DataFrame = df.sub(df2, fill_value=0)

    res_mul: pd.DataFrame = df * df2
    res_mul2: pd.DataFrame = df.mul(df2, fill_value=0)

    res_div: pd.DataFrame = df / df2
    res_div2: pd.DataFrame = df.div(df2, fill_value=0)

    res_floordiv: pd.DataFrame = df // df2
    res_floordiv2: pd.DataFrame = df.floordiv(df2, fill_value=0)

    res_mod: pd.DataFrame = df % df2
    res_mod2: pd.DataFrame = df.mod(df2, fill_value=0)

    res_pow: pd.DataFrame = df2**df
    res_pow2: pd.DataFrame = df2.pow(df, fill_value=0)

    # divmod operation was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    # noinspection PyTypeChecker
    res_divmod: tuple[pd.DataFrame, pd.DataFrame] = divmod(df, df2)
    res_divmod2: tuple[pd.DataFrame, pd.DataFrame] = df.__divmod__(df2)
    res_rdivmod: tuple[pd.DataFrame, pd.DataFrame] = df.__rdivmod__(df2)


def test_types_scalar_arithmetic() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})

    res_add1: pd.DataFrame = df + 1
    res_add2: pd.DataFrame = df.add(1, fill_value=0)

    res_sub: pd.DataFrame = df - 1
    res_sub2: pd.DataFrame = df.sub(1, fill_value=0)

    res_mul: pd.DataFrame = df * 2
    res_mul2: pd.DataFrame = df.mul(2, fill_value=0)

    res_div: pd.DataFrame = df / 2
    res_div2: pd.DataFrame = df.div(2, fill_value=0)

    res_floordiv: pd.DataFrame = df // 2
    res_floordiv2: pd.DataFrame = df.floordiv(2, fill_value=0)

    res_mod: pd.DataFrame = df % 2
    res_mod2: pd.DataFrame = df.mod(2, fill_value=0)

    res_pow: pd.DataFrame = df**2
    res_pow1: pd.DataFrame = df**0
    res_pow2: pd.DataFrame = df**0.213
    res_pow3: pd.DataFrame = df.pow(0.5)


def test_types_melt() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.melt()
    df.melt(id_vars=["col1"], value_vars=["col2"])
    df.melt(
        id_vars=["col1"],
        value_vars=["col2"],
        var_name="someVariable",
        value_name="someValue",
    )

    pd.melt(df)
    pd.melt(df, id_vars=["col1"], value_vars=["col2"])
    pd.melt(
        df,
        id_vars=["col1"],
        value_vars=["col2"],
        var_name="someVariable",
        value_name="someValue",
    )


def test_types_pivot() -> None:
    df = pd.DataFrame(
        data={
            "col1": ["first", "second", "third", "fourth"],
            "col2": [50, 70, 56, 111],
            "col3": ["A", "B", "B", "A"],
            "col4": [100, 102, 500, 600],
        }
    )
    df.pivot(index="col1", columns="col3", values="col2")
    df.pivot(index="col1", columns="col3")
    df.pivot(index="col1", columns="col3", values=["col2", "col4"])


def test_types_groupby() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]})
    df.index.name = "ind"
    df.groupby(by="col1")
    df.groupby(level="ind")
    df.groupby(by="col1", sort=False, as_index=True)
    df.groupby(by=["col1", "col2"])

    df1: pd.DataFrame = df.groupby(by="col1").agg("sum")
    df2: pd.DataFrame = df.groupby(level="ind").aggregate("sum")
    df3: pd.DataFrame = df.groupby(by="col1", sort=False, as_index=True).transform(
        lambda x: x.max()
    )
    df4: pd.DataFrame = df.groupby(by=["col1", "col2"]).count()
    df5: pd.DataFrame = df.groupby(by=["col1", "col2"]).filter(lambda x: x["col1"] > 0)
    df6: pd.DataFrame = df.groupby(by=["col1", "col2"]).nunique()
    df7: pd.DataFrame = df.groupby(by="col1").apply(sum)
    df8: pd.DataFrame = df.groupby("col1").transform("sum")
    s1: pd.Series = df.set_index("col1")["col2"]
    s2: pd.Series = s1.groupby("col1").transform("sum")
    s3: pd.Series = df.groupby("col1")["col3"].agg(min)
    df9: pd.DataFrame = df.groupby("col1")["col3"].agg([min, max])
    df10: pd.DataFrame = df.groupby("col1").agg(
        new_col=pd.NamedAgg(column="col2", aggfunc="max")
    )


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_group_by_with_dropna_keyword() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]}
    )
    df.groupby(by="col2", dropna=True).sum()
    df.groupby(by="col2", dropna=False).sum()
    df.groupby(by="col2").sum()


def test_types_groupby_any() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, 1, 2],
            "col2": [True, False, False],
            "col3": [False, False, False],
        }
    )
    check(assert_type(df.groupby("col1").any(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").all(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1")["col2"].any(), "pd.Series[bool]"),
        pd.Series,
        bool,
    )
    check(
        assert_type(df.groupby("col1")["col2"].any(), "pd.Series[bool]"),
        pd.Series,
        bool,
    )


def test_types_merge() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df2 = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [0, 1, 0]})
    df.merge(df2)
    df.merge(df2, on="col1")
    df.merge(df2, on="col1", how="left")
    df.merge(df2, on=["col1", "col2"], how="left")
    df.merge(df2, on=("col1", "col2"), how="left")
    df.merge(df2, on=("col1", "col2"), how="left", suffixes=(None, "s"))
    df.merge(df2, on=("col1", "col2"), how="left", suffixes=("t", "s"))
    df.merge(df2, on=("col1", "col2"), how="left", suffixes=("a", None))
    columns = ["col1", "col2"]
    df.merge(df2, on=columns)


def test_types_plot() -> None:
    if TYPE_CHECKING:  # skip pytest
        df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
        df.plot.hist()
        df.plot.scatter(x="col2", y="col1")


def test_types_window() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.expanding()
    with pytest.warns(FutureWarning, match="The `center` argument on"):
        df.expanding(axis=1, center=True)

    df.rolling(2)
    df.rolling(2, axis=1, center=True)


def test_types_cov() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.cov()
    df.cov(min_periods=1)
    # ddof param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.cov(ddof=2)


def test_types_to_numpy() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.to_numpy()
    df.to_numpy(dtype="str", copy=True)
    # na_value param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_numpy(na_value=0)


def test_to_markdown() -> None:
    pytest.importorskip("tabulate")
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.to_markdown()
    df.to_markdown(buf=None, mode="wt")
    # index param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_markdown(index=False)


def test_types_to_feather() -> None:
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.to_feather("dummy_path")
    # kwargs for pyarrow.feather.write_feather added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_feather("dummy_path", compression="zstd", compression_level=3, chunksize=2)

    # to_feather has been able to accept a buffer since pandas 1.0.0
    # See https://pandas.pydata.org/docs/whatsnew/v1.0.0.html
    # Docstring and type were updated in 1.2.0.
    # https://github.com/pandas-dev/pandas/pull/35408
    with tempfile.NamedTemporaryFile(delete=False) as f:
        df.to_feather(f.name)
        f.close()


# compare() method added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_compare() -> None:
    df1 = pd.DataFrame(
        data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]}
    )
    df2 = pd.DataFrame(
        data={"col1": [1, 2, 5, 6], "col2": [3, 4, 1, 1], "col3": [3, 4, 3, 2]}
    )
    df1.compare(df2)
    df2.compare(df1, align_axis=0, keep_shape=True, keep_equal=True)


def test_types_agg() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    df.agg("min")
    df.agg(x=("A", max), y=("B", "min"), z=("C", np.mean))
    df.agg("mean", axis=1)


def test_types_describe() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, 2, -4],
            "col2": [
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.datetime64("2010-01-01"),
            ],
        }
    )
    df.describe()
    with pytest.warns(FutureWarning, match="Treating datetime data as categorical"):
        df.describe(percentiles=[0.5], include="all")
    with pytest.warns(FutureWarning, match="Treating datetime data as categorical"):
        df.describe(exclude=[np.number])
    # datetime_is_numeric param added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.describe(datetime_is_numeric=True)


def test_types_to_string() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, None, -4],
            "col2": [
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.datetime64("2010-01-01"),
            ],
        }
    )
    df.to_string(
        index=True,
        col_space=2,
        header=["a", "b"],
        na_rep="0",
        justify="left",
        max_rows=2,
        min_rows=0,
        max_cols=2,
        show_dimensions=True,
        line_width=3,
    )
    # col_space accepting list or dict added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_string(col_space=[1, 2])
    df.to_string(col_space={"col1": 1, "col2": 3})


def test_types_to_html() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, None, -4],
            "col2": [
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.datetime64("2010-01-01"),
            ],
        }
    )
    df.to_html(
        index=True,
        col_space=2,
        header=True,
        na_rep="0",
        justify="left",
        max_rows=2,
        max_cols=2,
        show_dimensions=True,
    )
    # col_space accepting list or dict added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_html(col_space=[1, 2])
    df.to_html(col_space={"col1": 1, "col2": 3})


def test_types_resample() -> None:
    df = pd.DataFrame({"values": [2, 11, 3, 13, 14, 18, 17, 19]})
    df["date"] = pd.date_range("01/01/2018", periods=8, freq="W")
    df.resample("M", on="date")
    # origin and offset params added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.resample("20min", origin="epoch", offset=pd.Timedelta(2, "minutes"), on="date")


def test_types_to_dict() -> None:
    data = pd.DataFrame({"a": [1], "b": [2]})
    data.to_dict(orient="records")
    data.to_dict(orient="dict")
    data.to_dict(orient="list")
    data.to_dict(orient="series")
    data.to_dict(orient="split")
    data.to_dict(orient="index")

    # orient param accepting "tight" added in 1.4.0 https://pandas.pydata.org/docs/whatsnew/v1.4.0.html
    data.to_dict(orient="tight")


def test_types_from_dict() -> None:
    pd.DataFrame.from_dict({"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]})
    pd.DataFrame.from_dict({1: [3, 2, 1, 0], 2: ["a", "b", "c", "d"]})
    pd.DataFrame.from_dict({"a": {1: 2}, "b": {3: 4, 1: 4}}, orient="index")
    pd.DataFrame.from_dict({"a": {"row1": 2}, "b": {"row2": 4, "row1": 4}})
    pd.DataFrame.from_dict({"a": (1, 2, 3), "b": (2, 4, 5)})
    pd.DataFrame.from_dict(
        data={"col_1": {"a": 1}, "col_2": {"a": 1, "b": 2}}, orient="columns"
    )
    # orient param accepting "tight" added in 1.4.0 https://pandas.pydata.org/docs/whatsnew/v1.4.0.html
    pd.DataFrame.from_dict(
        data={
            "index": [("a", "b"), ("a", "c")],
            "columns": [("x", 1), ("y", 2)],
            "data": [[1, 3], [2, 4]],
            "index_names": ["n1", "n2"],
            "column_names": ["z1", "z2"],
        },
        orient="tight",
    )


def test_pipe() -> None:
    if TYPE_CHECKING:  # skip pytest

        def foo(df: pd.DataFrame) -> pd.DataFrame:
            return df

        df1: pd.DataFrame = pd.DataFrame({"a": [1]}).pipe(foo)

        df2: pd.DataFrame = (
            pd.DataFrame(
                {
                    "price": [10, 11, 9, 13, 14, 18, 17, 19],
                    "volume": [50, 60, 40, 100, 50, 100, 40, 50],
                }
            )
            .assign(week_starting=pd.date_range("01/01/2018", periods=8, freq="W"))
            .resample("M", on="week_starting")
            .pipe(foo)
        )

        df3: pd.DataFrame = pd.DataFrame({"a": [1], "b": [1]}).groupby("a").pipe(foo)

        df4: pd.DataFrame = pd.DataFrame({"a": [1], "b": [1]}).style.pipe(foo)


# set_flags() method added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
def test_types_set_flags() -> None:
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"]).set_flags(
        allows_duplicate_labels=False
    )
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "A"]).set_flags(
        allows_duplicate_labels=True
    )
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "A"])


def test_types_to_parquet() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("fastparquet")
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"]).set_flags(
        allows_duplicate_labels=False
    )
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_parquet(Path(file.name))
        file.close()
    # to_parquet() returns bytes when no path given since 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    b: bytes = df.to_parquet()


def test_types_to_latex() -> None:
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    with pytest.warns(FutureWarning, match="In future versions `DataFrame.to_latex`"):
        df.to_latex(
            columns=["A"], label="some_label", caption="some_caption", multirow=True
        )
    with pytest.warns(FutureWarning, match="In future versions `DataFrame.to_latex`"):
        df.to_latex(escape=False, decimal=",", column_format="r")
    # position param was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    with pytest.warns(FutureWarning, match="In future versions `DataFrame.to_latex`"):
        df.to_latex(position="some")
    # caption param was extended to accept tuple in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    with pytest.warns(FutureWarning, match="In future versions `DataFrame.to_latex`"):
        df.to_latex(caption=("cap1", "cap2"))


def test_types_explode() -> None:
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    res1: pd.DataFrame = df.explode("A")
    res2: pd.DataFrame = df.explode("A", ignore_index=False)
    res3: pd.DataFrame = df.explode("A", ignore_index=True)


def test_types_rename() -> None:
    df = pd.DataFrame(columns=["a"])
    col_map = {"a": "b"}
    df.rename(columns=col_map)
    df.rename(columns={"a": "b"})
    df.rename(columns={1: "b"})
    # Apparently all of these calls are accepted by pandas
    df.rename(columns={None: "b"})
    df.rename(columns={"": "b"})
    df.rename(columns={(2, 1): "b"})
    df.rename(columns=lambda s: s.upper())


def test_types_eq() -> None:
    df1 = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    res1: pd.DataFrame = df1 == 1
    df2 = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    res2: pd.DataFrame = df1 == df2


def test_types_as_type() -> None:
    df1 = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    df2: pd.DataFrame = df1.astype({"A": "int32"})


def test_types_dot() -> None:
    df1 = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
    df2 = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
    s1 = pd.Series([1, 1, 2, 1])
    np_array = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
    df3: pd.DataFrame = df1 @ df2
    df4: pd.DataFrame = df1.dot(df2)
    df5: pd.DataFrame = df1 @ np_array
    df6: pd.DataFrame = df1.dot(np_array)
    df7: pd.Series = df1 @ s1
    df8: pd.Series = df1.dot(s1)


def test_types_regressions() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/32
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5, 6]})
    df2: pd.DataFrame = df.astype(int)

    # https://github.com/microsoft/python-type-stubs/issues/38
    df0: pd.DataFrame = pd.DataFrame({"x": [12, 34], "y": [78, 9]})
    ds: pd.DataFrame = df.sort_values(["x", "y"], ascending=[True, False])

    # https://github.com/microsoft/python-type-stubs/issues/55
    df3 = pd.DataFrame([["a", 1], ["b", 2]], columns=["let", "num"]).set_index("let")
    df4: pd.DataFrame = df3.reset_index()
    df5: pd.DataFrame = df4[["num"]]

    # https://github.com/microsoft/python-type-stubs/issues/58
    df1 = pd.DataFrame(columns=["a", "b", "c"])
    df2 = pd.DataFrame(columns=["a", "c"])
    df6: pd.DataFrame = df1.drop(columns=df2.columns)

    # https://github.com/microsoft/python-type-stubs/issues/60
    df1 = pd.DataFrame([["a", 1], ["b", 2]], columns=["let", "num"]).set_index("let")
    s2 = df1["num"]
    res: pd.DataFrame = pd.merge(s2, df1, left_index=True, right_index=True)

    # https://github.com/microsoft/python-type-stubs/issues/62
    df7: pd.DataFrame = pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))
    index: pd.Index = pd.Index(["b"])
    df8: pd.DataFrame = df7.loc[index]

    # https://github.com/microsoft/python-type-stubs/issues/31
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
    column1: pd.DataFrame = df.iloc[:, [0]]
    column2: pd.Series = df.iloc[:, 0]

    df = pd.DataFrame(
        {
            "a_col": list(range(10)),
            "a_nother": list(range(10)),
            "b_col": list(range(10)),
        }
    )
    df.loc[:, lambda df: df.columns.str.startswith("a_")]

    df = df[::-1]

    # https://github.com/microsoft/python-type-stubs/issues/69
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([4, 5, 6])
    df = pd.concat([s1, s2], axis=1)
    ss1: pd.Series = pd.concat([s1, s2], axis=0)
    ss2: pd.Series = pd.concat([s1, s2])

    # https://github.com/microsoft/python-type-stubs/issues/110
    d: datetime.date = pd.Timestamp("2021-01-01")
    tslist: list[pd.Timestamp] = list(pd.to_datetime(["2022-01-01", "2022-01-02"]))
    sseries: pd.Series = pd.Series(tslist)
    sseries_plus1: pd.Series = sseries + pd.Timedelta(1, "d")

    # https://github.com/microsoft/pylance-release/issues/2133
    dr = pd.date_range(start="2021-12-01", periods=24, freq="H")
    time = dr.strftime("%H:%M:%S")

    # https://github.com/microsoft/python-type-stubs/issues/115
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
    pd.DatetimeIndex(
        data=df["A"], tz=None, normalize=False, closed=None, ambiguous="NaT", copy=True
    )


def test_read_csv() -> None:
    if TYPE_CHECKING:  # skip pytest
        #  https://github.com/microsoft/python-type-stubs/issues/87
        df11: pd.DataFrame = pd.read_csv("foo")
        df12: pd.DataFrame = pd.read_csv("foo", iterator=False)
        df13: pd.DataFrame = pd.read_csv("foo", iterator=False, chunksize=None)
        df14: TextFileReader = pd.read_csv("foo", chunksize=0)
        df15: TextFileReader = pd.read_csv("foo", iterator=False, chunksize=0)
        df16: TextFileReader = pd.read_csv("foo", iterator=True)
        df17: TextFileReader = pd.read_csv("foo", iterator=True, chunksize=None)
        df18: TextFileReader = pd.read_csv("foo", iterator=True, chunksize=0)
        df19: TextFileReader = pd.read_csv("foo", chunksize=0)

        # https://github.com/microsoft/python-type-stubs/issues/118
        pd.read_csv("foo", storage_options=None)


def test_groupby_series_methods() -> None:
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3], "y": [10, 20, 30, 40, 50]})
    gb = df.groupby("x")["y"]
    check(assert_type(gb.describe(), pd.DataFrame), pd.DataFrame)
    gb.count().loc[2]
    gb.pct_change().loc[2]
    gb.bfill().loc[2]
    gb.cummax().loc[2]
    gb.cummin().loc[2]
    gb.cumprod().loc[2]
    gb.cumsum().loc[2]
    gb.ffill().loc[2]
    gb.first().loc[2]
    gb.head().loc[2]
    gb.last().loc[2]
    gb.max().loc[2]
    gb.mean().loc[2]
    gb.median().loc[2]
    gb.min().loc[2]
    gb.nlargest().loc[2]
    gb.nsmallest().loc[2]
    gb.nth(0).loc[2]


def test_indexslice_setitem():
    df = pd.DataFrame(
        {"x": [1, 2, 2, 3], "y": [1, 2, 3, 4], "z": [10, 20, 30, 40]}
    ).set_index(["x", "y"])
    s = pd.Series([-1, -2])
    df.loc[pd.IndexSlice[2, :]] = s.values
    df.loc[pd.IndexSlice[2, :], "z"] = [200, 300]


def test_compute_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    s: pd.Series = pd.Series([10, 20, 30, 40])
    result: pd.Series = df["x"] + s.values


# https://github.com/microsoft/python-type-stubs/issues/164
def test_sum_get_add() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    s = df["x"]
    check(assert_type(s, pd.Series), pd.Series)
    summer = df.sum(axis=1)
    check(assert_type(summer, pd.Series), pd.Series)

    s2: pd.Series = s + summer
    s3: pd.Series = s + df["y"]
    s4: pd.Series = summer + summer


def test_getset_untyped() -> None:
    result: int = 10
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    # Tests that Dataframe.__getitem__ needs to return untyped series.
    result = df["x"].max()


def test_getmultiindex_columns() -> None:
    mi = pd.MultiIndex.from_product([[1, 2], ["a", "b"]])
    df = pd.DataFrame([[1, 2, 3, 4], [10, 20, 30, 40]], columns=mi)
    li: list[tuple[int, str]] = [(1, "a"), (2, "b")]
    res1: pd.DataFrame = df[[(1, "a"), (2, "b")]]
    res2: pd.DataFrame = df[li]
    res3: pd.DataFrame = df[
        [(i, s) for i in [1] for s in df.columns.get_level_values(1)]
    ]
    ndf: pd.DataFrame = df[[df.columns[0]]]


def test_frame_getitem_isin() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    check(assert_type(df[df.index.isin([1, 3, 5])], pd.DataFrame), pd.DataFrame)


def test_to_excel() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_excel(file.name, engine="openpyxl")
        file.close()
        df2: pd.DataFrame = pd.read_excel(file.name)
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_excel(Path(file.name), engine="openpyxl")
        file.close()
        df3: pd.DataFrame = pd.read_excel(file.name)
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_excel(file.name, engine="openpyxl", startrow=1, startcol=1, header=False)
        file.close()
        df4: pd.DataFrame = pd.read_excel(file.name)
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_excel(file.name, engine="openpyxl", sheet_name="sheet", index=False)
        file.close()
        df5: pd.DataFrame = pd.read_excel(file.name)
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_excel(file.name, engine="openpyxl", header=["x", "y"])
        file.close()
        df6: pd.DataFrame = pd.read_excel(file.name)
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_excel(file.name, engine="openpyxl", columns=["col1"])
        file.close()
        df7: pd.DataFrame = pd.read_excel(file.name)


def test_read_excel() -> None:
    if TYPE_CHECKING:  # skip pytest

        # https://github.com/pandas-dev/pandas-stubs/pull/33
        df11: pd.DataFrame = pd.read_excel("foo")
        df12: pd.DataFrame = pd.read_excel("foo", sheet_name="sheet")
        df13: dict[int | str, pd.DataFrame] = pd.read_excel("foo", sheet_name=["sheet"])
        # GH 98
        df14: pd.DataFrame = pd.read_excel("foo", sheet_name=0)
        df15: dict[int | str, pd.DataFrame] = pd.read_excel("foo", sheet_name=[0])
        df16: dict[int | str, pd.DataFrame] = pd.read_excel(
            "foo", sheet_name=[0, "sheet"]
        )
        df17: dict[int | str, pd.DataFrame] = pd.read_excel("foo", sheet_name=None)


def test_join() -> None:
    float_frame = pd.DataFrame(getSeriesData())
    # GH 29
    left = float_frame["A"].to_frame()
    seriesB = float_frame["B"]
    frameCD = float_frame[["C", "D"]]
    right: list[pd.Series | pd.DataFrame] = [seriesB, frameCD]
    result = left.join(right)


def test_types_ffill() -> None:
    # GH 44
    df = pd.DataFrame([[1, 2, 3]])
    check(assert_type(df.ffill(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.ffill(inplace=False), pd.DataFrame), pd.DataFrame)
    assert assert_type(df.ffill(inplace=True), None) is None


def test_types_bfill() -> None:
    # GH 44
    df = pd.DataFrame([[1, 2, 3]])
    check(assert_type(df.bfill(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.bfill(inplace=False), pd.DataFrame), pd.DataFrame)
    assert assert_type(df.bfill(inplace=True), None) is None


def test_types_replace() -> None:
    # GH 44
    df = pd.DataFrame([[1, 2, 3]])
    check(assert_type(df.replace(1, 2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(1, 2, inplace=False), pd.DataFrame), pd.DataFrame)
    assert assert_type(df.replace(1, 2, inplace=True), None) is None


def test_loop_dataframe() -> None:
    # GH 70
    df = pd.DataFrame({"x": [1, 2, 3]})
    for c in df:
        check(assert_type(df[c], pd.Series), pd.Series)


def test_groupby_index() -> None:
    # GH 42
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]}
    ).set_index("col1")
    check(assert_type(df.groupby(df.index).min(), pd.DataFrame), pd.DataFrame)


def test_iloc_npint() -> None:
    # GH 69
    df = pd.DataFrame({"a": [10, 20, 30], "b": [20, 40, 60], "c": [30, 60, 90]})
    iloc = np.argmin(np.random.standard_normal(3))
    df.iloc[iloc]


def test_set_columns() -> None:
    # GH 73
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.0, 1, 1]})
    # Next line should work, but it is a mypy bug
    # https://github.com/python/mypy/issues/3004
    df.columns = ["c", "d"]  # type: ignore[assignment]


def test_frame_index_numpy() -> None:
    # GH 80
    i = np.array([1.0, 2.0])
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"], index=i)


def test_frame_reindex() -> None:
    # GH 84
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    df.reindex([2, 1, 0])


def test_frame_ndarray_assignmment() -> None:
    # GH 100
    df_a = pd.DataFrame({"a": [0.0] * 10})
    df_a.iloc[:, :] = np.array([[-1.0]] * 10)

    df_b = pd.DataFrame({"a": [0.0] * 10, "b": [1.0] * 10})
    df_b.iloc[:, :] = np.array([[-1.0, np.inf]] * 10)


def test_not_hashable() -> None:
    # GH 113
    assert assert_type(pd.DataFrame.__hash__, None) is None
    assert assert_type(pd.DataFrame().__hash__, None) is None
    assert assert_type(pd.Series.__hash__, None) is None
    assert assert_type(pd.Series([], dtype=object).__hash__, None) is None
    assert assert_type(pd.Index.__hash__, None) is None
    assert assert_type(pd.Index([]).__hash__, None) is None

    def test_func(h: Hashable):
        pass

    test_func(pd.DataFrame())  # type: ignore[arg-type]
    test_func(pd.Series([], dtype=object))  # type: ignore[arg-type]
    test_func(pd.Index([]))  # type: ignore[arg-type]


def test_columns_mixlist() -> None:
    # GH 97
    df = pd.DataFrame({"a": [1, 2, 3], 1: [3, 4, 5]})
    key: list[int | str]
    key = [1]
    check(assert_type(df[key], pd.DataFrame), pd.DataFrame)


def test_frame_scalars_slice() -> None:
    # GH 133
    # scalars:
    # str, bytes, datetime.date, datetime.datetime, datetime.timedelta, bool, int,
    # float, complex, Timestamp, Timedelta

    str_ = "a"
    bytes_ = b"7"
    date = datetime.date(1999, 12, 31)
    datetime_ = datetime.datetime(1999, 12, 31)
    timedelta = datetime.datetime(2000, 1, 1) - datetime.datetime(1999, 12, 31)
    bool_ = True
    int_ = 2
    float_ = 3.14
    complex_ = 1.0 + 3.0j
    timestamp = pd.Timestamp(0)
    pd_timedelta = pd.Timedelta(0, unit="D")
    none = None
    idx = [
        str_,
        bytes_,
        date,
        datetime_,
        timedelta,
        bool_,
        int_,
        float_,
        complex_,
        timestamp,
        pd_timedelta,
        none,
    ]
    values = np.arange(len(idx))[:, None] + np.arange(len(idx))
    df = pd.DataFrame(values, columns=idx, index=idx)

    # Note: bool_ cannot be tested since the index is object and pandas does not
    # support boolean access using loc except when the index is boolean
    check(assert_type(df.loc[str_], pd.Series), pd.Series)
    check(assert_type(df.loc[bytes_], pd.Series), pd.Series)
    check(assert_type(df.loc[date], pd.Series), pd.Series)
    check(assert_type(df.loc[datetime_], pd.Series), pd.Series)
    check(assert_type(df.loc[timedelta], pd.Series), pd.Series)
    check(assert_type(df.loc[int_], pd.Series), pd.Series)
    check(assert_type(df.loc[float_], pd.Series), pd.Series)
    check(assert_type(df.loc[complex_], pd.Series), pd.Series)
    check(assert_type(df.loc[timestamp], pd.Series), pd.Series)
    check(assert_type(df.loc[pd_timedelta], pd.Series), pd.Series)
    check(assert_type(df.loc[none], pd.Series), pd.Series)

    check(assert_type(df.loc[:, str_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, bytes_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, date], pd.Series), pd.Series)
    check(assert_type(df.loc[:, datetime_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, timedelta], pd.Series), pd.Series)
    check(assert_type(df.loc[:, int_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, float_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, complex_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, timestamp], pd.Series), pd.Series)
    check(assert_type(df.loc[:, pd_timedelta], pd.Series), pd.Series)
    check(assert_type(df.loc[:, none], pd.Series), pd.Series)


def test_boolean_loc() -> None:
    # Booleans can only be used in loc when the index is boolean
    df = pd.DataFrame([[0, 1], [1, 0]], columns=[True, False], index=[True, False])
    check(assert_type(df.loc[True], pd.Series), pd.Series)
    check(assert_type(df.loc[:, False], pd.Series), pd.Series)


def test_groupby_result() -> None:
    # GH 142
    df = pd.DataFrame({"a": [0, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]})
    iterator = df.groupby(["a", "b"]).__iter__()
    assert_type(iterator, Iterator[Tuple[Tuple, pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), Tuple[Tuple, pd.DataFrame])

    check(assert_type(index, Tuple), tuple, np.int64)
    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    iterator2 = df.groupby("a").__iter__()
    assert_type(iterator2, Iterator[Tuple[Scalar, pd.DataFrame]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), Tuple[Scalar, pd.DataFrame])

    check(assert_type(index2, Scalar), int)
    check(assert_type(value2, pd.DataFrame), pd.DataFrame)

    # Want to make sure these cases are differentiated
    for (k1, k2), g in df.groupby(["a", "b"]):
        pass

    for kk, g in df.groupby("a"):
        pass


def test_setitem_list():
    # GH 153
    lst1: list[str] = ["a", "b", "c"]
    lst2: list[int] = [1, 2, 3]
    lst3: list[float] = [4.0, 5.0, 6.0]
    lst4: list[tuple[str, int]] = [("a", 1), ("b", 2), ("c", 3)]
    lst5: list[complex] = [0 + 1j, 0 + 2j, 0 + 3j]

    columns: list[Hashable] = [
        "a",
        "b",
        "c",
        1,
        2,
        3,
        4.0,
        5.0,
        6.0,
        ("a", 1),
        ("b", 2),
        ("c", 3),
        0 + 1j,
        0 + 2j,
        0 + 3j,
    ]

    df = pd.DataFrame(np.empty((3, 15)), columns=columns)

    check(assert_type(df.set_index(lst1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst3), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst4), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst5), pd.DataFrame), pd.DataFrame)

    iter1: Iterator[str] = (v for v in lst1)
    iter2: Iterator[tuple[str, int]] = (v for v in lst4)
    check(assert_type(df.set_index(iter1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(iter2), pd.DataFrame), pd.DataFrame)


def test_groupby_apply() -> None:
    # GH 167
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    def sum_mean(x: pd.DataFrame) -> float:
        return x.sum().mean()

    check(assert_type(df.groupby("col1").apply(sum_mean), pd.Series), pd.Series)

    lfunc: Callable[[pd.DataFrame], float] = lambda x: x.sum().mean()
    check(
        assert_type(df.groupby("col1").apply(lfunc), pd.Series),
        pd.Series,
    )

    def sum_to_list(x: pd.DataFrame) -> list:
        return x.sum().tolist()

    check(assert_type(df.groupby("col1").apply(sum_to_list), pd.Series), pd.Series)

    def sum_to_series(x: pd.DataFrame) -> pd.Series:
        return x.sum()

    check(
        assert_type(df.groupby("col1").apply(sum_to_series), pd.DataFrame), pd.DataFrame
    )

    def sample_to_df(x: pd.DataFrame) -> pd.DataFrame:
        return x.sample()

    check(
        assert_type(df.groupby("col1").apply(sample_to_df), pd.DataFrame), pd.DataFrame
    )
