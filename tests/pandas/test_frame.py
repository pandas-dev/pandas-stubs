# flake8: noqa: F841
from datetime import date, datetime
import io
import tempfile
from pathlib import Path
from typing import List, Tuple, Iterable, Any, Dict, Hashable

import pandas as pd
from pandas.io.parsers import TextFileReader
import numpy as np

from . import check_dataframe_result, check_series_result

import pytest


def test_types_init() -> None:
    pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}, index=[2, 1])
    pd.DataFrame(data=[1, 2, 3, 4], dtype=np.int8)
    pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"], dtype=np.int8, copy=True)


def test_types_append() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})

    res1: pd.DataFrame = df.append(df2)
    res2: pd.DataFrame = df.append([1, 2, 3])
    res3: pd.DataFrame = df.append([[1, 2, 3]])
    res4: pd.DataFrame = df.append({("a", 1): [1, 2, 3], "b": df2}, ignore_index=True)
    res5: pd.DataFrame = df.append({1: [1, 2, 3]}, ignore_index=True)
    res6: pd.DataFrame = df.append({1: [1, 2, 3], "col2": [1, 2, 3]}, ignore_index=True)
    res7: pd.DataFrame = df.append(pd.Series([5, 6]), ignore_index=True)
    res8: pd.DataFrame = df.append(pd.Series([5, 6], index=["col1", "col2"]), ignore_index=True)


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
    df.sample(frac=0.5)
    df.sample(n=1)


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
    level2: List[str] = ["a", "b", "c"]
    res3: pd.DataFrame = df2.sort_index(level=level2)
    res4: pd.DataFrame = df.sort_index(ascending=False, level=3)
    res5: None = df.sort_index(kind="mergesort", inplace=True)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_index_with_key() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=["a", "b", "C", "d"])
    res: pd.DataFrame = df.sort_index(key=lambda k: k.str.lower())


def test_types_set_index() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}, index=[5, 1, 3, 2])
    res: pd.DataFrame = df.set_index("col1")
    res2: pd.DataFrame = df.set_index("col1", drop=False)
    res3: pd.DataFrame = df.set_index("col1", append=True)
    res4: pd.DataFrame = df.set_index("col1", verify_integrity=True)
    res5: pd.DataFrame = df.set_index(["col1", "col2"])
    res6: None = df.set_index("col1", inplace=True)


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
    df2: pd.DataFrame = df.mean(level=0)
    df3: pd.DataFrame = df.mean(axis=1, level=0)
    df4: pd.DataFrame = df.mean(1, True, level=0)
    s3: pd.Series = df.mean(axis=1, skipna=True, numeric_only=False)


def test_types_median() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    s1: pd.Series = df.median()
    s2: pd.Series = df.median(axis=0)
    df2: pd.DataFrame = df.median(level=0)
    df3: pd.DataFrame = df.median(axis=1, level=0)
    df4: pd.DataFrame = df.median(1, True, level=0)
    s3: pd.Series = df.median(axis=1, skipna=True, numeric_only=False)


def test_types_itertuples() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    res1: Iterable[Tuple[Any, ...]] = df.itertuples()
    res2: Iterable[Tuple[Any, ...]] = df.itertuples(index=False, name="Foobar")
    res3: Iterable[Tuple[Any, ...]] = df.itertuples(index=False, name=None)


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
    df.apply(lambda x: x ** 2)
    df.apply(np.exp)
    df.apply(str)


def test_types_applymap() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.applymap(lambda x: x ** 2)
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

    res_pow: pd.DataFrame = df2 ** df
    res_pow2: pd.DataFrame = df2.pow(df, fill_value=0)

    # divmod operation was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    # noinspection PyTypeChecker
    res_divmod: Tuple[pd.DataFrame, pd.DataFrame] = divmod(df, df2)
    res_divmod2: Tuple[pd.DataFrame, pd.DataFrame] = df.__divmod__(df2)
    res_rdivmod: Tuple[pd.DataFrame, pd.DataFrame] = df.__rdivmod__(df2)


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

    res_pow: pd.DataFrame = df ** 2
    res_pow1: pd.DataFrame = df ** 0
    res_pow2: pd.DataFrame = df ** 0.213
    res_pow3: pd.DataFrame = df.pow(0.5)


def test_types_melt() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.melt()
    df.melt(id_vars=["col1"], value_vars=["col2"])
    df.melt(id_vars=["col1"], value_vars=["col2"], var_name="someVariable", value_name="someValue")

    pd.melt(df)
    pd.melt(df, id_vars=["col1"], value_vars=["col2"])
    pd.melt(df, id_vars=["col1"], value_vars=["col2"], var_name="someVariable", value_name="someValue")


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
    df3: pd.DataFrame = df.groupby(by="col1", sort=False, as_index=True).transform(lambda x: x.max())
    df4: pd.DataFrame = df.groupby(by=["col1", "col2"]).count()
    df5: pd.DataFrame = df.groupby(by=["col1", "col2"]).filter(lambda x: x["col1"] > 0)
    df6: pd.DataFrame = df.groupby(by=["col1", "col2"]).nunique()
    df7: pd.DataFrame = df.groupby(by="col1").apply(sum)
    df8: pd.DataFrame = df.groupby("col1").transform("sum")
    s1: pd.Series = df.set_index("col1")["col2"]
    s2: pd.Series = s1.groupby("col1").transform("sum")
    s3: pd.Series = df.groupby("col1")["col3"].agg(min)
    df9: pd.DataFrame = df.groupby("col1")["col3"].agg([min, max])


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_group_by_with_dropna_keyword() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]})
    df.groupby(by="col2", dropna=True).sum()
    df.groupby(by="col2", dropna=False).sum()
    df.groupby(by="col2").sum()


def test_types_groupby_any() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [True, False, False], "col3": [False, False, False]})
    check_dataframe_result(df.groupby("col1").any())
    check_dataframe_result(df.groupby("col1").all())
    check_series_result(df.groupby("col1")["col2"].any())
    check_series_result(df.groupby("col1")["col2"].any())


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
    l = ["col1", "col2"]
    df.merge(df2, on=l)


def test_types_plot() -> None:
    pytest.skip()
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.plot.hist()
    df.plot.scatter(x="col2", y="col1")


def test_types_window() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.expanding()
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
    df1 = pd.DataFrame(data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]})
    df2 = pd.DataFrame(data={"col1": [1, 2, 5, 6], "col2": [3, 4, 1, 1], "col3": [3, 4, 3, 2]})
    df1.compare(df2)
    df2.compare(df1, align_axis=0, keep_shape=True, keep_equal=True)


def test_types_agg() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    df.agg("min")
    df.agg(x=("A", max), y=("B", "min"), z=("C", np.mean))
    df.agg("mean", axis=1)


def test_types_describe() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 2, -4], "col2": [np.datetime64("2000-01-01"), np.datetime64("2010-01-01"), np.datetime64("2010-01-01")]}
    )
    df.describe()
    df.describe(percentiles=[0.5], include="all")
    df.describe(exclude=[np.number])
    # datetime_is_numeric param added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.describe(datetime_is_numeric=True)


def test_types_to_string() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, None, -4],
            "col2": [np.datetime64("2000-01-01"), np.datetime64("2010-01-01"), np.datetime64("2010-01-01")],
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
            "col2": [np.datetime64("2000-01-01"), np.datetime64("2010-01-01"), np.datetime64("2010-01-01")],
        }
    )
    df.to_html(index=True, col_space=2, header=True, na_rep="0", justify="left", max_rows=2, max_cols=2, show_dimensions=True)
    # col_space accepting list or dict added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_html(col_space=[1, 2])
    df.to_html(col_space={"col1": 1, "col2": 3})


def test_types_resample() -> None:
    df = pd.DataFrame({"values": [2, 11, 3, 13, 14, 18, 17, 19]})
    df["date"] = pd.date_range("01/01/2018", periods=8, freq="W")
    df.resample("M", on="date")
    # origin and offset params added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.resample("20min", origin="epoch", offset=pd.Timedelta(2, "minutes"), on="date")


def test_types_from_dict() -> None:
    pd.DataFrame.from_dict({"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]})
    pd.DataFrame.from_dict({1: [3, 2, 1, 0], 2: ["a", "b", "c", "d"]})
    pd.DataFrame.from_dict({"a": {1: 2}, "b": {3: 4, 1: 4}}, orient="index")
    pd.DataFrame.from_dict({"a": {"row1": 2}, "b": {"row2": 4, "row1": 4}})
    pd.DataFrame.from_dict({"a": (1, 2, 3), "b": (2, 4, 5)})
    pd.DataFrame.from_dict(data={"col_1": {"a": 1}, "col_2": {"a": 1, "b": 2}}, orient="columns")


def test_pipe() -> None:
    pytest.skip("jinja2")

    def foo(df: pd.DataFrame) -> pd.DataFrame:
        return df

    df1: pd.DataFrame = pd.DataFrame({"a": [1]}).pipe(foo)

    df2: pd.DataFrame = (
        pd.DataFrame({"price": [10, 11, 9, 13, 14, 18, 17, 19], "volume": [50, 60, 40, 100, 50, 100, 40, 50]})
        .assign(week_starting=pd.date_range("01/01/2018", periods=8, freq="W"))
        .resample("M", on="week_starting")
        .pipe(foo)
    )

    df3: pd.DataFrame = pd.DataFrame({"a": [1], "b": [1]}).groupby("a").pipe(foo)

    df4: pd.DataFrame = pd.DataFrame({"a": [1], "b": [1]}).style.pipe(foo)


# set_flags() method added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
def test_types_set_flags() -> None:
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"]).set_flags(allows_duplicate_labels=False)
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "A"]).set_flags(allows_duplicate_labels=True)
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "A"])


def test_types_to_parquet() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("fastparquet")
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"]).set_flags(allows_duplicate_labels=False)
    with tempfile.NamedTemporaryFile(delete=False) as file:
        df.to_parquet(Path(file.name))
        file.close()
    # to_parquet() returns bytes when no path given since 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    b: bytes = df.to_parquet()


def test_types_to_latex() -> None:
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    df.to_latex(columns=["A"], label="some_label", caption="some_caption", multirow=True)
    df.to_latex(escape=False, decimal=",", column_format="r")
    # position param was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.to_latex(position="some")
    # caption param was extended to accept tuple in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
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
    df.rename(columns={type("AnyObject")(): "b"})
    df.rename(columns={(2, 1): "b"})
    df.rename(columns=lambda s: s.upper())
    f = lambda s: s.upper()
    df.rename(columns=f)


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

    df = pd.DataFrame({"a_col": list(range(10)), "a_nother": list(range(10)), "b_col": list(range(10))})
    df.loc[:, lambda df: df.columns.str.startswith("a_")]

    df = df[::-1]

    # https://github.com/microsoft/python-type-stubs/issues/69
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([4, 5, 6])
    df = pd.concat([s1, s2], axis=1)
    ss1: pd.Series = pd.concat([s1, s2], axis=0)
    ss2: pd.Series = pd.concat([s1, s2])

    # https://github.com/microsoft/python-type-stubs/issues/110
    d: date = pd.Timestamp("2021-01-01")
    tslist: List[pd.Timestamp] = list(pd.to_datetime(["2022-01-01", "2022-01-02"]))
    sseries: pd.Series = pd.Series(tslist)
    sseries_plus1: pd.Series = sseries + pd.Timedelta(1, "d")

    # https://github.com/microsoft/pylance-release/issues/2133
    dr = pd.date_range(start="2021-12-01", periods=24, freq="H")
    time = dr.strftime("%H:%M:%S")

    # https://github.com/microsoft/python-type-stubs/issues/115
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
    pd.DatetimeIndex(data=df["A"], tz=None, normalize=False, closed=None, ambiguous="NaT", copy=True)


def test_read_csv() -> None:
    pytest.skip()
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
    check_dataframe_result(gb.describe())
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
    df = pd.DataFrame({"x": [1, 2, 2, 3], "y": [1, 2, 3, 4], "z": [10, 20, 30, 40]}).set_index(["x", "y"])
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
    check_series_result(s)
    summer: pd.Series = df.sum(axis=1)
    check_series_result(summer)

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
    li: List[Tuple[int, str]] = [(1, "a"), (2, "b")]
    res1: pd.DataFrame = df[[(1, "a"), (2, "b")]]
    res2: pd.DataFrame = df[li]
    res3: pd.DataFrame = df[[(i, s) for i in [1] for s in df.columns.get_level_values(1)]]
    ndf: pd.DataFrame = df[[df.columns[0]]]


def test_frame_getitem_isin() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    check_dataframe_result(df[df.index.isin([1, 3, 5])])
