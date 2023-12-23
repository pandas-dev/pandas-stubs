import platform

import pandas as pd
from typing_extensions import assert_type

from tests import (
    check,
    pytest_warns_bounded,
)


def test_show_version():
    with pytest_warns_bounded(
        UserWarning,
        match="Setuptools is replacing distutils",
        upper="3.11.99",
        version_str=platform.python_version(),
    ):
        check(assert_type(pd.show_versions(True), None), type(None))
        check(assert_type(pd.show_versions(False), None), type(None))


def test_dummies():
    df = pd.DataFrame(
        pd.Series(["a", "b", "a", "b", "c", "a", "a"], dtype="category"), columns=["A"]
    )
    dummies = pd.get_dummies(df)
    check(assert_type(dummies, pd.DataFrame), pd.DataFrame)
    check(assert_type(pd.from_dummies(dummies), pd.DataFrame), pd.DataFrame)

    df2 = pd.DataFrame(
        pd.Series(["a", "b", "a", "b", "c", "a", "a"], dtype="category"),
        columns=[("A",)],
    )
    check(
        assert_type(pd.get_dummies(df2, prefix={("A",): "bar"}), pd.DataFrame),
        pd.DataFrame,
    )


def test_get_dummies_args():
    df = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", "b", "c", "a", "a"], dtype="category"),
            "B": pd.Series([1, 2, 1, 2, 3, 1, 1]),
        }
    )
    check(
        assert_type(
            pd.get_dummies(df, prefix="foo", prefix_sep="-", sparse=True), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.get_dummies(
                df, prefix=["foo"], dummy_na=True, drop_first=True, dtype="bool"
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.get_dummies(df, prefix={"A": "foo", "B": "baz"}, columns=["A", "B"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_from_dummies_args():
    df = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", "b", "c", "a", "a"], dtype="category"),
        }
    )
    dummies = pd.get_dummies(df, drop_first=True)

    check(
        assert_type(
            pd.from_dummies(dummies, sep="_", default_category="a"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
