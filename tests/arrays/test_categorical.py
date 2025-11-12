import numpy as np
import pandas as pd
from pandas.core.arrays.categorical import Categorical
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    check(assert_type(pd.array(["🐼"], dtype="category"), Categorical), Categorical)
    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd.array(np.array(["🐼"]), dtype="category"), Categorical
        ),
        Categorical,
    )
    check(
        assert_type(pd.array(pd.array(["🐼"]), dtype="category"), Categorical),
        Categorical,
    )
    check(
        assert_type(pd.array(pd.Index(["🐼"]), dtype="category"), Categorical),
        Categorical,
    )
    check(
        assert_type(pd.array(pd.Series(["🐼"]), dtype="category"), Categorical),
        Categorical,
    )

    check(
        assert_type(pd.array(pd.array(["🐼"], dtype="category")), Categorical),
        Categorical,
    )
    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd.array(pd.Index(["🐼"], dtype="category")), Categorical
        ),
        Categorical,
    )
    # TODO: Categorical Series pandas-dev/pandas-stubs#1415
    # check(
    #     assert_type(pd.array(pd.Series(["🐼"], dtype="category")), Categorical),
    #     Categorical,
    # )
