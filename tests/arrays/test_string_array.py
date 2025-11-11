import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    check(assert_type(pd.array(["🐼"]), StringArray), StringArray)
    check(
        assert_type(pd.array(["🐼", np.str_("🐼")]), StringArray),
        StringArray,
    )
    check(assert_type(pd.array(["🐼", None]), StringArray), StringArray)
    check(assert_type(pd.array(["🐼", pd.NA, None]), StringArray), StringArray)

    check(assert_type(pd.array(pd.array(["🐼"])), StringArray), StringArray)
