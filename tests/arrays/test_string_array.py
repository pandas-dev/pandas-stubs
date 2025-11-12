import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


def test_constructor() -> None:
    check(assert_type(pd.array(["🐼"]), StringArray), StringArray)
    check(
        assert_type(pd.array(["🐼", np.str_("🐼")]), StringArray),
        StringArray,
    )
    check(assert_type(pd.array(["🐼", None]), StringArray), StringArray)
    check(assert_type(pd.array(["🐼", pd.NA, None]), StringArray), StringArray)

    check(assert_type(pd.array(pd.array(["🐼"])), StringArray), StringArray)

    if TYPE_CHECKING_INVALID_USAGE:
        pd.array("🐼🎫")  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
