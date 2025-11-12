import numpy as np
import pandas as pd
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    check(
        assert_type(pd.array([pd.NA, None]), NumpyExtensionArray), NumpyExtensionArray
    )

    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd.array([1, "🐼"]), NumpyExtensionArray
        ),
        NumpyExtensionArray,
    )
    check(
        assert_type(  # type: ignore[assert-type] # I do not understand, mypy must have problem with two Generic Variables somehow
            pd.array(np.array([1, "🐼"], np.object_)), NumpyExtensionArray
        ),
        NumpyExtensionArray,
    )
    check(
        assert_type(pd.array(pd.array([pd.NA, None])), NumpyExtensionArray),
        NumpyExtensionArray,
    )
    check(
        assert_type(pd.array(pd.RangeIndex(0, 1)), NumpyExtensionArray),
        NumpyExtensionArray,
    )
