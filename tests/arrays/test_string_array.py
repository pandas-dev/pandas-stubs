from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import PandasStrDtypeArg
from tests.dtypes import PANDAS_STRING_ARGS


@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_STRING_ARGS.items())
def test_construction_dtype(dtype: PandasStrDtypeArg, target_dtype: type) -> None:
    check(pd.array(["🐼", np.nan], dtype=dtype), target_dtype)

    if TYPE_CHECKING:
        assert_type(pd.array(["🐼", np.nan], dtype=pd.StringDtype()), StringArray)
        assert_type(pd.array(["🐼", np.nan], dtype="string"), StringArray)
