from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import StrDtypeArg
from tests.dtypes import ASTYPE_STRING_ARGS


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_STRING_ARGS.items(), ids=repr)
def test_astype_string(cast_arg: StrDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python string
        assert_type(s.astype(str), "pd.Series[str]")
        assert_type(s.astype("str"), "pd.Series[str]")
        # pandas string
        assert_type(s.astype(pd.StringDtype()), "pd.Series[str]")
        assert_type(s.astype("string"), "pd.Series[str]")
        # numpy string
        assert_type(s.astype(np.str_), "pd.Series[str]")
        assert_type(s.astype("str_"), "pd.Series[str]")
        assert_type(s.astype("unicode"), "pd.Series[str]")
        assert_type(s.astype("U"), "pd.Series[str]")
        # pyarrow string
        assert_type(s.astype("string[pyarrow]"), "pd.Series[str]")
