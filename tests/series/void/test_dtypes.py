from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import VoidDtypeArg
from tests.dtypes import ASTYPE_VOID_ARGS


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_VOID_ARGS.items(), ids=repr)
def test_astype_void(cast_arg: VoidDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # numpy void
        assert_type(s.astype(np.void), pd.Series)
        assert_type(s.astype("void"), pd.Series)
        assert_type(s.astype("V"), pd.Series)
