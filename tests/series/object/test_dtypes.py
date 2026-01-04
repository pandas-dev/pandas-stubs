from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check
from tests._typing import ObjectDtypeArg
from tests.dtypes import ASTYPE_OBJECT_ARGS


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_OBJECT_ARGS.items(), ids=repr)
def test_astype_object(cast_arg: ObjectDtypeArg, target_type: type) -> None:
    s = pd.Series([object(), 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python object
        assert_type(s.astype(object), pd.Series)
        assert_type(s.astype("object"), pd.Series)
        # numpy object
        assert_type(s.astype(np.object_), pd.Series)
        # assert_type(s.astype("object_"), pd.Series)  # NOTE: not assigned
        # assert_type(s.astype("object0"), pd.Series)  # NOTE: not assigned
        assert_type(s.astype("O"), pd.Series)
