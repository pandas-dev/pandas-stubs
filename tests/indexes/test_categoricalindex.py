from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import (
    assert_type,
)

from tests import (
    check,
    np_1darray,
)


def test_categoricalindex_unique() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    check(
        assert_type(ci.unique(), "pd.CategoricalIndex[str]"),
        pd.CategoricalIndex,
    )


def test_categoricalindex_reindex() -> None:
    ci = pd.CategoricalIndex(["a", "b"])
    reindexed = ci.reindex(["b", "c"])
    check(assert_type(reindexed[0], pd.Index), pd.Index)
    check(assert_type(reindexed[1], np_1darray[np.intp] | None), np_1darray)
