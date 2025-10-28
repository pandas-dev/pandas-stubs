from __future__ import annotations

import pandas as pd
from typing_extensions import (
    assert_type,
)

from tests import (
    check,
)


def test_rangeindex_floordiv() -> None:
    ri = pd.RangeIndex(3)
    check(
        assert_type(ri // 2, "pd.Index[int]"),
        pd.Index,
    )


def test_rangeindex_min_max() -> None:
    ri = pd.RangeIndex(3)
    check(
        assert_type(ri.min(), int),
        int,
    )
    check(
        assert_type(ri.max(axis=0), int),
        int,
    )


def test_rangeindex_equals() -> None:
    ri = pd.RangeIndex(3)
    check(
        assert_type(ri.equals(ri), bool),
        bool,
    )


def test_rangeindex_tolist() -> None:
    ri = pd.RangeIndex(3)
    check(
        assert_type(ri.tolist(), list[int]),
        list[int],
    )
