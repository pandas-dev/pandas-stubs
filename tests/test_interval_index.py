from __future__ import annotations

import pandas as pd
from typing_extensions import assert_type

from tests import check


def test_from_breaks() -> None:
    ind1: pd.IntervalIndex = pd.IntervalIndex.from_breaks([0, 1, 2, 3], name="test")
    ind2: pd.IntervalIndex = pd.IntervalIndex.from_breaks([0, 1, 2, 3], closed="right", name=123)

def test_from_arrays() -> None:
    ind1: pd.IntervalIndex = pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3], name="test")
    ind2: pd.IntervalIndex = pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3], closed="right", name=123)

def test_from_tuples() -> None:
    ind1: pd.IntervalIndex = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], name="test")
    ind2: pd.IntervalIndex = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3)], closed="right", name=123)

def test_is_overlapping() -> None:
    ind = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
    check(assert_type(ind.is_overlapping, bool), bool)
