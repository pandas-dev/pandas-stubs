from builtins import str as _str
from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.Series(["1", "23", "456"])  # left operand


def test_add_py_scalar() -> None:
    """Testpd.Series[str]+ Python native str"""
    r0 = "right"

    check(assert_type(left + r0, "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(r0 + left, "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.add(r0), "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.radd(r0), "pd.Series[_str]"), pd.Series, _str)


def test_add_py_sequence() -> None:
    """Testpd.Series[str]+ Python native sequence"""
    r0 = ["a", "bc", "def"]
    r1 = tuple(r0)

    check(assert_type(left + r0, "pd.Series[_str]"), pd.Series, _str)
    check(assert_type(left + r1, "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(r0 + left, "pd.Series[_str]"), pd.Series, _str)
    check(assert_type(r1 + left, "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.add(r0), "pd.Series[_str]"), pd.Series, _str)
    check(assert_type(left.add(r1), "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.radd(r0), "pd.Series[_str]"), pd.Series, _str)
    check(assert_type(left.radd(r1), "pd.Series[_str]"), pd.Series, _str)


def test_add_numpy_array() -> None:
    """Testpd.Series[str]+ numpy array"""
    r0 = np.array(["a", "bc", "def"], np.str_)

    check(assert_type(left + r0, "pd.Series[_str]"), pd.Series, _str)

    # `numpy` typing gives `Any` in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`s.
    check(assert_type(r0 + left, Any), pd.Series, _str)

    check(assert_type(left.add(r0), "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.radd(r0), "pd.Series[_str]"), pd.Series, _str)


def test_add_pd_series() -> None:
    """Testpd.Series[str]+ pandas series"""
    r0 = pd.Series(["a", "bc", "def"])

    check(assert_type(left + r0, "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(r0 + left, "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.add(r0), "pd.Series[_str]"), pd.Series, _str)

    check(assert_type(left.radd(r0), "pd.Series[_str]"), pd.Series, _str)
