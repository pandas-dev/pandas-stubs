"""Test module for arithmetic operations on Series."""

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check


def test_element_wise_int_int() -> None:
    s = pd.Series([0, 1, -10])
    s2 = pd.Series([7, -5, 10])

    check(assert_type(s + s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.add(s2, fill_value=0), "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s - s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.sub(s2, fill_value=0), "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s * s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.mul(s2, fill_value=0), "pd.Series[int]"), pd.Series, np.integer)

    # GH1089 should be the following
    check(assert_type(s / s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.div(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s // s2, "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s.floordiv(s2, fill_value=0), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )

    check(assert_type(s % s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.mod(s2, fill_value=0), "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s ** s2.abs(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s.pow(s2.abs(), fill_value=0), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )

    check(assert_type(divmod(s, s2), tuple["pd.Series[int]", "pd.Series[int]"]), tuple)


def test_element_wise_int_float() -> None:
    s = pd.Series([0, 1, -10])
    s2 = pd.Series([7.0, -5.5, 10.4])

    check(assert_type(s + s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.add(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s - s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.sub(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s * s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.mul(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    # GH1089 should be the following
    check(assert_type(s / s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.div(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(
        assert_type(s.floordiv(s2, fill_value=0), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )

    check(assert_type(s % s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.mod(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s ** s2.abs(), "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.pow(s2.abs(), fill_value=0), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )

    check(assert_type(divmod(s, s2), tuple["pd.Series[int]", "pd.Series[int]"]), tuple)


def test_element_wise_float_int() -> None:
    s = pd.Series([0.0, 1.4, -10.25])
    s2 = pd.Series([7, -5, 10])

    check(assert_type(s + s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.add(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s - s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.sub(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s * s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.mul(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    # GH1089 should be the following
    check(assert_type(s / s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.div(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s // s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.floordiv(s2, fill_value=0), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )

    check(assert_type(s % s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.mod(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s ** s2.abs(), "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.pow(s2.abs(), fill_value=0), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )

    check(
        assert_type(divmod(s, s2), tuple["pd.Series[float]", "pd.Series[float]"]), tuple
    )


def test_element_wise_float_float() -> None:
    s = pd.Series([0.439, 1.43829, -10.432])
    s2 = pd.Series([7.4389, -5.543, 10.432])

    check(assert_type(s + s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.add(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s - s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.sub(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s * s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.mul(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    # GH1089 should be the following
    check(assert_type(s / s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.div(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s // s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.floordiv(s2, fill_value=0), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )

    check(assert_type(s % s2, "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.mod(s2, fill_value=0), "pd.Series[float]"), pd.Series, np.float64
    )

    check(assert_type(s ** s2.abs(), "pd.Series[float]"), pd.Series, np.float64)
    check(
        assert_type(s.pow(s2.abs(), fill_value=0), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )

    check(
        assert_type(divmod(s, s2), tuple["pd.Series[float]", "pd.Series[float]"]), tuple
    )
