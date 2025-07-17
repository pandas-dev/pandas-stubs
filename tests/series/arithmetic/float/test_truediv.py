import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

left = pd.Series([1.0, 2.0, 3.0])  # left operand


def test_truediv_py_scalar() -> None:
    """Test pd.Series[float] / Python native scalars"""
    i, f, c = 1, 1.0, 1j

    check(assert_type(left / i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i / left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(f / left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.truediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.div(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rtruediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rdiv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_truediv_py_sequence() -> None:
    """Test pd.Series[float] / Python native sequence"""
    i, f, c = [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left / i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i / left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(f / left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.truediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.div(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rtruediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rdiv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_truediv_numpy_array() -> None:
    """Test pd.Series[float] / numpy array"""
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left / i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complex128)

    # numpy typing gives ndarray instead of `pd.Series[...]` in reality, which we cannot fix
    check(
        assert_type(  # type: ignore[assert-type]
            i / left, "pd.Series[float]"  # pyright: ignore[reportAssertTypeFailure]
        ),
        pd.Series,
        np.float64,
    )
    check(
        assert_type(  # type: ignore[assert-type]
            f / left, "pd.Series[float]"  # pyright: ignore[reportAssertTypeFailure]
        ),
        pd.Series,
        np.float64,
    )
    check(
        assert_type(  # type: ignore[assert-type]
            c / left, "pd.Series[complex]"  # pyright: ignore[reportAssertTypeFailure]
        ),
        pd.Series,
        np.complex128,
    )

    check(assert_type(left.truediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.div(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rtruediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rdiv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complex128)


def test_truediv_pd_series() -> None:
    """Test pd.Series[float] / pandas series"""
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left / i, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / f, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left / c, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(i / left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(f / left, "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(c / left, "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.truediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.truediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.div(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.div(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rtruediv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rtruediv(c), "pd.Series[complex]"), pd.Series, np.complex128)

    check(assert_type(left.rdiv(i), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(f), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(left.rdiv(c), "pd.Series[complex]"), pd.Series, np.complex128)
