import pandas as pd

from tests import TYPE_CHECKING_INVALID_USAGE

series = pd.Series(["1", "a", "🐼"])


def test_window_str() -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        series.cumprod()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue]
