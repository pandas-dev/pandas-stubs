import pandas as pd
from pandas.core.arrays.period import PeriodArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    prd = pd.Period("2023-01-01")
    check(
        assert_type(  # type: ignore[assert-type]  # I do not understand
            pd.array([prd]), PeriodArray
        ),
        PeriodArray,
    )
    check(
        assert_type(  # type: ignore[assert-type]  # I do not understand
            pd.array([prd, None]), PeriodArray
        ),
        PeriodArray,
    )
    check(
        assert_type(  # type: ignore[assert-type]  # I do not understand
            pd.array([prd, pd.NaT, None]), PeriodArray
        ),
        PeriodArray,
    )

    check(
        assert_type(  # type: ignore[assert-type]  # I do not understand
            pd.array(pd.array([prd])), PeriodArray
        ),
        PeriodArray,
    )

    check(assert_type(pd.array(pd.Index([prd])), PeriodArray), PeriodArray)

    check(assert_type(pd.array(pd.Series([prd])), PeriodArray), PeriodArray)
