from datetime import datetime

import pandas as pd
from typing_extensions import assert_type

from tests import check

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USMemorialDay,
    nearest_workday,
)


def test_custom_calendar() -> None:
    class ExampleCalendar(AbstractHolidayCalendar):
        rules = [
            USMemorialDay,
            Holiday("July 4th", month=7, day=4, observance=nearest_workday),
            Holiday(
                "Columbus Day",
                month=10,
                day=1,
                offset=pd.DateOffset(weekday=1),
            ),
        ]

    cal = ExampleCalendar()

    result = cal.holidays(datetime(2012, 1, 1), datetime(2012, 12, 31))
    check(assert_type(result, pd.DatetimeIndex), pd.DatetimeIndex)
