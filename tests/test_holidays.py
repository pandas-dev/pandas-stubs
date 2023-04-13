from datetime import datetime

from dateutil.relativedelta import MO
import pandas as pd
from pandas.tseries.holiday import (
    Holiday,
    USMemorialDay,
    AbstractHolidayCalendar,
    nearest_workday,
)
from typing_extensions import assert_type

from tests import check


def test_custom_calendar() -> None:
    class ExampleCalendar(AbstractHolidayCalendar):
        rules = [
            USMemorialDay,
            Holiday("July 4th", month=7, day=4, observance=nearest_workday),
            Holiday(
                "Columbus Day",
                month=10,
                day=1,
                offset=pd.DateOffset(weekday=MO(2)),
            ),
        ]

    cal = ExampleCalendar()

    result = cal.holidays(datetime(2012, 1, 1), datetime(2012, 12, 31))
    check(assert_type(result, pd.DatetimeIndex), pd.DatetimeIndex)

    result = pd.date_range(
        start="7/1/2012", end="7/10/2012", freq=pd.offsets.CDay(calendar=cal)
    ).to_pydatetime()
