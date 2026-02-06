from datetime import datetime
from typing import assert_type

from dateutil.relativedelta import MO
import pandas as pd

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


def test_holiday_exclude_dates() -> None:
    """Test construction of a Holiday with dates excluded from it."""
    exclude = pd.DatetimeIndex([pd.Timestamp("2022-05-30")])  # Queen's platinum Jubilee

    queens_jubilee_uk_spring_bank_holiday = Holiday(
        "Queen's Jubilee UK Spring Bank Holiday",
        month=5,
        day=31,
        offset=pd.DateOffset(weekday=MO(-1)),
        exclude_dates=exclude,
    )
    check(assert_type(queens_jubilee_uk_spring_bank_holiday, Holiday), Holiday)
