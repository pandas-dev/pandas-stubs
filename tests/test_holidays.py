from datetime import (
    datetime,
    timedelta,
)
from typing import assert_type

from dateutil.relativedelta import MO
import numpy as np
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


def test_custom_business_month() -> None:

    cal = np.busdaycalendar()

    check(
        assert_type(
            pd.offsets.CustomBusinessMonthBegin(calendar=cal),
            pd.offsets.CustomBusinessMonthBegin,
        ),
        pd.offsets.CustomBusinessMonthBegin,
    )
    check(
        assert_type(
            pd.offsets.CustomBusinessMonthEnd(calendar=cal),
            pd.offsets.CustomBusinessMonthEnd,
        ),
        pd.offsets.CustomBusinessMonthEnd,
    )
    check(
        assert_type(
            pd.offsets.CustomBusinessMonthBegin(weekmask="Mon Tue Wed Thu Fri"),
            pd.offsets.CustomBusinessMonthBegin,
        ),
        pd.offsets.CustomBusinessMonthBegin,
    )
    check(
        assert_type(
            pd.offsets.CustomBusinessMonthEnd(weekmask="Mon Tue Wed Thu Fri"),
            pd.offsets.CustomBusinessMonthEnd,
        ),
        pd.offsets.CustomBusinessMonthEnd,
    )
    check(
        assert_type(
            pd.offsets.CustomBusinessMonthBegin(offset=timedelta(hours=1)),
            pd.offsets.CustomBusinessMonthBegin,
        ),
        pd.offsets.CustomBusinessMonthBegin,
    )
    check(
        assert_type(
            pd.offsets.CustomBusinessMonthEnd(offset=timedelta(hours=1)),
            pd.offsets.CustomBusinessMonthEnd,
        ),
        pd.offsets.CustomBusinessMonthEnd,
    )


def test_rollforward_rollback_return_type() -> None:

    bmb = pd.offsets.CustomBusinessMonthBegin()
    bme = pd.offsets.CustomBusinessMonthEnd()

    check(
        assert_type(bmb.rollforward(datetime(2024, 1, 1)), pd.Timestamp), pd.Timestamp
    )
    check(assert_type(bme.rollback(datetime(2024, 1, 1)), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(bmb.rollforward(pd.Timestamp("2024-01-01")), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(bme.rollback(pd.Timestamp("2024-01-01")), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(bmb.rollforward(datetime(2024, 1, 1, 10, 30, 45)), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(bme.rollback(datetime(2024, 1, 1, 10, 30, 45)), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(bmb.rollforward(pd.Timestamp("2024-01-01 10:30:45")), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(bme.rollback(pd.Timestamp("2024-01-01 10:30:45")), pd.Timestamp),
        pd.Timestamp,
    )
