from collections.abc import Callable
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Any,
    assert_type,
)

from dateutil.relativedelta import (
    MO,
    relativedelta,
)
import numpy as np
import pandas as pd

from tests import check

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    USMemorialDay,
    nearest_workday,
)
from pandas.tseries.offsets import BaseOffset


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


def test_holiday_attributes() -> None:
    exclude = pd.DatetimeIndex([pd.Timestamp("2022-05-30")])
    offset = pd.DateOffset(weekday=1)

    holiday = Holiday(
        "Boxing Day",
        month=12,
        day=26,
        year=2020,
        offset=offset,
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2030, 1, 1),
        days_of_week=(0, 1),
        exclude_dates=exclude,
    )
    check(assert_type(holiday.name, str), str)
    check(assert_type(holiday.month, int | None), int)
    check(assert_type(holiday.day, int | None), int)
    check(assert_type(holiday.year, int | None), int)
    check(
        assert_type(holiday.days_of_week, tuple[int | relativedelta, ...] | None), tuple
    )
    check(assert_type(holiday.offset, BaseOffset | list[BaseOffset] | None), BaseOffset)
    check(assert_type(holiday.observance, Callable[..., Any] | None), type(None))
    check(assert_type(holiday.start_date, pd.Timestamp | None), pd.Timestamp)
    check(assert_type(holiday.end_date, pd.Timestamp | None), pd.Timestamp)
    check(assert_type(holiday.exclude_dates, pd.DatetimeIndex | None), pd.DatetimeIndex)

    holiday_with_observance = Holiday(
        "July 4th", month=7, day=4, observance=nearest_workday
    )
    check(
        assert_type(holiday_with_observance.observance, Callable[..., Any] | None),
        type(nearest_workday),
    )
    check(
        assert_type(
            holiday_with_observance.offset, BaseOffset | list[BaseOffset] | None
        ),
        type(None),
    )
    check(
        assert_type(holiday_with_observance.start_date, pd.Timestamp | None), type(None)
    )
    check(
        assert_type(holiday_with_observance.end_date, pd.Timestamp | None), type(None)
    )
    check(
        assert_type(holiday_with_observance.exclude_dates, pd.DatetimeIndex | None),
        type(None),
    )
    check(
        assert_type(
            holiday_with_observance.days_of_week, tuple[int | relativedelta, ...] | None
        ),
        type(None),
    )


def test_calendar_attributes() -> None:
    class ExampleCalendar(AbstractHolidayCalendar):
        rules = [
            USMemorialDay,
            Holiday("July 4th", month=7, day=4, observance=nearest_workday),
        ]

    cal = ExampleCalendar(name="example")
    check(assert_type(cal.name, str), str)
    check(assert_type(cal.rules, list[Holiday]), list)
    check(assert_type(cal.start_date, pd.Timestamp), pd.Timestamp)
    check(assert_type(cal.end_date, pd.Timestamp), pd.Timestamp)

    default_cal = AbstractHolidayCalendar()
    check(assert_type(default_cal.name, str), str)


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
