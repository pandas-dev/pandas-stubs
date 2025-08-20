import datetime as dt
from typing import (
    assert_type,
    reveal_type,
)

import numpy as np
import pandas as pd
from pandas.core.series import TimedeltaSeries  # noqa: F401

from tests import check

df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
s1 = df.min(axis=1)
s2 = df.max(axis=1)
sa = s1 + s2
ss = s1 - s2
sm = s1 * s2
sd = s1 / s2
check(assert_type(sa, pd.Series), pd.Series)
reveal_type(s1.__sub__(s2))
reveal_type(s2.__rsub__(s1))
check(assert_type(ss, pd.Series), pd.Series)
check(assert_type(sm, pd.Series), pd.Series)
check(assert_type(sd, pd.Series), pd.Series)

ts1 = pd.to_datetime(pd.Series(["2022-03-05", "2022-03-06"]))
assert isinstance(ts1.iloc[0], pd.Timestamp)
td1 = pd.to_timedelta([2, 3], "seconds")
ts2 = pd.to_datetime(pd.Series(["2022-03-08", "2022-03-10"]))
r1 = ts1 - ts2
check(assert_type(r1, "TimedeltaSeries"), pd.Series, pd.Timedelta)
r2 = r1 / td1
check(assert_type(r2, "pd.Series[float]"), pd.Series, float)
r3 = r1 - td1
check(assert_type(r3, "TimedeltaSeries"), pd.Series, pd.Timedelta)
r4 = pd.Timedelta(5, "days") / r1
check(assert_type(r4, "pd.Series[float]"), pd.Series, float)
sb = pd.Series([1, 2]) == pd.Series([1, 3])
check(assert_type(sb, "pd.Series[bool]"), pd.Series, np.bool_)
r5 = sb * r1
check(assert_type(r5, "TimedeltaSeries"), pd.Series, pd.Timedelta)
r6 = r1 * 4
check(assert_type(r6, "TimedeltaSeries"), pd.Series, pd.Timedelta)

tsp1 = pd.Timestamp("2022-03-05")
dt1 = dt.datetime(2022, 9, 1, 12, 5, 30)
r7 = ts1 - tsp1
check(assert_type(r7, "TimedeltaSeries"), pd.Series, pd.Timedelta)
r8 = ts1 - dt1
check(assert_type(r8, "TimedeltaSeries"), pd.Series, pd.Timedelta)
