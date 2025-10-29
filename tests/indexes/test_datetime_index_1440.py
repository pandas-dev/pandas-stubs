from typing import assert_type
import pandas as pd

# GH#1440 – constructing Index from DatetimeIndex should return DatetimeIndex
data = pd.date_range("2022-01-01", "2022-01-03", freq="D")
idx = pd.Index(data, name="date")

assert_type(idx, pd.DatetimeIndex)
