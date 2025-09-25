from typing import reveal_type

import pandas as pd

s1 = pd.Series(pd.to_datetime(["2022-05-01", "2022-06-01"]))
reveal_type(s1)
s2 = pd.Series(pd.to_datetime(["2022-05-15", "2022-06-15"]))
reveal_type(s2)
td = s1 - s2
reveal_type(td)
ssum = s1 + s2
