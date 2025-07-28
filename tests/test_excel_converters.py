from functools import partial
import pandas as pd
from typing_extensions import assert_type

partial_func = partial(pd.to_datetime, errors="coerce")

df = pd.read_excel("foo.xlsx", converters={"field_1": partial_func})

assert_type(df, pd.DataFrame)