import numpy as np

from pandas.core.dtypes.generic import ABCSeries

def should_extension_dispatch(left: ABCSeries, right) -> bool: ...
