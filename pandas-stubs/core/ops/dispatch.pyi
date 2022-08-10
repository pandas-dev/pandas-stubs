import numpy as np

from pandas.core.dtypes.generic import (
    ABCExtensionArray as ABCExtensionArray,
    ABCSeries,
)

def should_extension_dispatch(left: ABCSeries, right) -> bool: ...
