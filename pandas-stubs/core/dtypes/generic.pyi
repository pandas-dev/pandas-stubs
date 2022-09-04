from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.generic import NDFrame

ABCIndex = type[Index]

ABCNDFrame = type[NDFrame]
ABCSeries = type[Series]
ABCDataFrame = type[DataFrame]

ABCExtensionArray = type[ExtensionArray]
