from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.string_ import BaseStringArray
from pandas.core.strings.object_array import ObjectStringArrayMixin

class ArrowStringArray(
    ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray
): ...
