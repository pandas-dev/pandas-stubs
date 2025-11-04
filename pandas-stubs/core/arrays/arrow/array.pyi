from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.base import ExtensionArraySupportsAnyAll

class ArrowExtensionArray(
    OpsMixin, ExtensionArraySupportsAnyAll, ArrowStringArrayMixin
): ...
