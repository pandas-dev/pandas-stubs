from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.strings.object_array import ObjectStringArrayMixin

class NumpyExtensionArray(
    OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin
): ...
