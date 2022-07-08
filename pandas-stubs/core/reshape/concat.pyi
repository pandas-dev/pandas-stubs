from typing import (
    Hashable,
    Iterable,
    Literal,
    Mapping,
    TypeVar,
    Union,
    overload,
)

from pandas import (
    DataFrame,
    Series,
)

from pandas._typing import (
    Axis,
    HashableT,
)

@overload
def concat(
    objs: Union[Iterable[DataFrame], Mapping[HashableT, DataFrame]],
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys=...,
    levels=...,
    names=...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> DataFrame: ...
@overload
def concat(
    objs: Union[Iterable[Series], Mapping[HashableT, Series]],
    axis: Literal[0, "index"] = ...,
    join: str = ...,
    ignore_index: bool = ...,
    keys=...,
    levels=...,
    names=...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> Series: ...
@overload
def concat(
    objs: Union[
        Iterable[Union[Series, DataFrame]], Mapping[HashableT, Union[Series, DataFrame]]
    ],
    axis: Literal[1, "columns"],
    join: str = ...,
    ignore_index: bool = ...,
    keys=...,
    levels=...,
    names=...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...,
) -> DataFrame: ...

# Including either of the next 2 overloads causes mypy to complain about
# test_pandas.py:test_types_concat() in assert_type(pd.concat([s, s2]), "pd.Series")
# It thinks that pd.concat([s, s2]) is Any .  May be due to Series being
# Generic, or the axis argument being unspecified, and then there is partial
# overlap with the first 2 overloads.
#
# @overload
# def concat(
#     objs: Union[
#         Iterable[Union[Series, DataFrame]], Mapping[HashableT, Union[Series, DataFrame]]
#     ],
#     axis: Literal[0, "index"] = ...,
#     join: str = ...,
#     ignore_index: bool = ...,
#     keys=...,
#     levels=...,
#     names=...,
#     verify_integrity: bool = ...,
#     sort: bool = ...,
#     copy: bool = ...,
# ) -> Union[DataFrame, Series]: ...

# @overload
# def concat(
#     objs: Union[
#         Iterable[Union[Series, DataFrame]], Mapping[HashableT, Union[Series, DataFrame]]
#     ],
#     axis: Axis = ...,
#     join: str = ...,
#     ignore_index: bool = ...,
#     keys=...,
#     levels=...,
#     names=...,
#     verify_integrity: bool = ...,
#     sort: bool = ...,
#     copy: bool = ...,
# ) -> Union[DataFrame, Series]: ...
