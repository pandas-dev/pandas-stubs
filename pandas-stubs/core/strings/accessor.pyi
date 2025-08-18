# pyright: strict
from builtins import slice as _slice
from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
import re
from typing import (
    Generic,
    Literal,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
from pandas.core.base import NoNewAttributesMixin

from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    AlignJoin,
    DtypeObj,
    Scalar,
    T,
    np_1darray,
)

# Used for the result of str.split with expand=True
_T_EXPANDING = TypeVar("_T_EXPANDING", bound=DataFrame | MultiIndex)
# Used for the result of str.split with expand=False
_T_LIST_STR = TypeVar("_T_LIST_STR", bound=Series[list[str]] | Index[list[str]])
# Used for the result of str.match
_T_BOOL = TypeVar("_T_BOOL", bound=Series[bool] | np_1darray[np.bool])
# Used for the result of str.index / str.find
_T_INT = TypeVar("_T_INT", bound=Series[int] | Index[int])
# Used for the result of str.encode
_T_BYTES = TypeVar("_T_BYTES", bound=Series[bytes] | Index[bytes])
# Used for the result of str.decode
_T_STR = TypeVar("_T_STR", bound=Series[str] | Index[str])
# Used for the result of str.partition
_T_OBJECT = TypeVar("_T_OBJECT", bound=Series | Index)

class StringMethods(
    NoNewAttributesMixin,
    Generic[T, _T_EXPANDING, _T_BOOL, _T_LIST_STR, _T_INT, _T_BYTES, _T_STR, _T_OBJECT],
):
    def __init__(self, data: T) -> None: ...
    def __getitem__(self, key: _slice | int) -> _T_STR: ...
    def __iter__(self) -> _T_STR: ...
    @overload
    def cat(
        self,
        others: None = None,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> str: ...
    @overload
    def cat(
        self,
        others: (
            Series[str] | Index[str] | pd.DataFrame | npt.NDArray[np.str_] | list[str]
        ),
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> _T_STR: ...
    @overload
    def split(
        self,
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
        regex: bool | None = None,
    ) -> _T_EXPANDING: ...
    @overload
    def split(
        self,
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
        regex: bool | None = None,
    ) -> _T_LIST_STR: ...
    @overload
    def rsplit(
        self, pat: str | None = None, *, n: int = -1, expand: Literal[True]
    ) -> _T_EXPANDING: ...
    @overload
    def rsplit(
        self, pat: str | None = None, *, n: int = -1, expand: Literal[False] = False
    ) -> _T_LIST_STR: ...
    @overload  # expand=True
    def partition(
        self, sep: str = " ", expand: Literal[True] = True
    ) -> _T_EXPANDING: ...
    @overload  # expand=False (positional argument)
    def partition(self, sep: str, expand: Literal[False]) -> _T_OBJECT: ...
    @overload  # expand=False (keyword argument)
    def partition(self, sep: str = " ", *, expand: Literal[False]) -> _T_OBJECT: ...
    @overload  # expand=True
    def rpartition(
        self, sep: str = " ", expand: Literal[True] = True
    ) -> _T_EXPANDING: ...
    @overload  # expand=False (positional argument)
    def rpartition(self, sep: str, expand: Literal[False]) -> _T_OBJECT: ...
    @overload  # expand=False (keyword argument)
    def rpartition(self, sep: str = " ", *, expand: Literal[False]) -> _T_OBJECT: ...
    def get(self, i: int | Hashable) -> _T_STR: ...
    def join(self, sep: str) -> _T_STR: ...
    def contains(
        self,
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
        regex: bool = True,
    ) -> _T_BOOL: ...
    def match(
        self,
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> _T_BOOL: ...
    def fullmatch(
        self,
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> _T_BOOL: ...
    def replace(
        self,
        pat: str | re.Pattern[str],
        repl: str | Callable[[re.Match[str]], str],
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> _T_STR: ...
    def repeat(self, repeats: int | Sequence[int]) -> _T_STR: ...
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> _T_STR: ...
    def center(self, width: int, fillchar: str = " ") -> _T_STR: ...
    def ljust(self, width: int, fillchar: str = " ") -> _T_STR: ...
    def rjust(self, width: int, fillchar: str = " ") -> _T_STR: ...
    def zfill(self, width: int) -> _T_STR: ...
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> T: ...
    def slice_replace(
        self, start: int | None = None, stop: int | None = None, repl: str | None = None
    ) -> _T_STR: ...
    def decode(
        self, encoding: str, errors: str = "strict", dtype: str | DtypeObj | None = None
    ) -> _T_STR: ...
    def encode(self, encoding: str, errors: str = "strict") -> _T_BYTES: ...
    def strip(self, to_strip: str | None = None) -> _T_STR: ...
    def lstrip(self, to_strip: str | None = None) -> _T_STR: ...
    def rstrip(self, to_strip: str | None = None) -> _T_STR: ...
    def removeprefix(self, prefix: str) -> _T_STR: ...
    def removesuffix(self, suffix: str) -> _T_STR: ...
    def wrap(
        self,
        width: int,
        *,
        # kwargs passed to textwrap.TextWrapper
        expand_tabs: bool = True,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
    ) -> _T_STR: ...
    def get_dummies(self, sep: str = "|") -> _T_EXPANDING: ...
    def translate(self, table: Mapping[int, int | str | None] | None) -> _T_STR: ...
    def count(self, pat: str, flags: int = 0) -> _T_INT: ...
    def startswith(
        self, pat: str | tuple[str, ...], na: Scalar | NaTType | None = ...
    ) -> _T_BOOL: ...
    def endswith(
        self, pat: str | tuple[str, ...], na: Scalar | NaTType | None = ...
    ) -> _T_BOOL: ...
    def findall(self, pat: str | re.Pattern[str], flags: int = 0) -> _T_LIST_STR: ...
    @overload  # expand=True
    def extract(
        self, pat: str | re.Pattern[str], flags: int = 0, expand: Literal[True] = True
    ) -> pd.DataFrame: ...
    @overload  # expand=False (positional argument)
    def extract(
        self, pat: str | re.Pattern[str], flags: int, expand: Literal[False]
    ) -> _T_OBJECT: ...
    @overload  # expand=False (keyword argument)
    def extract(
        self, pat: str | re.Pattern[str], flags: int = 0, *, expand: Literal[False]
    ) -> _T_OBJECT: ...
    def extractall(
        self, pat: str | re.Pattern[str], flags: int = 0
    ) -> pd.DataFrame: ...
    def find(self, sub: str, start: int = 0, end: int | None = None) -> _T_INT: ...
    def rfind(self, sub: str, start: int = 0, end: int | None = None) -> _T_INT: ...
    def normalize(self, form: Literal["NFC", "NFKC", "NFD", "NFKD"]) -> _T_STR: ...
    def index(self, sub: str, start: int = 0, end: int | None = None) -> _T_INT: ...
    def rindex(self, sub: str, start: int = 0, end: int | None = None) -> _T_INT: ...
    def len(self) -> _T_INT: ...
    def lower(self) -> _T_STR: ...
    def upper(self) -> _T_STR: ...
    def title(self) -> _T_STR: ...
    def capitalize(self) -> _T_STR: ...
    def swapcase(self) -> _T_STR: ...
    def casefold(self) -> _T_STR: ...
    def isalnum(self) -> _T_BOOL: ...
    def isalpha(self) -> _T_BOOL: ...
    def isdigit(self) -> _T_BOOL: ...
    def isspace(self) -> _T_BOOL: ...
    def islower(self) -> _T_BOOL: ...
    def isupper(self) -> _T_BOOL: ...
    def istitle(self) -> _T_BOOL: ...
    def isnumeric(self) -> _T_BOOL: ...
    def isdecimal(self) -> _T_BOOL: ...
