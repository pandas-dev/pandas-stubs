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
    Never,
    overload,
    type_check_only,
)

from pandas.core.base import NoNewAttributesMixin
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    S2,
    AlignJoin,
    Dtype,
    DtypeObj,
    Scalar,
    np_1darray_bool,
    np_ndarray_str,
)

class StringMethods(NoNewAttributesMixin, Generic[S2]):
    def __iter__(self) -> Never: ...

@type_check_only
class IndexStringMethods(StringMethods[S2]):
    def __getitem__(self, key: _slice | int) -> Index[str]: ...
    @overload
    def cat(
        self: IndexStringMethods[str],
        others: None = None,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> str: ...
    @overload
    def cat(
        self: IndexStringMethods[str],
        others: list[str] | np_ndarray_str | Index[str] | DataFrame,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> Index[str]: ...
    @overload
    def split(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
        regex: bool | None = None,
    ) -> MultiIndex: ...
    @overload
    def split(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
        regex: bool | None = None,
    ) -> Index[list[str]]: ...
    @overload
    def rsplit(
        self: IndexStringMethods[str],
        pat: str | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
    ) -> MultiIndex: ...
    @overload
    def rsplit(
        self: IndexStringMethods[str],
        pat: str | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
    ) -> Index[list[str]]: ...
    @overload  # expand=True
    def partition(
        self: IndexStringMethods[str], sep: str = " ", expand: Literal[True] = True
    ) -> MultiIndex: ...
    @overload  # expand=False (positional argument)
    def partition(
        self: IndexStringMethods[str], sep: str, expand: Literal[False]
    ) -> Index: ...
    @overload  # expand=False (keyword argument)
    def partition(self, sep: str = " ", *, expand: Literal[False]) -> Index: ...
    @overload  # expand=True
    def rpartition(
        self: IndexStringMethods[str], sep: str = " ", expand: Literal[True] = True
    ) -> MultiIndex: ...
    @overload  # expand=False (positional argument)
    def rpartition(
        self: IndexStringMethods[str], sep: str, expand: Literal[False]
    ) -> Index: ...
    @overload  # expand=False (keyword argument)
    def rpartition(
        self: IndexStringMethods[str], sep: str = " ", *, expand: Literal[False]
    ) -> Index: ...
    def get(self: IndexStringMethods[str], i: int | Hashable) -> Index[str]: ...
    def join(self, sep: str) -> Index[str]: ...
    def contains(
        self: IndexStringMethods[float | str] | IndexStringMethods[str],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: float | NAType | NaTType | bool | None = ...,
        regex: bool = True,
    ) -> np_1darray_bool: ...
    def match(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> np_1darray_bool: ...
    def fullmatch(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> np_1darray_bool: ...
    @overload
    def replace(
        self: IndexStringMethods[str],
        pat: dict[str, str],
        repl: None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Index[str]: ...
    @overload
    def replace(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str],
        repl: str | Callable[[re.Match[str]], str] | None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Index[str]: ...
    def repeat(
        self: IndexStringMethods[str], repeats: int | Sequence[int]
    ) -> Index[str]: ...
    def pad(
        self: IndexStringMethods[str],
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Index[str]: ...
    def center(
        self: IndexStringMethods[str], width: int, fillchar: str = " "
    ) -> Index[str]: ...
    def ljust(
        self: IndexStringMethods[str], width: int, fillchar: str = " "
    ) -> Index[str]: ...
    def rjust(
        self: IndexStringMethods[str], width: int, fillchar: str = " "
    ) -> Index[str]: ...
    def zfill(self: IndexStringMethods[str], width: int) -> Index[str]: ...
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Index[S2]: ...
    def slice_replace(
        self: IndexStringMethods[str],
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None,
    ) -> Index[str]: ...
    def decode(
        self: IndexStringMethods[bytes],
        encoding: str,
        errors: str = "strict",
        dtype: str | DtypeObj | None = None,
    ) -> Index[str]: ...
    def encode(
        self: IndexStringMethods[str], encoding: str, errors: str = "strict"
    ) -> Index[bytes]: ...
    def strip(
        self: IndexStringMethods[str], to_strip: str | None = None
    ) -> Index[str]: ...
    def lstrip(
        self: IndexStringMethods[str], to_strip: str | None = None
    ) -> Index[str]: ...
    def rstrip(
        self: IndexStringMethods[str], to_strip: str | None = None
    ) -> Index[str]: ...
    def removeprefix(self: IndexStringMethods[str], prefix: str) -> Index[str]: ...
    def removesuffix(self: IndexStringMethods[str], suffix: str) -> Index[str]: ...
    def wrap(
        self: IndexStringMethods[str],
        width: int,
        *,
        # kwargs passed to textwrap.TextWrapper
        expand_tabs: bool = True,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
    ) -> Index[str]: ...
    def get_dummies(
        self: IndexStringMethods[str], sep: str = "|", dtype: Dtype | None = None
    ) -> MultiIndex: ...
    def translate(
        self: IndexStringMethods[str], table: Mapping[int, int | str | None] | None
    ) -> Index[str]: ...
    def count(
        self: IndexStringMethods[str], pat: str, flags: int = 0
    ) -> Index[int]: ...
    def startswith(
        self: IndexStringMethods[float | str] | IndexStringMethods[str],
        pat: str | tuple[str, ...],
        na: float | NAType | NaTType | bool | None = ...,
    ) -> np_1darray_bool: ...
    def endswith(
        self: IndexStringMethods[float | str] | IndexStringMethods[str],
        pat: str | tuple[str, ...],
        na: float | NAType | NaTType | bool | None = ...,
    ) -> np_1darray_bool: ...
    def findall(
        self: IndexStringMethods[str], pat: str | re.Pattern[str], flags: int = 0
    ) -> Index[list[str]]: ...
    @overload  # expand=True
    def extract(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str],
        flags: int = 0,
        expand: Literal[True] = True,
    ) -> DataFrame: ...
    @overload  # expand=False (positional argument)
    def extract(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str],
        flags: int,
        expand: Literal[False],
    ) -> Index: ...
    @overload  # expand=False (keyword argument)
    def extract(
        self: IndexStringMethods[str],
        pat: str | re.Pattern[str],
        flags: int = 0,
        *,
        expand: Literal[False],
    ) -> Index: ...
    def extractall(
        self: IndexStringMethods[str], pat: str | re.Pattern[str], flags: int = 0
    ) -> DataFrame: ...
    def find(
        self: IndexStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Index[int]: ...
    def rfind(
        self: IndexStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Index[int]: ...
    def normalize(
        self: IndexStringMethods[str], form: Literal["NFC", "NFKC", "NFD", "NFKD"]
    ) -> Index[str]: ...
    def index(
        self: IndexStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Index[int]: ...
    def rindex(
        self: IndexStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Index[int]: ...
    def len(self: IndexStringMethods[str]) -> Index[int]: ...
    def lower(self: IndexStringMethods[str]) -> Index[str]: ...
    def upper(self: IndexStringMethods[str]) -> Index[str]: ...
    def title(self: IndexStringMethods[str]) -> Index[str]: ...
    def capitalize(self: IndexStringMethods[str]) -> Index[str]: ...
    def swapcase(self: IndexStringMethods[str]) -> Index[str]: ...
    def casefold(self: IndexStringMethods[str]) -> Index[str]: ...
    def isalnum(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isalpha(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isdigit(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isspace(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def islower(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isupper(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def istitle(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isnumeric(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isdecimal(self: IndexStringMethods[str]) -> np_1darray_bool: ...
    def isascii(self: IndexStringMethods[str]) -> np_1darray_bool: ...

@type_check_only
class SeriesStringMethods(StringMethods[S2]):
    def __getitem__(self, key: _slice | int) -> Series[str]: ...
    @overload
    def cat(
        self: SeriesStringMethods[str],
        others: None = None,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> str: ...
    @overload
    def cat(
        self: SeriesStringMethods[str],
        others: list[str] | np_ndarray_str | Series[str] | Index[str] | DataFrame,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> Series[str]: ...
    @overload
    def split(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
        regex: bool | None = None,
    ) -> DataFrame: ...
    @overload
    def split(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
        regex: bool | None = None,
    ) -> Series[list[str]]: ...
    @overload
    def rsplit(
        self: SeriesStringMethods[str],
        pat: str | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
    ) -> DataFrame: ...
    @overload
    def rsplit(
        self: SeriesStringMethods[str],
        pat: str | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
    ) -> Series[list[str]]: ...
    @overload  # expand=True
    def partition(
        self: SeriesStringMethods[str], sep: str = " ", expand: Literal[True] = True
    ) -> DataFrame: ...
    @overload  # expand=False (positional argument)
    def partition(
        self: SeriesStringMethods[str], sep: str, expand: Literal[False]
    ) -> Series: ...
    @overload  # expand=False (keyword argument)
    def partition(self, sep: str = " ", *, expand: Literal[False]) -> Series: ...
    @overload  # expand=True
    def rpartition(
        self: SeriesStringMethods[str], sep: str = " ", expand: Literal[True] = True
    ) -> DataFrame: ...
    @overload  # expand=False (positional argument)
    def rpartition(
        self: SeriesStringMethods[str], sep: str, expand: Literal[False]
    ) -> Series: ...
    @overload  # expand=False (keyword argument)
    def rpartition(
        self: SeriesStringMethods[str], sep: str = " ", *, expand: Literal[False]
    ) -> Series: ...
    def get(self: SeriesStringMethods[str], i: int | Hashable) -> Series[str]: ...
    def join(self, sep: str) -> Series[str]: ...
    def contains(
        self: SeriesStringMethods[float | str] | SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: float | NAType | NaTType | bool | None = ...,
        regex: bool = True,
    ) -> Series[bool]: ...
    def match(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> Series[bool]: ...
    def fullmatch(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> Series[bool]: ...
    @overload
    def replace(
        self: SeriesStringMethods[str],
        pat: dict[str, str],
        repl: None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Series[str]: ...
    @overload
    def replace(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        repl: str | Callable[[re.Match[str]], str] | None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> Series[str]: ...
    def repeat(
        self: SeriesStringMethods[str], repeats: int | Sequence[int]
    ) -> Series[str]: ...
    def pad(
        self: SeriesStringMethods[str],
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Series[str]: ...
    def center(
        self: SeriesStringMethods[str], width: int, fillchar: str = " "
    ) -> Series[str]: ...
    def ljust(
        self: SeriesStringMethods[str], width: int, fillchar: str = " "
    ) -> Series[str]: ...
    def rjust(
        self: SeriesStringMethods[str], width: int, fillchar: str = " "
    ) -> Series[str]: ...
    def zfill(self: SeriesStringMethods[str], width: int) -> Series[str]: ...
    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Series[S2]: ...
    def slice_replace(
        self: SeriesStringMethods[str],
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None,
    ) -> Series[str]: ...
    def decode(
        self: SeriesStringMethods[bytes],
        encoding: str,
        errors: str = "strict",
        dtype: str | DtypeObj | None = None,
    ) -> Series[str]: ...
    def encode(
        self: SeriesStringMethods[str], encoding: str, errors: str = "strict"
    ) -> Series[bytes]: ...
    def strip(
        self: SeriesStringMethods[str], to_strip: str | None = None
    ) -> Series[str]: ...
    def lstrip(
        self: SeriesStringMethods[str], to_strip: str | None = None
    ) -> Series[str]: ...
    def rstrip(
        self: SeriesStringMethods[str], to_strip: str | None = None
    ) -> Series[str]: ...
    def removeprefix(self: SeriesStringMethods[str], prefix: str) -> Series[str]: ...
    def removesuffix(self: SeriesStringMethods[str], suffix: str) -> Series[str]: ...
    def wrap(
        self: SeriesStringMethods[str],
        width: int,
        *,
        # kwargs passed to textwrap.TextWrapper
        expand_tabs: bool = True,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
    ) -> Series[str]: ...
    def get_dummies(
        self: SeriesStringMethods[str], sep: str = "|", dtype: Dtype | None = None
    ) -> DataFrame: ...
    def translate(
        self: SeriesStringMethods[str], table: Mapping[int, int | str | None] | None
    ) -> Series[str]: ...
    def count(
        self: SeriesStringMethods[str], pat: str, flags: int = 0
    ) -> Series[int]: ...
    def startswith(
        self: SeriesStringMethods[float | str] | SeriesStringMethods[str],
        pat: str | tuple[str, ...],
        na: float | NAType | NaTType | bool | None = ...,
    ) -> Series[bool]: ...
    def endswith(
        self: SeriesStringMethods[float | str] | SeriesStringMethods[str],
        pat: str | tuple[str, ...],
        na: float | NAType | NaTType | bool | None = ...,
    ) -> Series[bool]: ...
    def findall(
        self: SeriesStringMethods[str], pat: str | re.Pattern[str], flags: int = 0
    ) -> Series[list[str]]: ...
    @overload  # expand=True
    def extract(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        flags: int = 0,
        expand: Literal[True] = True,
    ) -> DataFrame: ...
    @overload  # expand=False (positional argument)
    def extract(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        flags: int,
        expand: Literal[False],
    ) -> Series: ...
    @overload  # expand=False (keyword argument)
    def extract(
        self: SeriesStringMethods[str],
        pat: str | re.Pattern[str],
        flags: int = 0,
        *,
        expand: Literal[False],
    ) -> Series: ...
    def extractall(
        self: SeriesStringMethods[str], pat: str | re.Pattern[str], flags: int = 0
    ) -> DataFrame: ...
    def find(
        self: SeriesStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Series[int]: ...
    def rfind(
        self: SeriesStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Series[int]: ...
    def normalize(
        self: SeriesStringMethods[str], form: Literal["NFC", "NFKC", "NFD", "NFKD"]
    ) -> Series[str]: ...
    def index(
        self: SeriesStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Series[int]: ...
    def rindex(
        self: SeriesStringMethods[str], sub: str, start: int = 0, end: int | None = None
    ) -> Series[int]: ...
    def len(self: SeriesStringMethods[str]) -> Series[int]: ...
    def lower(self: SeriesStringMethods[str]) -> Series[str]: ...
    def upper(self: SeriesStringMethods[str]) -> Series[str]: ...
    def title(self: SeriesStringMethods[str]) -> Series[str]: ...
    def capitalize(self: SeriesStringMethods[str]) -> Series[str]: ...
    def swapcase(self: SeriesStringMethods[str]) -> Series[str]: ...
    def casefold(self: SeriesStringMethods[str]) -> Series[str]: ...
    def isalnum(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isalpha(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isdigit(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isspace(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def islower(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isupper(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def istitle(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isnumeric(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isdecimal(self: SeriesStringMethods[str]) -> Series[bool]: ...
    def isascii(self: SeriesStringMethods[str]) -> Series[bool]: ...

@type_check_only
class StrDescriptor:
    @overload
    def __get__(
        self, instance: Series[S2], owner: type[Series]
    ) -> SeriesStringMethods[S2]: ...
    @overload
    def __get__(
        self, instance: Index[S2], owner: type[Index]
    ) -> IndexStringMethods[S2]: ...
