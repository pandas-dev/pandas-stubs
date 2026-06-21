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
    TypeVar,
    overload,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
from pandas.core.base import NoNewAttributesMixin

from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from pandas._typing import (
    S2,
    AlignJoin,
    Dtype,
    DtypeObj,
    Scalar,
    T,
    np_1darray_bool,
    np_ndarray_str,
)

# Used for the result of str.split with expand=True
_T_EXPANDING = TypeVar("_T_EXPANDING", bound=DataFrame | MultiIndex)
# Used for the result of str.split with expand=False
_T_LIST_STR = TypeVar("_T_LIST_STR", bound=Series[list[str]] | Index[list[str]])
# Used for the result of str.match
_T_BOOL = TypeVar("_T_BOOL", bound=Series[bool] | np_1darray_bool)
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
    Generic[
        S2, T, _T_EXPANDING, _T_BOOL, _T_LIST_STR, _T_INT, _T_BYTES, _T_STR, _T_OBJECT
    ],
):
    def __init__(self, data: T) -> None: ...
    def __getitem__(self, key: _slice | int) -> _T_STR: ...
    def __iter__(self) -> Never: ...
    @overload
    def cat(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        others: None = None,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> str: ...
    @overload
    def cat(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        others: list[str] | np_ndarray_str | Series[str] | Index[str] | pd.DataFrame,
        sep: str | None = None,
        na_rep: str | None = None,
        join: AlignJoin = "left",
    ) -> _T_STR: ...
    @overload
    def split(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
        regex: bool | None = None,
    ) -> _T_EXPANDING: ...
    @overload
    def split(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str] | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
        regex: bool | None = None,
    ) -> _T_LIST_STR: ...
    @overload
    def rsplit(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | None = None,
        *,
        n: int = -1,
        expand: Literal[True],
    ) -> _T_EXPANDING: ...
    @overload
    def rsplit(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | None = None,
        *,
        n: int = -1,
        expand: Literal[False] = False,
    ) -> _T_LIST_STR: ...
    @overload  # expand=True
    def partition(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sep: str = " ",
        expand: Literal[True] = True,
    ) -> _T_EXPANDING: ...
    @overload  # expand=False (positional argument)
    def partition(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sep: str,
        expand: Literal[False],
    ) -> _T_OBJECT: ...
    @overload  # expand=False (keyword argument)
    def partition(self, sep: str = " ", *, expand: Literal[False]) -> _T_OBJECT: ...
    @overload  # expand=True
    def rpartition(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sep: str = " ",
        expand: Literal[True] = True,
    ) -> _T_EXPANDING: ...
    @overload  # expand=False (positional argument)
    def rpartition(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sep: str,
        expand: Literal[False],
    ) -> _T_OBJECT: ...
    @overload  # expand=False (keyword argument)
    def rpartition(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sep: str = " ",
        *,
        expand: Literal[False],
    ) -> _T_OBJECT: ...
    def get(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        i: int | Hashable,
    ) -> _T_STR: ...
    def join(self, sep: str) -> _T_STR: ...
    def contains(
        self: (
            StringMethods[
                float | str,
                T,
                _T_EXPANDING,
                _T_BOOL,
                _T_LIST_STR,
                _T_INT,
                _T_BYTES,
                _T_STR,
                _T_OBJECT,
            ]
            | StringMethods[
                str,
                T,
                _T_EXPANDING,
                _T_BOOL,
                _T_LIST_STR,
                _T_INT,
                _T_BYTES,
                _T_STR,
                _T_OBJECT,
            ]
        ),
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: float | NAType | NaTType | bool | None = ...,
        regex: bool = True,
    ) -> _T_BOOL: ...
    def match(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> _T_BOOL: ...
    def fullmatch(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | NaTType | None = ...,
    ) -> _T_BOOL: ...
    @overload
    def replace(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: dict[str, str],
        repl: None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> _T_STR: ...
    @overload
    def replace(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        repl: str | Callable[[re.Match[str]], str] | None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ) -> _T_STR: ...
    def repeat(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        repeats: int | Sequence[int],
    ) -> _T_STR: ...
    def pad(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> _T_STR: ...
    def center(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        width: int,
        fillchar: str = " ",
    ) -> _T_STR: ...
    def ljust(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        width: int,
        fillchar: str = " ",
    ) -> _T_STR: ...
    def rjust(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        width: int,
        fillchar: str = " ",
    ) -> _T_STR: ...
    def zfill(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        width: int,
    ) -> _T_STR: ...
    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> T: ...
    def slice_replace(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        start: int | None = None,
        stop: int | None = None,
        repl: str | None = None,
    ) -> _T_STR: ...
    def decode(
        self: StringMethods[
            bytes,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        encoding: str,
        errors: str = "strict",
        dtype: str | DtypeObj | None = None,
    ) -> _T_STR: ...
    def encode(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        encoding: str,
        errors: str = "strict",
    ) -> _T_BYTES: ...
    def strip(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        to_strip: str | None = None,
    ) -> _T_STR: ...
    def lstrip(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        to_strip: str | None = None,
    ) -> _T_STR: ...
    def rstrip(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        to_strip: str | None = None,
    ) -> _T_STR: ...
    def removeprefix(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        prefix: str,
    ) -> _T_STR: ...
    def removesuffix(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        suffix: str,
    ) -> _T_STR: ...
    def wrap(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        width: int,
        *,
        # kwargs passed to textwrap.TextWrapper
        expand_tabs: bool = True,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
    ) -> _T_STR: ...
    def get_dummies(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sep: str = "|",
        dtype: Dtype | None = None,
    ) -> _T_EXPANDING: ...
    def translate(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        table: Mapping[int, int | str | None] | None,
    ) -> _T_STR: ...
    def count(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str,
        flags: int = 0,
    ) -> _T_INT: ...
    def startswith(
        self: (
            StringMethods[
                float | str,
                T,
                _T_EXPANDING,
                _T_BOOL,
                _T_LIST_STR,
                _T_INT,
                _T_BYTES,
                _T_STR,
                _T_OBJECT,
            ]
            | StringMethods[
                str,
                T,
                _T_EXPANDING,
                _T_BOOL,
                _T_LIST_STR,
                _T_INT,
                _T_BYTES,
                _T_STR,
                _T_OBJECT,
            ]
        ),
        pat: str | tuple[str, ...],
        na: float | NAType | NaTType | bool | None = ...,
    ) -> _T_BOOL: ...
    def endswith(
        self: (
            StringMethods[
                float | str,
                T,
                _T_EXPANDING,
                _T_BOOL,
                _T_LIST_STR,
                _T_INT,
                _T_BYTES,
                _T_STR,
                _T_OBJECT,
            ]
            | StringMethods[
                str,
                T,
                _T_EXPANDING,
                _T_BOOL,
                _T_LIST_STR,
                _T_INT,
                _T_BYTES,
                _T_STR,
                _T_OBJECT,
            ]
        ),
        pat: str | tuple[str, ...],
        na: float | NAType | NaTType | bool | None = ...,
    ) -> _T_BOOL: ...
    def findall(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        flags: int = 0,
    ) -> _T_LIST_STR: ...
    @overload  # expand=True
    def extract(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        flags: int = 0,
        expand: Literal[True] = True,
    ) -> pd.DataFrame: ...
    @overload  # expand=False (positional argument)
    def extract(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        flags: int,
        expand: Literal[False],
    ) -> _T_OBJECT: ...
    @overload  # expand=False (keyword argument)
    def extract(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        flags: int = 0,
        *,
        expand: Literal[False],
    ) -> _T_OBJECT: ...
    def extractall(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        pat: str | re.Pattern[str],
        flags: int = 0,
    ) -> pd.DataFrame: ...
    def find(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sub: str,
        start: int = 0,
        end: int | None = None,
    ) -> _T_INT: ...
    def rfind(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sub: str,
        start: int = 0,
        end: int | None = None,
    ) -> _T_INT: ...
    def normalize(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        form: Literal["NFC", "NFKC", "NFD", "NFKD"],
    ) -> _T_STR: ...
    def index(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sub: str,
        start: int = 0,
        end: int | None = None,
    ) -> _T_INT: ...
    def rindex(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
        sub: str,
        start: int = 0,
        end: int | None = None,
    ) -> _T_INT: ...
    def len(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_INT: ...
    def lower(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_STR: ...
    def upper(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_STR: ...
    def title(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_STR: ...
    def capitalize(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_STR: ...
    def swapcase(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_STR: ...
    def casefold(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_STR: ...
    def isalnum(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isalpha(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isdigit(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isspace(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def islower(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isupper(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def istitle(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isnumeric(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isdecimal(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
    def isascii(
        self: StringMethods[
            str,
            T,
            _T_EXPANDING,
            _T_BOOL,
            _T_LIST_STR,
            _T_INT,
            _T_BYTES,
            _T_STR,
            _T_OBJECT,
        ],
    ) -> _T_BOOL: ...
