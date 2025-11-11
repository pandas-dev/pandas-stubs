import functools
import re

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    np_1darray_bool,
)

DATA = ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
DATA_BYTES = [b"applep", b"bananap"]


def test_string_accessors_type_preserving_series() -> None:
    s_str = pd.Series(DATA)
    s_bytes = pd.Series(DATA_BYTES)
    check(assert_type(s_str.str.slice(0, 4, 2), "pd.Series[str]"), pd.Series, str)
    check(assert_type(s_bytes.str.slice(0, 4, 2), "pd.Series[bytes]"), pd.Series, bytes)


def test_string_accessors_type_preserving_index() -> None:
    idx_str = pd.Index(DATA)
    idx_bytes = pd.Index(DATA_BYTES)
    check(assert_type(idx_str.str.slice(0, 4, 2), "pd.Index[str]"), pd.Index, str)
    check(assert_type(idx_bytes.str.slice(0, 4, 2), "pd.Index[bytes]"), pd.Index, bytes)


def test_string_accessors_boolean_series() -> None:
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=np.bool_)
    _check(assert_type(s.str.startswith("a"), "pd.Series[bool]"))
    _check(
        assert_type(s.str.startswith(("a", "b")), "pd.Series[bool]"),
    )
    _check(
        assert_type(s.str.contains("a"), "pd.Series[bool]"),
    )
    _check(assert_type(s.str.contains(re.compile(r"a"), regex=True), "pd.Series[bool]"))
    _check(assert_type(s.str.endswith("e"), "pd.Series[bool]"))
    _check(assert_type(s.str.endswith(("e", "f")), "pd.Series[bool]"))
    _check(assert_type(s.str.fullmatch("apple"), "pd.Series[bool]"))
    _check(assert_type(s.str.fullmatch(re.compile(r"apple")), "pd.Series[bool]"))
    _check(assert_type(s.str.isalnum(), "pd.Series[bool]"))
    _check(assert_type(s.str.isalpha(), "pd.Series[bool]"))
    _check(assert_type(s.str.isdecimal(), "pd.Series[bool]"))
    _check(assert_type(s.str.isdigit(), "pd.Series[bool]"))
    _check(assert_type(s.str.isnumeric(), "pd.Series[bool]"))
    _check(assert_type(s.str.islower(), "pd.Series[bool]"))
    _check(assert_type(s.str.isspace(), "pd.Series[bool]"))
    _check(assert_type(s.str.istitle(), "pd.Series[bool]"))
    _check(assert_type(s.str.isupper(), "pd.Series[bool]"))
    _check(assert_type(s.str.match("pp"), "pd.Series[bool]"))
    _check(assert_type(s.str.match(re.compile(r"pp")), "pd.Series[bool]"))


def test_string_accessors_boolean_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=np_1darray_bool)
    _check(assert_type(idx.str.startswith("a"), np_1darray_bool))
    _check(
        assert_type(idx.str.startswith(("a", "b")), np_1darray_bool),
    )
    _check(assert_type(idx.str.contains("a"), np_1darray_bool))
    _check(assert_type(idx.str.contains(re.compile(r"a"), regex=True), np_1darray_bool))
    _check(assert_type(idx.str.endswith("e"), np_1darray_bool))
    _check(assert_type(idx.str.endswith(("e", "f")), np_1darray_bool))
    _check(assert_type(idx.str.fullmatch("apple"), np_1darray_bool))
    _check(assert_type(idx.str.fullmatch(re.compile(r"apple")), np_1darray_bool))
    _check(assert_type(idx.str.isalnum(), np_1darray_bool))
    _check(assert_type(idx.str.isalpha(), np_1darray_bool))
    _check(assert_type(idx.str.isdecimal(), np_1darray_bool))
    _check(assert_type(idx.str.isdigit(), np_1darray_bool))
    _check(assert_type(idx.str.isnumeric(), np_1darray_bool))
    _check(assert_type(idx.str.islower(), np_1darray_bool))
    _check(assert_type(idx.str.isspace(), np_1darray_bool))
    _check(assert_type(idx.str.istitle(), np_1darray_bool))
    _check(assert_type(idx.str.isupper(), np_1darray_bool))
    _check(assert_type(idx.str.match("pp"), np_1darray_bool))
    _check(assert_type(idx.str.match(re.compile(r"pp")), np_1darray_bool))


def test_string_accessors_integer_series() -> None:
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=np.integer)
    _check(assert_type(s.str.find("p"), "pd.Series[int]"))
    _check(assert_type(s.str.index("p"), "pd.Series[int]"))
    _check(assert_type(s.str.rfind("e"), "pd.Series[int]"))
    _check(assert_type(s.str.rindex("p"), "pd.Series[int]"))
    _check(assert_type(s.str.count("pp"), "pd.Series[int]"))
    _check(assert_type(s.str.len(), "pd.Series[int]"))

    # unlike findall, find doesn't accept a compiled pattern
    with pytest.raises(TypeError):
        s.str.find(re.compile(r"p"))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_string_accessors_integer_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=pd.Index, dtype=np.integer)
    _check(assert_type(idx.str.find("p"), "pd.Index[int]"))
    _check(assert_type(idx.str.index("p"), "pd.Index[int]"))
    _check(assert_type(idx.str.rfind("e"), "pd.Index[int]"))
    _check(assert_type(idx.str.rindex("p"), "pd.Index[int]"))
    _check(assert_type(idx.str.count("pp"), "pd.Index[int]"))
    _check(assert_type(idx.str.len(), "pd.Index[int]"))

    # unlike findall, find doesn't accept a compiled pattern
    with pytest.raises(TypeError):
        idx.str.find(re.compile(r"p"))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_string_accessors_string_series() -> None:
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=str)
    _check(assert_type(s.str.capitalize(), "pd.Series[str]"))
    _check(assert_type(s.str.casefold(), "pd.Series[str]"))
    check(assert_type(s.str.cat(sep="X"), str), str)
    _check(assert_type(s.str.center(10), "pd.Series[str]"))
    _check(assert_type(s.str.get(2), "pd.Series[str]"))
    s_dict = pd.Series(  # example from the doc of str.get
        [{"name": "Hello", "value": "World"}, {"name": "Goodbye", "value": "Planet"}]
    )
    _check(assert_type(s_dict.str.get("name"), "pd.Series[str]"))
    _check(assert_type(s.str.ljust(80), "pd.Series[str]"))
    _check(assert_type(s.str.lower(), "pd.Series[str]"))
    _check(assert_type(s.str.lstrip("a"), "pd.Series[str]"))
    _check(assert_type(s.str.normalize("NFD"), "pd.Series[str]"))
    _check(assert_type(s.str.pad(80, "right"), "pd.Series[str]"))
    _check(assert_type(s.str.removeprefix("a"), "pd.Series[str]"))
    _check(assert_type(s.str.removesuffix("e"), "pd.Series[str]"))
    _check(assert_type(s.str.repeat(2), "pd.Series[str]"))
    _check(assert_type(s.str.replace("a", "X"), "pd.Series[str]"))
    _check(
        assert_type(s.str.replace(re.compile(r"a"), "X", regex=True), "pd.Series[str]")
    )
    _check(assert_type(s.str.rjust(80), "pd.Series[str]"))
    _check(assert_type(s.str.rstrip(), "pd.Series[str]"))
    _check(assert_type(s.str.slice_replace(0, 2, "XX"), "pd.Series[str]"))
    _check(assert_type(s.str.strip(), "pd.Series[str]"))
    _check(assert_type(s.str.swapcase(), "pd.Series[str]"))
    _check(assert_type(s.str.title(), "pd.Series[str]"))
    _check(
        assert_type(s.str.translate({241: "n"}), "pd.Series[str]"),
    )
    _check(
        assert_type(s.str.translate({241: 240}), "pd.Series[str]"),
    )
    trans_table: dict[int, int] = {ord("a"): ord("b")}
    _check(  # tests covariance of table values (table is read-only)
        assert_type(s.str.translate(trans_table), "pd.Series[str]"),
    )
    _check(assert_type(s.str.upper(), "pd.Series[str]"))
    _check(assert_type(s.str.wrap(80), "pd.Series[str]"))
    _check(assert_type(s.str.zfill(10), "pd.Series[str]"))
    s_bytes = pd.Series([b"a1", b"b2", b"c3"])
    _check(assert_type(s_bytes.str.decode("utf-8"), "pd.Series[str]"))
    _check(
        assert_type(
            s_bytes.str.decode("utf-8", dtype=pd.StringDtype()), "pd.Series[str]"
        )
    )
    s_list = pd.Series([["apple", "banana"], ["cherry", "date"], ["one", "eggplant"]])
    _check(assert_type(s_list.str.join("-"), "pd.Series[str]"))

    # wrap doesn't accept positional arguments other than width
    if TYPE_CHECKING_INVALID_USAGE:
        s.str.wrap(80, False)  # type: ignore[misc] # pyright: ignore[reportCallIssue]


def test_string_accessors_string_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=pd.Index, dtype=str)
    _check(assert_type(idx.str.capitalize(), "pd.Index[str]"))
    _check(assert_type(idx.str.casefold(), "pd.Index[str]"))
    check(assert_type(idx.str.cat(sep="X"), str), str)
    _check(assert_type(idx.str.center(10), "pd.Index[str]"))
    _check(assert_type(idx.str.get(2), "pd.Index[str]"))
    idx_dict = pd.Index(
        [{"name": "Hello", "value": "World"}, {"name": "Goodbye", "value": "Planet"}]
    )
    _check(assert_type(idx_dict.str.get("name"), "pd.Index[str]"))
    _check(assert_type(idx.str.ljust(80), "pd.Index[str]"))
    _check(assert_type(idx.str.lower(), "pd.Index[str]"))
    _check(assert_type(idx.str.lstrip("a"), "pd.Index[str]"))
    _check(assert_type(idx.str.normalize("NFD"), "pd.Index[str]"))
    _check(assert_type(idx.str.pad(80, "right"), "pd.Index[str]"))
    _check(assert_type(idx.str.removeprefix("a"), "pd.Index[str]"))
    _check(assert_type(idx.str.removesuffix("e"), "pd.Index[str]"))
    _check(assert_type(idx.str.repeat(2), "pd.Index[str]"))
    _check(assert_type(idx.str.replace("a", "X"), "pd.Index[str]"))
    _check(
        assert_type(idx.str.replace(re.compile(r"a"), "X", regex=True), "pd.Index[str]")
    )
    _check(assert_type(idx.str.rjust(80), "pd.Index[str]"))
    _check(assert_type(idx.str.rstrip(), "pd.Index[str]"))
    _check(assert_type(idx.str.slice_replace(0, 2, "XX"), "pd.Index[str]"))
    _check(assert_type(idx.str.strip(), "pd.Index[str]"))
    _check(assert_type(idx.str.swapcase(), "pd.Index[str]"))
    _check(assert_type(idx.str.title(), "pd.Index[str]"))
    _check(
        assert_type(idx.str.translate({241: "n"}), "pd.Index[str]"),
    )
    _check(
        assert_type(idx.str.translate({241: 240}), "pd.Index[str]"),
    )
    trans_table: dict[int, int] = {ord("a"): ord("b")}
    _check(  # tests covariance of table values (table is read-only)
        assert_type(idx.str.translate(trans_table), "pd.Index[str]"),
    )
    _check(assert_type(idx.str.upper(), "pd.Index[str]"))
    _check(assert_type(idx.str.wrap(80), "pd.Index[str]"))
    _check(assert_type(idx.str.zfill(10), "pd.Index[str]"))
    idx_bytes = pd.Index([b"a1", b"b2", b"c3"])
    _check(assert_type(idx_bytes.str.decode("utf-8"), "pd.Index[str]"))
    _check(
        assert_type(
            idx_bytes.str.decode("utf-8", dtype=pd.StringDtype()), "pd.Index[str]"
        )
    )
    idx_list = pd.Index([["apple", "banana"], ["cherry", "date"], ["one", "eggplant"]])
    _check(assert_type(idx_list.str.join("-"), "pd.Index[str]"))

    # wrap doesn't accept positional arguments other than width
    if TYPE_CHECKING_INVALID_USAGE:
        idx.str.wrap(80, False)  # type: ignore[misc] # pyright: ignore[reportCallIssue]


def test_string_accessors_bytes_series() -> None:
    s = pd.Series(["a1", "b2", "c3"])
    check(assert_type(s.str.encode("latin-1"), "pd.Series[bytes]"), pd.Series, bytes)


def test_string_accessors_bytes_index() -> None:
    s = pd.Index(["a1", "b2", "c3"])
    check(assert_type(s.str.encode("latin-1"), "pd.Index[bytes]"), pd.Index, bytes)


def test_string_accessors_list_series() -> None:
    s = pd.Series(DATA)
    _check = functools.partial(check, klass=pd.Series, dtype=list)
    _check(assert_type(s.str.findall("pp"), "pd.Series[list[str]]"))
    _check(assert_type(s.str.findall(re.compile(r"pp")), "pd.Series[list[str]]"))
    _check(assert_type(s.str.split("a"), "pd.Series[list[str]]"))
    _check(assert_type(s.str.split(re.compile(r"a")), "pd.Series[list[str]]"))
    # GH 194
    _check(assert_type(s.str.split("a", expand=False), "pd.Series[list[str]]"))
    _check(assert_type(s.str.rsplit("a"), "pd.Series[list[str]]"))
    _check(assert_type(s.str.rsplit("a", expand=False), "pd.Series[list[str]]"))

    # rsplit doesn't accept compiled pattern
    # it doesn't raise at runtime but produces a nan
    if TYPE_CHECKING_INVALID_USAGE:
        _bad_rsplit_result = s.str.rsplit(
            re.compile(r"a")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        )


def test_string_accessors_list_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=pd.Index, dtype=list)
    _check(assert_type(idx.str.findall("pp"), "pd.Index[list[str]]"))
    _check(assert_type(idx.str.findall(re.compile(r"pp")), "pd.Index[list[str]]"))
    _check(assert_type(idx.str.split("a"), "pd.Index[list[str]]"))
    _check(assert_type(idx.str.split(re.compile(r"a")), "pd.Index[list[str]]"))
    # GH 194
    _check(assert_type(idx.str.split("a", expand=False), "pd.Index[list[str]]"))
    _check(assert_type(idx.str.rsplit("a"), "pd.Index[list[str]]"))
    _check(assert_type(idx.str.rsplit("a", expand=False), "pd.Index[list[str]]"))

    # rsplit doesn't accept compiled pattern
    # it doesn't raise at runtime but produces a nan
    if TYPE_CHECKING_INVALID_USAGE:
        _bad_rsplit_result = idx.str.rsplit(
            re.compile(r"a")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        )


def test_string_accessors_expanding_series() -> None:
    s = pd.Series(["a1", "b2", "c3"])
    _check = functools.partial(check, klass=pd.DataFrame)
    _check(assert_type(s.str.extract(r"([ab])?(\d)"), pd.DataFrame))
    _check(assert_type(s.str.extract(re.compile(r"([ab])?(\d)")), pd.DataFrame))
    _check(assert_type(s.str.extractall(r"([ab])?(\d)"), pd.DataFrame))
    _check(assert_type(s.str.extractall(re.compile(r"([ab])?(\d)")), pd.DataFrame))
    _check(assert_type(s.str.get_dummies(), pd.DataFrame))
    _check(assert_type(s.str.partition("p"), pd.DataFrame))
    _check(assert_type(s.str.rpartition("p"), pd.DataFrame))
    _check(assert_type(s.str.rsplit("a", expand=True), pd.DataFrame))
    _check(assert_type(s.str.split("a", expand=True), pd.DataFrame))


def test_string_accessors_expanding_index() -> None:
    idx = pd.Index(["a1", "b2", "c3"])
    _check = functools.partial(check, klass=pd.MultiIndex)
    _check(assert_type(idx.str.get_dummies(), pd.MultiIndex))
    _check(assert_type(idx.str.partition("p"), pd.MultiIndex))
    _check(assert_type(idx.str.rpartition("p"), pd.MultiIndex))
    _check(assert_type(idx.str.rsplit("a", expand=True), pd.MultiIndex))
    _check(assert_type(idx.str.split("a", expand=True), pd.MultiIndex))

    # These ones are the odd ones out?
    check(assert_type(idx.str.extractall(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(idx.str.extractall(re.compile(r"([ab])?(\d)")), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(idx.str.extract(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(idx.str.extract(re.compile(r"([ab])?(\d)")), pd.DataFrame),
        pd.DataFrame,
    )


def test_series_overloads_partition() -> None:
    s = pd.Series(
        [
            "ap;pl;ep",
            "ban;an;ap",
            "Che;rr;yp",
            "DA;TEp",
            "eGGp;LANT;p",
            "12;3p",
            "23.45p",
        ]
    )
    check(assert_type(s.str.partition(sep=";"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.str.partition(sep=";", expand=True), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(s.str.partition(sep=";", expand=False), pd.Series),
        pd.Series,
        object,
    )
    check(
        assert_type(s.str.partition(expand=False), pd.Series),
        pd.Series,
        object,
    )

    check(assert_type(s.str.rpartition(sep=";"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.str.rpartition(sep=";", expand=True), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(s.str.rpartition(sep=";", expand=False), pd.Series),
        pd.Series,
        object,
    )
    check(assert_type(s.str.rpartition(expand=False), pd.Series), pd.Series, object)


def test_index_overloads_partition() -> None:
    idx = pd.Index(
        [
            "ap;pl;ep",
            "ban;an;ap",
            "Che;rr;yp",
            "DA;TEp",
            "eGGp;LANT;p",
            "12;3p",
            "23.45p",
        ]
    )
    check(assert_type(idx.str.partition(sep=";"), pd.MultiIndex), pd.MultiIndex)
    check(
        assert_type(idx.str.partition(sep=";", expand=True), pd.MultiIndex),
        pd.MultiIndex,
    )
    check(
        assert_type(idx.str.partition(sep=";", expand=False), pd.Index),
        pd.Index,
        object,
    )

    check(assert_type(idx.str.rpartition(sep=";"), pd.MultiIndex), pd.MultiIndex)
    check(
        assert_type(idx.str.rpartition(sep=";", expand=True), pd.MultiIndex),
        pd.MultiIndex,
    )
    check(
        assert_type(idx.str.rpartition(sep=";", expand=False), pd.Index),
        pd.Index,
        object,
    )


def test_series_overloads_cat() -> None:
    s = pd.Series(DATA)
    check(assert_type(s.str.cat(sep=";"), str), str)
    check(assert_type(s.str.cat(None, sep=";"), str), str)
    check(
        assert_type(
            s.str.cat(["A", "B", "C", "D", "E", "F", "G"], sep=";"),
            "pd.Series[str]",
        ),
        pd.Series,
        str,
    )
    check(
        assert_type(
            s.str.cat(pd.Series(["A", "B", "C", "D", "E", "F", "G"]), sep=";"),
            "pd.Series[str]",
        ),
        pd.Series,
        str,
    )
    unknown_s = pd.DataFrame({"a": list("abcdefg")})["a"]
    check(assert_type(s.str.cat(unknown_s, sep=";"), "pd.Series[str]"), pd.Series, str)
    check(assert_type(unknown_s.str.cat(s, sep=";"), "pd.Series[str]"), pd.Series, str)
    check(
        assert_type(unknown_s.str.cat(unknown_s, sep=";"), "pd.Series[str]"),
        pd.Series,
        str,
    )


def test_index_overloads_cat() -> None:
    idx = pd.Index(DATA)
    check(assert_type(idx.str.cat(sep=";"), str), str)
    check(assert_type(idx.str.cat(None, sep=";"), str), str)
    check(
        assert_type(
            idx.str.cat(["A", "B", "C", "D", "E", "F", "G"], sep=";"),
            "pd.Index[str]",
        ),
        pd.Index,
        str,
    )
    check(
        assert_type(
            idx.str.cat(pd.Index(["A", "B", "C", "D", "E", "F", "G"]), sep=";"),
            "pd.Index[str]",
        ),
        pd.Index,
        str,
    )
    unknown_idx = pd.DataFrame({"a": list("abcdefg")}).set_index("a").index
    check(
        assert_type(idx.str.cat(unknown_idx, sep=";"), "pd.Index[str]"), pd.Index, str
    )
    check(
        assert_type(unknown_idx.str.cat(idx, sep=";"), "pd.Index[str]"), pd.Index, str
    )
    check(
        assert_type(unknown_idx.str.cat(unknown_idx, sep=";"), "pd.Index[str]"),
        pd.Index,
        str,
    )


def test_series_overloads_extract() -> None:
    s = pd.Series(DATA)
    check(assert_type(s.str.extract(r"[ab](\d)"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.str.extract(r"[ab](\d)", expand=True), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(s.str.extract(r"[ab](\d)", expand=False), pd.Series),
        pd.Series,
        object,
    )
    check(
        assert_type(s.str.extract(r"[ab](\d)", re.IGNORECASE, False), pd.Series),
        pd.Series,
        object,
    )


def test_index_overloads_extract() -> None:
    idx = pd.Index(DATA)
    check(assert_type(idx.str.extract(r"[ab](\d)"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(idx.str.extract(r"[ab](\d)", expand=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(idx.str.extract(r"[ab](\d)", expand=False), pd.Index),
        pd.Index,
        object,
    )
    check(
        assert_type(idx.str.extract(r"[ab](\d)", re.IGNORECASE, False), pd.Index),
        pd.Index,
        object,
    )
