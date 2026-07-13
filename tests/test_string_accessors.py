import functools
import re
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.core.strings.accessor import StringMethods
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import np_1darray_bool

DATA = ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
DATA_BYTES = [b"applep", b"bananap"]


def test_accessors_isinstance() -> None:
    """Test that Series.str and Index.str supertype is `StringMethods`."""
    s_str = pd.Series(DATA)
    i_str = pd.Series(DATA)
    assert isinstance(s_str.str, StringMethods)
    assert isinstance(i_str.str, StringMethods)


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
    _check(assert_type(s.str.startswith(("a", "b")), "pd.Series[bool]"))
    _check(assert_type(s.str.contains("a"), "pd.Series[bool]"))
    _check(assert_type(s.str.contains(re.compile(r"a"), regex=True), "pd.Series[bool]"))
    _check(assert_type(s.str.endswith("e"), "pd.Series[bool]"))
    _check(assert_type(s.str.endswith(("e", "f")), "pd.Series[bool]"))
    _check(assert_type(s.str.fullmatch("apple"), "pd.Series[bool]"))
    _check(assert_type(s.str.fullmatch(re.compile(r"apple")), "pd.Series[bool]"))
    _check(assert_type(s.str.isalnum(), "pd.Series[bool]"))
    _check(assert_type(s.str.isascii(), "pd.Series[bool]"))
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

    sr_a = pd.DataFrame({"a": DATA})["a"]

    _check(assert_type(sr_a.str.startswith("a"), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.startswith(("a", "b")), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.contains("a"), "pd.Series[bool]"))
    _check(
        assert_type(sr_a.str.contains(re.compile(r"a"), regex=True), "pd.Series[bool]")
    )
    _check(assert_type(sr_a.str.endswith("e"), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.endswith(("e", "f")), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.fullmatch("apple"), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.fullmatch(re.compile(r"apple")), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isalnum(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isascii(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isalpha(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isdecimal(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isdigit(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isnumeric(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.islower(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isspace(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.istitle(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.isupper(), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.match("pp"), "pd.Series[bool]"))
    _check(assert_type(sr_a.str.match(re.compile(r"pp")), "pd.Series[bool]"))

    # test deprecated allowing non-bool values for na in .str.contains, str.startswith, and .str.endswith
    sr = pd.Series(["om", np.nan, "foo_nom", "nom", "bar_foo", np.nan, "foo"])

    # only None, pd.NA, np.nan, True, or False are allowed
    _check(assert_type(sr.str.startswith("kapow", na=None), "pd.Series[bool]"))
    _check(assert_type(sr.str.startswith("kapow", na=pd.NA), "pd.Series[bool]"))
    _check(assert_type(sr.str.startswith("kapow", na=np.nan), "pd.Series[bool]"))
    _check(assert_type(sr.str.startswith("kapow", na=True), "pd.Series[bool]"))
    _check(assert_type(sr.str.startswith("kapow", na=False), "pd.Series[bool]"))

    _check(assert_type(sr.str.endswith("kapow", na=None), "pd.Series[bool]"))
    _check(assert_type(sr.str.endswith("kapow", na=pd.NA), "pd.Series[bool]"))
    _check(assert_type(sr.str.endswith("kapow", na=np.nan), "pd.Series[bool]"))
    _check(assert_type(sr.str.endswith("kapow", na=True), "pd.Series[bool]"))
    _check(assert_type(sr.str.endswith("kapow", na=False), "pd.Series[bool]"))

    _check(assert_type(sr.str.contains("kapow", na=None), "pd.Series[bool]"))
    _check(assert_type(sr.str.contains("kapow", na=pd.NA), "pd.Series[bool]"))
    _check(assert_type(sr.str.contains("kapow", na=np.nan), "pd.Series[bool]"))
    _check(assert_type(sr.str.contains("kapow", na=True), "pd.Series[bool]"))
    _check(assert_type(sr.str.contains("kapow", na=False), "pd.Series[bool]"))

    if TYPE_CHECKING_INVALID_USAGE:
        sr.str.startswith("kapow", na="baz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]
        sr.str.endswith("kapow", na="baz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]
        sr.str.contains("kapow", na="baz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]

        sr_i = pd.Series([1])

        sr_i.str.startswith("kapow")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.endswith("kapow")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.contains("kapow")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]


def test_string_accessors_boolean_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=np_1darray_bool)

    _check(assert_type(idx.str.startswith("a"), np_1darray_bool))
    _check(assert_type(idx.str.startswith(("a", "b")), np_1darray_bool))
    _check(assert_type(idx.str.contains("a"), np_1darray_bool))
    _check(assert_type(idx.str.contains(re.compile(r"a"), regex=True), np_1darray_bool))
    _check(assert_type(idx.str.endswith("e"), np_1darray_bool))
    _check(assert_type(idx.str.endswith(("e", "f")), np_1darray_bool))
    _check(assert_type(idx.str.fullmatch("apple"), np_1darray_bool))
    _check(assert_type(idx.str.fullmatch(re.compile(r"apple")), np_1darray_bool))
    _check(assert_type(idx.str.isalnum(), np_1darray_bool))
    _check(assert_type(idx.str.isascii(), np_1darray_bool))
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

    idx_a = pd.MultiIndex(levels=[DATA], codes=[[0]]).levels[0]
    _check(assert_type(idx_a.str.startswith("a"), np_1darray_bool))
    _check(assert_type(idx_a.str.startswith(("a", "b")), np_1darray_bool))
    _check(assert_type(idx_a.str.contains("a"), np_1darray_bool))
    _check(
        assert_type(idx_a.str.contains(re.compile(r"a"), regex=True), np_1darray_bool)
    )
    _check(assert_type(idx_a.str.endswith("e"), np_1darray_bool))
    _check(assert_type(idx_a.str.endswith(("e", "f")), np_1darray_bool))
    _check(assert_type(idx_a.str.fullmatch("apple"), np_1darray_bool))
    _check(assert_type(idx_a.str.fullmatch(re.compile(r"apple")), np_1darray_bool))
    _check(assert_type(idx_a.str.isalnum(), np_1darray_bool))
    _check(assert_type(idx_a.str.isascii(), np_1darray_bool))
    _check(assert_type(idx_a.str.isalpha(), np_1darray_bool))
    _check(assert_type(idx_a.str.isdecimal(), np_1darray_bool))
    _check(assert_type(idx_a.str.isdigit(), np_1darray_bool))
    _check(assert_type(idx_a.str.isnumeric(), np_1darray_bool))
    _check(assert_type(idx_a.str.islower(), np_1darray_bool))
    _check(assert_type(idx_a.str.isspace(), np_1darray_bool))
    _check(assert_type(idx_a.str.istitle(), np_1darray_bool))
    _check(assert_type(idx_a.str.isupper(), np_1darray_bool))
    _check(assert_type(idx_a.str.match("pp"), np_1darray_bool))
    _check(assert_type(idx_a.str.match(re.compile(r"pp")), np_1darray_bool))

    # only None, pd.NA, np.nan, True, or False are allowed
    _check(assert_type(idx.str.startswith("kapow", na=None), np_1darray_bool))
    _check(assert_type(idx.str.startswith("kapow", na=pd.NA), np_1darray_bool))
    _check(assert_type(idx.str.startswith("kapow", na=np.nan), np_1darray_bool))
    _check(assert_type(idx.str.startswith("kapow", na=True), np_1darray_bool))
    _check(assert_type(idx.str.startswith("kapow", na=False), np_1darray_bool))

    _check(assert_type(idx.str.endswith("kapow", na=None), np_1darray_bool))
    _check(assert_type(idx.str.endswith("kapow", na=pd.NA), np_1darray_bool))
    _check(assert_type(idx.str.endswith("kapow", na=np.nan), np_1darray_bool))
    _check(assert_type(idx.str.endswith("kapow", na=True), np_1darray_bool))
    _check(assert_type(idx.str.endswith("kapow", na=False), np_1darray_bool))

    _check(assert_type(idx.str.contains("kapow", na=None), np_1darray_bool))
    _check(assert_type(idx.str.contains("kapow", na=pd.NA), np_1darray_bool))
    _check(assert_type(idx.str.contains("kapow", na=np.nan), np_1darray_bool))
    _check(assert_type(idx.str.contains("kapow", na=True), np_1darray_bool))
    _check(assert_type(idx.str.contains("kapow", na=False), np_1darray_bool))

    if TYPE_CHECKING_INVALID_USAGE:
        idx.str.startswith("kapow", na="baz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]
        idx.str.endswith("kapow", na="baz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]
        idx.str.contains("kapow", na="baz")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]

        idx_i = pd.Index([1])

        idx_i.str.startswith("kapow")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.endswith("kapow")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.contains("kapow")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]


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
        s.str.find(re.compile(r"p"))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]


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
        idx.str.find(re.compile(r"p"))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]


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

    sr_a = pd.DataFrame({"a": DATA})["a"]

    _check(assert_type(sr_a.str.capitalize(), "pd.Series[str]"))
    _check(assert_type(sr_a.str.casefold(), "pd.Series[str]"))
    check(assert_type(sr_a.str.cat(sep="X"), str), str)
    _check(assert_type(sr_a.str.center(10), "pd.Series[str]"))
    _check(assert_type(sr_a.str.get(2), "pd.Series[str]"))
    _check(assert_type(sr_a.str.ljust(80), "pd.Series[str]"))
    _check(assert_type(sr_a.str.lower(), "pd.Series[str]"))
    _check(assert_type(sr_a.str.lstrip("a"), "pd.Series[str]"))
    _check(assert_type(sr_a.str.normalize("NFD"), "pd.Series[str]"))
    _check(assert_type(sr_a.str.pad(80, "right"), "pd.Series[str]"))
    _check(assert_type(sr_a.str.removeprefix("a"), "pd.Series[str]"))
    _check(assert_type(sr_a.str.removesuffix("e"), "pd.Series[str]"))
    _check(assert_type(sr_a.str.repeat(2), "pd.Series[str]"))
    _check(assert_type(sr_a.str.replace("a", "X"), "pd.Series[str]"))
    _check(
        assert_type(
            sr_a.str.replace(re.compile(r"a"), "X", regex=True), "pd.Series[str]"
        )
    )
    _check(assert_type(sr_a.str.rjust(80), "pd.Series[str]"))
    _check(assert_type(sr_a.str.rstrip(), "pd.Series[str]"))
    _check(assert_type(sr_a.str.slice_replace(0, 2, "XX"), "pd.Series[str]"))
    _check(assert_type(sr_a.str.strip(), "pd.Series[str]"))
    _check(assert_type(sr_a.str.swapcase(), "pd.Series[str]"))
    _check(assert_type(sr_a.str.title(), "pd.Series[str]"))
    _check(
        assert_type(sr_a.str.translate({241: "n"}), "pd.Series[str]"),
    )
    _check(
        assert_type(sr_a.str.translate({241: 240}), "pd.Series[str]"),
    )
    _check(  # tests covariance of table values (table is read-only)
        assert_type(sr_a.str.translate(trans_table), "pd.Series[str]"),
    )
    _check(assert_type(sr_a.str.upper(), "pd.Series[str]"))
    _check(assert_type(sr_a.str.wrap(80), "pd.Series[str]"))
    _check(assert_type(sr_a.str.zfill(10), "pd.Series[str]"))
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
        s.str.wrap(80, False)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[bad-argument-count]

        sr_i = pd.Series([1])

        sr_i.str.capitalize()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.casefold()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.cat(sep="X")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        sr_i.str.center(10)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.get(2)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.ljust(80)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.lower()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.lstrip("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.normalize("NFD")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.pad(80, "right")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.removeprefix("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.removesuffix("e")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.repeat(2)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.replace("a", "X")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        sr_i.str.replace(re.compile(r"a"), "X", regex=True)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        sr_i.str.rjust(80)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.rstrip()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.slice_replace(0, 2, "XX")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.strip()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.swapcase()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.title()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.translate({241: "n"})  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.translate({241: 240})  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.translate(trans_table)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.upper()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.wrap(80)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.zfill(10)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.decode("utf-8")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        sr_i.str.decode("utf-8", dtype=pd.StringDtype())  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]


def test_string_accessors_string_index() -> None:
    idx = pd.Index(DATA)
    _check = functools.partial(check, klass=pd.Index, dtype=str)

    _check(assert_type(idx.str.capitalize(), "pd.Index[str]"))
    _check(assert_type(idx.str.casefold(), "pd.Index[str]"))
    check(assert_type(idx.str.cat(sep="X"), str), str)
    _check(assert_type(idx.str.center(10), "pd.Index[str]"))
    _check(assert_type(idx.str.get(2), "pd.Index[str]"))

    idx_dict = pd.Index(
        [
            {"name": "Hello", "value": "World"},
            {"name": "Goodbye", "value": "Planet"},
        ]
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
    _check(assert_type(idx.str.translate({241: "n"}), "pd.Index[str]"))
    _check(assert_type(idx.str.translate({241: 240}), "pd.Index[str]"))
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

    idx_a = pd.MultiIndex(levels=[DATA], codes=[[0]]).levels[0]
    _check(assert_type(idx_a.str.capitalize(), "pd.Index[str]"))
    _check(assert_type(idx_a.str.casefold(), "pd.Index[str]"))
    check(assert_type(idx_a.str.cat(sep="X"), str), str)
    _check(assert_type(idx_a.str.center(10), "pd.Index[str]"))
    _check(assert_type(idx_a.str.get(2), "pd.Index[str]"))
    _check(assert_type(idx_a.str.ljust(80), "pd.Index[str]"))
    _check(assert_type(idx_a.str.lower(), "pd.Index[str]"))
    _check(assert_type(idx_a.str.lstrip("a"), "pd.Index[str]"))
    _check(assert_type(idx_a.str.normalize("NFD"), "pd.Index[str]"))
    _check(assert_type(idx_a.str.pad(80, "right"), "pd.Index[str]"))
    _check(assert_type(idx_a.str.removeprefix("a"), "pd.Index[str]"))
    _check(assert_type(idx_a.str.removesuffix("e"), "pd.Index[str]"))
    _check(assert_type(idx_a.str.repeat(2), "pd.Index[str]"))
    _check(assert_type(idx_a.str.replace("a", "X"), "pd.Index[str]"))
    _check(
        assert_type(
            idx_a.str.replace(re.compile(r"a"), "X", regex=True), "pd.Index[str]"
        )
    )
    _check(assert_type(idx_a.str.rjust(80), "pd.Index[str]"))
    _check(assert_type(idx_a.str.rstrip(), "pd.Index[str]"))
    _check(assert_type(idx_a.str.slice_replace(0, 2, "XX"), "pd.Index[str]"))
    _check(assert_type(idx_a.str.strip(), "pd.Index[str]"))
    _check(assert_type(idx_a.str.swapcase(), "pd.Index[str]"))
    _check(assert_type(idx_a.str.title(), "pd.Index[str]"))
    _check(
        assert_type(idx_a.str.translate({241: "n"}), "pd.Index[str]"),
    )
    _check(assert_type(idx_a.str.translate({241: 240}), "pd.Index[str]"))
    _check(  # tests covariance of table values (table is read-only)
        assert_type(idx_a.str.translate(trans_table), "pd.Index[str]"),
    )
    _check(assert_type(idx_a.str.upper(), "pd.Index[str]"))
    _check(assert_type(idx_a.str.wrap(80), "pd.Index[str]"))
    _check(assert_type(idx_a.str.zfill(10), "pd.Index[str]"))

    # wrap doesn't accept positional arguments other than width
    if TYPE_CHECKING_INVALID_USAGE:
        idx.str.wrap(80, False)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[bad-argument-count]

        idx_i = pd.Index([1])

        idx_i.str.capitalize()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.casefold()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.cat(sep="X")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        idx_i.str.center(10)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.get(2)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.ljust(80)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.lower()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.lstrip("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.normalize("NFD")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.pad(80, "right")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.removeprefix("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.removesuffix("e")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.repeat(2)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.replace("a", "X")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        idx_i.str.replace(re.compile(r"a"), "X", regex=True)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        idx_i.str.rjust(80)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.rstrip()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.slice_replace(0, 2, "XX")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.strip()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.swapcase()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.title()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.translate({241: "n"})  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.translate({241: 240})  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.translate(trans_table)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.upper()  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.wrap(80)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.zfill(5)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.decode("utf-8")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.decode("utf-8", dtype=pd.StringDtype())  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]


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

    s_a = pd.DataFrame({"a": DATA})["a"]
    _check(assert_type(s_a.str.findall("pp"), "pd.Series[list[str]]"))
    _check(assert_type(s_a.str.findall(re.compile(r"pp")), "pd.Series[list[str]]"))
    _check(assert_type(s_a.str.split("a"), "pd.Series[list[str]]"))
    _check(assert_type(s_a.str.split(re.compile(r"a")), "pd.Series[list[str]]"))
    _check(assert_type(s_a.str.split("a", expand=False), "pd.Series[list[str]]"))
    _check(assert_type(s_a.str.rsplit("a"), "pd.Series[list[str]]"))
    _check(assert_type(s_a.str.rsplit("a", expand=False), "pd.Series[list[str]]"))

    if TYPE_CHECKING_INVALID_USAGE:
        # rsplit doesn't accept compiled pattern
        # it doesn't raise at runtime but produces a nan
        _bad_rsplit_result = s.str.rsplit(re.compile(r"a"))  # type: ignore[call-overload] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]

        idx_i = pd.Index([1])

        idx_i.str.findall("pp")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.findall(re.compile(r"pp"))  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.split("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.split(re.compile(r"a"))  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.split("a", expand=False)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        idx_i.str.rsplit("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.rsplit("a", expand=False)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]


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

    idx_a = pd.MultiIndex(levels=[DATA], codes=[[0]]).levels[0]
    _check(assert_type(idx_a.str.findall("pp"), "pd.Index[list[str]]"))
    _check(assert_type(idx_a.str.findall(re.compile(r"pp")), "pd.Index[list[str]]"))
    _check(assert_type(idx_a.str.split("a"), "pd.Index[list[str]]"))
    _check(assert_type(idx_a.str.split(re.compile(r"a")), "pd.Index[list[str]]"))
    _check(assert_type(idx_a.str.split("a", expand=False), "pd.Index[list[str]]"))
    _check(assert_type(idx_a.str.rsplit("a"), "pd.Index[list[str]]"))
    _check(assert_type(idx_a.str.rsplit("a", expand=False), "pd.Index[list[str]]"))

    if TYPE_CHECKING_INVALID_USAGE:
        # rsplit doesn't accept compiled pattern
        # it doesn't raise at runtime but produces a nan
        _bad_rsplit_result = idx.str.rsplit(re.compile(r"a"))  # type: ignore[call-overload] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]

        idx_i = pd.Index([1])

        idx_i.str.findall("pp")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.findall(re.compile(r"pp"))  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.split("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.split(re.compile(r"a"))  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.split("a", expand=False)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
        idx_i.str.rsplit("a")  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[bad-argument-type]
        idx_i.str.rsplit("a", expand=False)  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]


def test_string_accessors_expanding_series() -> None:
    s = pd.Series(["a1", "b2", "c3"])
    _check = functools.partial(check, klass=pd.DataFrame)
    _check(assert_type(s.str.extract(r"([ab])?(\d)"), pd.DataFrame))
    _check(assert_type(s.str.extract(re.compile(r"([ab])?(\d)")), pd.DataFrame))
    _check(assert_type(s.str.extractall(r"([ab])?(\d)"), pd.DataFrame))
    _check(assert_type(s.str.extractall(re.compile(r"([ab])?(\d)")), pd.DataFrame))
    _check(assert_type(s.str.get_dummies(), pd.DataFrame))
    _check(assert_type(s.str.get_dummies(dtype="boolean"), pd.DataFrame))
    _check(assert_type(s.str.get_dummies(dtype=bool), pd.DataFrame))
    _check(assert_type(s.str.partition("p"), pd.DataFrame))
    _check(assert_type(s.str.rpartition("p"), pd.DataFrame))
    _check(assert_type(s.str.rsplit("a", expand=True), pd.DataFrame))
    _check(assert_type(s.str.split("a", expand=True), pd.DataFrame))


def test_string_accessors_expanding_index() -> None:
    idx = pd.Index(["a1", "b2", "c3"])
    _check = functools.partial(check, klass=pd.MultiIndex)
    _check(assert_type(idx.str.get_dummies(), pd.MultiIndex))
    _check(assert_type(idx.str.get_dummies(dtype=np.uint16), pd.MultiIndex))
    _check(assert_type(idx.str.get_dummies(dtype=np.uint8), pd.MultiIndex))
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


def test_series_str_replace() -> None:
    """Test replace method for Series.str GH1654."""
    sr = pd.Series(data=["A", "B_junk", "C_gunk"], name="my_messy_col")

    check(
        assert_type(sr.str.replace(pat={"A": "a", "B": "b"}), "pd.Series[str]"),
        pd.Series,
        str,
    )

    sr_a = pd.DataFrame({"my_messy_col": ["A", "B_junk", "C_gunk"]})["my_messy_col"]

    check(
        assert_type(sr_a.str.replace(pat={"A": "a", "B": "b"}), "pd.Series[str]"),
        pd.Series,
        str,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        sr.str.replace(pat={"A": "a", "B": "b"}, repl="A")  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType,reportCallIssue] # pyrefly: ignore[no-matching-overload]

        sr_i = pd.Series([1], name="my_messy_col")

        sr_i.str.replace(pat={"A": "a", "B": "b"})  # type: ignore[misc] # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # pyrefly: ignore[no-matching-overload]
