import functools
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import check


@pytest.mark.parametrize("constructor", ["series", "index"])
@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("capitalize", {}),
    ],
)
def test_string_accessors_type_preserving_series(
    constructor: Any, method: str, kwargs: Any
) -> None:
    data = ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    s = pd.Series(data)
    _check = functools.partial(check, klass=pd.Series, dtype=str)
    _check(assert_type(s.str.capitalize(), "pd.Series[str]"))
    _check(assert_type(s.str.casefold(), "pd.Series[str]"))
    check(assert_type(s.str.cat(sep="X"), str), str)
    _check(assert_type(s.str.center(10), "pd.Series[str]"))
    _check(assert_type(s.str.get(2), "pd.Series[str]"))
    _check(assert_type(s.str.ljust(80), "pd.Series[str]"))
    _check(assert_type(s.str.lower(), "pd.Series[str]"))
    _check(assert_type(s.str.lstrip("a"), "pd.Series[str]"))
    _check(assert_type(s.str.normalize("NFD"), "pd.Series[str]"))
    _check(assert_type(s.str.pad(80, "right"), "pd.Series[str]"))
    _check(assert_type(s.str.removeprefix("a"), "pd.Series[str]"))
    _check(assert_type(s.str.removesuffix("e"), "pd.Series[str]"))
    _check(assert_type(s.str.repeat(2), "pd.Series[str]"))
    _check(assert_type(s.str.replace("a", "X"), "pd.Series[str]"))
    _check(assert_type(s.str.rjust(80), "pd.Series[str]"))
    _check(assert_type(s.str.rstrip(), "pd.Series[str]"))
    _check(assert_type(s.str.slice(0, 4, 2), "pd.Series[str]"))
    _check(assert_type(s.str.slice_replace(0, 2, "XX"), "pd.Series[str]"))
    _check(assert_type(s.str.strip(), "pd.Series[str]"))
    _check(assert_type(s.str.swapcase(), "pd.Series[str]"))
    _check(assert_type(s.str.title(), "pd.Series[str]"))
    _check(
        assert_type(s.str.translate({241: "n"}), "pd.Series[str]"),
    )
    _check(assert_type(s.str.upper(), "pd.Series[str]"))
    _check(assert_type(s.str.wrap(80), "pd.Series[str]"))
    _check(assert_type(s.str.zfill(10), "pd.Series[str]"))


def test_string_accessors_type_boolean():
    s = pd.Series(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    check(assert_type(s.str.startswith("a"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s.str.startswith(("a", "b")), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type(s.str.contains("a"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s.str.contains(re.compile(r"a")), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type(s.str.endswith("e"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s.str.endswith(("e", "f")), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(assert_type(s.str.fullmatch("apple"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isalnum(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isalpha(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isdecimal(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isdigit(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isnumeric(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.islower(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isspace(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.istitle(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isupper(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.match("pp"), "pd.Series[bool]"), pd.Series, np.bool_)


def test_string_accessors_type_integer():
    s = pd.Series(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    check(assert_type(s.str.find("p"), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.str.index("p"), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.str.rfind("e"), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.str.rindex("p"), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.str.count("pp"), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.str.len(), "pd.Series[int]"), pd.Series, np.integer)


def test_string_accessors_encode_decode():
    s_str = pd.Series(["a1", "b2", "c3"])
    s_bytes = pd.Series([b"a1", b"b2", b"c3"])
    s2 = pd.Series([["apple", "banana"], ["cherry", "date"], [1, "eggplant"]])
    check(
        assert_type(s_bytes.str.decode("utf-8"), "pd.Series[str]"),
        "pd.Series[str]",
        str,
    )
    check(
        assert_type(s_str.str.encode("latin-1"), "pd.Series[bytes]"), pd.Series, bytes
    )
    check(assert_type(s2.str.join("-"), "pd.Series[str]"), pd.Series, str)


def test_string_accessors_list():
    s = pd.Series(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    check(assert_type(s.str.findall("pp"), "pd.Series[list[str]]"), pd.Series, list)
    check(assert_type(s.str.split("a"), "pd.Series[list[str]]"), pd.Series, list)
    # GH 194
    check(
        assert_type(s.str.split("a", expand=False), "pd.Series[list[str]]"),
        pd.Series,
        list,
    )
    check(assert_type(s.str.rsplit("a"), "pd.Series[list[str]]"), pd.Series, list)
    check(
        assert_type(s.str.rsplit("a", expand=False), "pd.Series[list[str]]"),
        pd.Series,
        list,
    )


# def test_string_accessors_expanding():
#     check(assert_type(s3.str.extract(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s3.str.extractall(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.get_dummies(), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.partition("p"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.rpartition("p"), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.rsplit("a", expand=True), pd.DataFrame), pd.DataFrame)
#     check(assert_type(s.str.split("a", expand=True), pd.DataFrame), pd.DataFrame)
