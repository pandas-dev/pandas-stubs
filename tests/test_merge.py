from __future__ import annotations

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check


def test_types_merge() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df2 = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [0, 1, 0]})
    columns = ["col1", "col2"]
    df.merge(df2, on=columns)

    check(
        assert_type(df.merge(df2, on=pd.Series([1, 2, 3])), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(df.merge(df2, on=pd.Index([1, 2, 3])), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(df.merge(df2, on=np.array([1, 2, 3])), pd.DataFrame), pd.DataFrame
    )

    check(
        assert_type(
            df.merge(df2, left_on=pd.Series([1, 2, 3]), right_on=pd.Series([1, 2, 3])),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.merge(df2, left_on=pd.Index([1, 2, 3]), right_on=pd.Series([1, 2, 3])),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.merge(df2, left_on=pd.Index([1, 2, 3]), right_on=pd.Index([1, 2, 3])),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(
            df.merge(df2, left_on=np.array([1, 2, 3]), right_on=pd.Series([1, 2, 3])),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.merge(df2, left_on=np.array([1, 2, 3]), right_on=pd.Index([1, 2, 3])),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.merge(df2, left_on=np.array([1, 2, 3]), right_on=np.array([1, 2, 3])),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(pd.merge(df, df2, on=pd.Series([1, 2, 3])), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge(df, df2, on=pd.Index([1, 2, 3])), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(pd.merge(df, df2, on=np.array([1, 2, 3])), pd.DataFrame),
        pd.DataFrame,
    )

    check(
        assert_type(
            pd.merge(
                df, df2, left_on=pd.Series([1, 2, 3]), right_on=pd.Series([1, 2, 3])
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                df, df2, left_on=pd.Index([1, 2, 3]), right_on=pd.Series([1, 2, 3])
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                df, df2, left_on=pd.Index([1, 2, 3]), right_on=pd.Index([1, 2, 3])
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(
            pd.merge(
                df, df2, left_on=np.array([1, 2, 3]), right_on=pd.Series([1, 2, 3])
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                df, df2, left_on=np.array([1, 2, 3]), right_on=pd.Index([1, 2, 3])
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.merge(
                df, df2, left_on=np.array([1, 2, 3]), right_on=np.array([1, 2, 3])
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
