from __future__ import annotations

import pandas as pd


def test_types_merge() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df2 = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [0, 1, 0]})
    columns = ["col1", "col2"]
    df.merge(df2, on=columns)
