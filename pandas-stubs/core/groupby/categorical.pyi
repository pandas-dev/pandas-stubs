from pandas.core.arrays.categorical import Categorical

def recode_for_groupby(
    c: Categorical, sort: bool, observed: bool
) -> tuple[Categorical, Categorical | None]: ...
