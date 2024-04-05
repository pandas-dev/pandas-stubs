import pandas as pd

s = pd.Series(["A", "B", "AB"])

s.apply(tuple)
s.apply(list)
s.apply(set)
s.apply(frozenset)