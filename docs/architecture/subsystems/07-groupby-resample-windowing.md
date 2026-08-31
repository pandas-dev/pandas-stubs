**Target Module**: [pandas-stubs/core/groupby/generic.pyi](../../../pandas-stubs/core/groupby/generic.pyi)

# Subsystem: Split-Apply-Combine, GroupBy, Resampler & Windowing

## 1. Overview & Architectural Role

Data aggregation workflows use `DataFrameGroupBy`, `SeriesGroupBy[S1]`, `Resampler`, and `Rolling` windowing engines.

## 2. Historical Debates & Structural Evolution

### Generic SeriesGroupBy and Method Alignment
In PR pandas-dev/pandas-stubs#148: groupby.__iter__() fix types (pandas-dev/pandas-stubs@a6dd774bcb0cb43f209dd88e5adee05998824dd8) and PR pandas-dev/pandas-stubs#190: Added missing groupby methods and made SeriesGroupBy generic (pandas-dev/pandas-stubs@fed3be4c53250ad749f3f78ce7831bb6b27f909c), `Dr-Irv` and `amotzop` made `SeriesGroupBy` generic over element type `S1`, ensuring aggregation methods like `.mean()` and `.sum()` return typed Series.

In PR pandas-dev/pandas-stubs#848: Rework groupby and resample core modules (pandas-dev/pandas-stubs@e35c3ca0c3fc2803cf00ca22ea75d2ae7f0b3948), `hamdanal` performed a major overhaul of groupby and resample core modules (57 discussion threads), aligning custom apply/agg functions.

### Aggregation Return Types
In PR pandas-dev/pandas-stubs#966: GroupBy[Series].count() return type should be Series[int] (pandas-dev/pandas-stubs@7e6aee4e41f8f60b4ce23df87ccfd4f39eb042ef), `chrisyeh96` corrected `GroupBy[Series].count()` to strictly return `Series[int]`. In PR pandas-dev/pandas-stubs#1242: GH456 First attempt GroupBy.transform improved typing (pandas-dev/pandas-stubs@b12c28d7a987e9b67a13ad0e3335f531973c9114), `loicdiridollou` improved `GroupBy.transform` typing.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#148: groupby.__iter__() fix types (pandas-dev/pandas-stubs@a6dd774bcb0cb43f209dd88e5adee05998824dd8)
- pandas-dev/pandas-stubs#166: CLEAN: Align Groupby types (pandas-dev/pandas-stubs@927d4388775c829859e5caf4600b2f8ecf8e190d)
- pandas-dev/pandas-stubs#173: Standardised aggregate functions typing (pandas-dev/pandas-stubs@8f9ba75f595b434987454881e8e016669ab45100)
- pandas-dev/pandas-stubs#177: More specific types for GroupBy.apply. (pandas-dev/pandas-stubs@02e1748becb97e485da6930ab4ed9fea382d8ed9)
- pandas-dev/pandas-stubs#190: Added missing groupby methods and made SeriesGroupBy generic (pandas-dev/pandas-stubs@fed3be4c53250ad749f3f78ce7831bb6b27f909c)
- pandas-dev/pandas-stubs#848: Rework groupby and resample core modules (pandas-dev/pandas-stubs@e35c3ca0c3fc2803cf00ca22ea75d2ae7f0b3948)
- pandas-dev/pandas-stubs#966: GroupBy[Series].count() return type should be Series[int] (pandas-dev/pandas-stubs@7e6aee4e41f8f60b4ce23df87ccfd4f39eb042ef)
- pandas-dev/pandas-stubs#1242: GH456 First attempt GroupBy.transform improved typing (pandas-dev/pandas-stubs@b12c28d7a987e9b67a13ad0e3335f531973c9114)
