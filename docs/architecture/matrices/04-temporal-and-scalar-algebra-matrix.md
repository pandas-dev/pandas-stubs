# Temporal & Scalar Type Algebra Matrix

## 1. Overview

Pandas temporal scalar hierarchy consists of `Timestamp` (datetime), `Timedelta` (duration), `Period` (fixed interval on a regular grid), `Interval[T]` (generic mathematical bounded range), `DateOffset` (calendar offset rules), and sentinel values `NaT` and `pd.NA`.

---

## 2. Temporal Scalar Interaction Matrix

| Left Operand | Operator | Right Operand | Result Type | Semantic Rule | Key Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Timestamp` | `-` | `Timestamp` \| `datetime` | `Timedelta` | Date difference produces duration | pandas-dev/pandas-stubs#383, pandas-dev/pandas-stubs#1274 |
| `Timestamp` | `+`, `-` | `Timedelta` \| `timedelta` | `Timestamp` | Shifting point in time by duration | pandas-dev/pandas-stubs#383, pandas-dev/pandas-stubs#1274 |
| `Timestamp` | `+`, `-` | `BaseOffset` \| `DateOffset` | `Timestamp` | Shifting by calendar frequency | pandas-dev/pandas-stubs#1878, pandas-dev/pandas-stubs#1911 |
| `Timestamp` | `+` | `Timestamp` | **Statically Rejected** (`Never`) | Sum of two dates is mathematically undefined | pandas-dev/pandas-stubs#1274, pandas-dev/pandas-stubs#1312 |
| `Timedelta` | `+`, `-` | `Timedelta` \| `timedelta` | `Timedelta` | Accumulation of durations | pandas-dev/pandas-stubs#388, pandas-dev/pandas-stubs#1273 |
| `Timedelta` | `*`, `/` | `int` \| `float` | `Timedelta` | Scaling duration by scalar factor | pandas-dev/pandas-stubs#388, pandas-dev/pandas-stubs#1397 |
| `Timedelta` | `/` | `Timedelta` \| `timedelta` | `float` | Dimensionless ratio of durations | pandas-dev/pandas-stubs#388, pandas-dev/pandas-stubs#1312 |
| `Timedelta` | `//` | `Timedelta` \| `timedelta` | `int` | Quotient of durations | pandas-dev/pandas-stubs#388, pandas-dev/pandas-stubs#1452 |
| `Period` | `+`, `-` | `int` \| `BaseOffset` | `Period` | Shifting period by frequency units | pandas-dev/pandas-stubs#389, pandas-dev/pandas-stubs#1151 |
| `Period` | `-` | `Period` | `BaseOffset` (or `int`) | Distance between periods | pandas-dev/pandas-stubs#389, pandas-dev/pandas-stubs#1151 |
| `Interval[T]` | `+`, `-` | `T` (numeric) | `Interval[T]` | Translating interval endpoints | pandas-dev/pandas-stubs#174, pandas-dev/pandas-stubs#1845 |
| `Interval[Timestamp]` | `+`, `-` | `Timedelta` | `Interval[Timestamp]` | Shifting time interval by duration | pandas-dev/pandas-stubs#174, pandas-dev/pandas-stubs#1845 |
| `Interval[Timestamp]` | `+` | `Timestamp` | **Statically Rejected** (`Never`) | Adding timestamp to time interval rejected | pandas-dev/pandas-stubs#174, pandas-dev/pandas-stubs#1845 |
| `NaT` | `==`, `!=`, `<`, `>` | `Any` | `bool` (or `NaTType`) | Sentinel comparison semantics | pandas-dev/pandas-stubs#1915 |
| `pd.NA` (`NAType`) | `&`, `\|`, `^` | `bool` \| `NAType` | `bool` \| `NAType` | Kleene three-valued logic modeling | pandas-dev/pandas-stubs#945, pandas-dev/pandas-stubs#1909 |
