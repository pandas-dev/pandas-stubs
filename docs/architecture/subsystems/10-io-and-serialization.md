**Target Module**: [pandas-stubs/io/api.pyi](../../../pandas-stubs/io/api.pyi)

# Subsystem: I/O Parsers, Serialization & Table Styling

## 1. Overview & Architectural Role

Pandas I/O parsers (`read_csv`, `read_excel`, `read_json`, `read_parquet`, `read_stata`) convert external formats into DataFrames, accepting dozens of engine-specific options.

## 2. Historical Debates & Design Decisions

### CSV usecols and Quoting
In PR pandas-dev/pandas-stubs#65: added type hint to read_csv usecols (pandas-dev/pandas-stubs@a4860495b0a7852719ad7503899d5a10befd8066), `sofcalca` typed `read_csv(usecols=...)`. Later, CSV quoting was updated for Python 3.12 compatibility (`QUOTE_STRINGS`, `QUOTE_NOTNULL`).

### Styler Typing
In PR pandas-dev/pandas-stubs#282: ENH: Improve Styler typing (pandas-dev/pandas-stubs@ad517c74050f6721469aeba1cdf8ce1a8d5f277e), `bashtage` overhauled `Styler` typing (38 discussion threads), introducing `StyleExportDict` and CSS property maps.

### JSON Reader & Holiday Calendar Attributes
In PR pandas-dev/pandas-stubs#1819: type `JsonReader.__init__`, `AbstractHolidayCalendar` attributes, `Holiday` attributes (pandas-dev/pandas-stubs@6c036a9f0cc108465109e4b971587bbd53bc9997), `MarcoGorelli` typed `JsonReader.__init__` and calendar objects.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#65: added type hint to read_csv usecols (pandas-dev/pandas-stubs@a4860495b0a7852719ad7503899d5a10befd8066)
- pandas-dev/pandas-stubs#282: ENH: Improve Styler typing (pandas-dev/pandas-stubs@ad517c74050f6721469aeba1cdf8ce1a8d5f277e)
- pandas-dev/pandas-stubs#1765: Remove `pyrefly: ignore-errors` in `test_io.py` (pandas-dev/pandas-stubs@ba78c4b331b02316cf6e3eb6d9a82af2c083750a)
- pandas-dev/pandas-stubs#1819: type `JsonReader.__init__`, `AbstractHolidayCalendar` attributes, `Holiday` attributes (pandas-dev/pandas-stubs@6c036a9f0cc108465109e4b971587bbd53bc9997)
