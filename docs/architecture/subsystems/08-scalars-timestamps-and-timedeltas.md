**Target Module**: [pandas-stubs/_libs/tslibs/timestamps.pyi](../../../pandas-stubs/_libs/tslibs/timestamps.pyi)

# Subsystem: Temporal Scalars, Timestamps, Timedeltas & DateOffsets

## 1. Overview & Architectural Role

Pandas provides scalar objects (`Timestamp`, `Timedelta`, `Period`, `Interval`, `DateOffset`, `BaseOffset`) that interoperate with Python standard library `datetime` objects.

## 2. Historical Debates & Enhancements

In late 2022, `bashtage` executed a comprehensive series of enhancements to scalar typing:
- PR pandas-dev/pandas-stubs#383: ENH: Improve Pandas scalars: Improved Pandas scalar definitions
- PR pandas-dev/pandas-stubs#388: ENH: Improve typing for Timedelta (pandas-dev/pandas-stubs@5df1c515c8f8b14d402bd5dee0aaa8503074e4fb): Improved typing for `Timedelta`
- PR pandas-dev/pandas-stubs#389: ENH: Improve typing for Timestamp (pandas-dev/pandas-stubs@56bcec7df68bf41bc12136a7cb1166c80d5d5be6): Improved typing for `Timestamp`
- PR pandas-dev/pandas-stubs#390: ENH: Improve typing for Period (pandas-dev/pandas-stubs@69710a158023631b6ee5aafe2c40b94fe9b0262d): Improved typing for `Period`

In PR pandas-dev/pandas-stubs#1151: make tslibs strptime, timedeltas, and timestamps pass with pyright-strict (pandas-dev/pandas-stubs@69b833cc8343055b47c12b1db8cad7fce3fe26a7), `loicdiridollou` updated timestamp and timedelta aliases for pyright-strict compliance. In PR pandas-dev/pandas-stubs#1878: BUG: relocate `to_offset` and other `pyrefly`-inspired changes (pandas-dev/pandas-stubs@b0e70149eacc63fcf053c7838cca0509f13d6008), `cmp0xff` relocated `to_offset` to address pyrefly strictness. In PR pandas-dev/pandas-stubs#1911: TYP: accept datetime.timedelta as freq for round, floor, and ceil (pandas-dev/pandas-stubs@6528be476d3583bb8f56cf3a062a1d4ec32a15ba), `ghackebeil` added support for `datetime.timedelta` as frequency arguments.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#383: ENH: Improve Pandas scalars
- pandas-dev/pandas-stubs#388: ENH: Improve typing for Timedelta (pandas-dev/pandas-stubs@5df1c515c8f8b14d402bd5dee0aaa8503074e4fb)
- pandas-dev/pandas-stubs#389: ENH: Improve typing for Timestamp (pandas-dev/pandas-stubs@56bcec7df68bf41bc12136a7cb1166c80d5d5be6)
- pandas-dev/pandas-stubs#390: ENH: Improve typing for Period (pandas-dev/pandas-stubs@69710a158023631b6ee5aafe2c40b94fe9b0262d)
- pandas-dev/pandas-stubs#1151: make tslibs strptime, timedeltas, and timestamps pass with pyright-strict (pandas-dev/pandas-stubs@69b833cc8343055b47c12b1db8cad7fce3fe26a7)
- pandas-dev/pandas-stubs#1878: BUG: relocate `to_offset` and other `pyrefly`-inspired changes (pandas-dev/pandas-stubs@b0e70149eacc63fcf053c7838cca0509f13d6008)
- pandas-dev/pandas-stubs#1911: TYP: accept datetime.timedelta as freq for round, floor, and ceil (pandas-dev/pandas-stubs@6528be476d3583bb8f56cf3a062a1d4ec32a15ba)
