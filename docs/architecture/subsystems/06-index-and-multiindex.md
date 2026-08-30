**Target Module**: [pandas-stubs/core/indexes/base.pyi](../../../pandas-stubs/core/indexes/base.pyi)

# Subsystem: Index Hierarchy, IndexOpsMixin, and MultiIndex

## 1. Overview & Architectural Role

Pandas indices manage row and column labels. `Index[S0]` is generic over label type `S0`, sharing core collection operations with `Series` via `IndexOpsMixin[S0]`.

## 2. Historical Debates & Structural Changes

### Generic `IndexOpsMixin` and `Index`
In PR pandas-dev/pandas-stubs#760: Make IndexOpsMixin (and Index) generic (pandas-dev/pandas-stubs@f7621f408f4cfed08453e1b906bf9a6a17b34b04), `twoertwein` made `IndexOpsMixin` and `Index` generic, creating a unified type hierarchy across `Index`, `DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`, and `MultiIndex`.

### MultiIndex Tuple Selection and Level Swapping
In PR pandas-dev/pandas-stubs#1437: Fix: proper return types for MultiIndex.swaplevel & MultiIndex.union (pandas-dev/pandas-stubs@143bab4291b054ea9ec6c6b30a95b32f485e0373), `zacharybrownjohn` resolved long-standing return type ambiguities in `MultiIndex.swaplevel()` and `MultiIndex.union()`.

## 3. Key Pull Requests & Commits

- pandas-dev/pandas-stubs#760: Make IndexOpsMixin (and Index) generic (pandas-dev/pandas-stubs@f7621f408f4cfed08453e1b906bf9a6a17b34b04)
- pandas-dev/pandas-stubs#1437: Fix: proper return types for MultiIndex.swaplevel & MultiIndex.union (pandas-dev/pandas-stubs@143bab4291b054ea9ec6c6b30a95b32f485e0373)
