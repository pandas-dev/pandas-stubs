from __future__ import annotations

def cut(
    x,
    bins,
    right: bool = ...,
    labels=...,
    retbins: bool = ...,
    precision: int = ...,
    include_lowest: bool = ...,
    duplicates: str = ...,
): ...
def qcut(
    x, q, labels=..., retbins: bool = ..., precision: int = ..., duplicates: str = ...
): ...
