from typing import Callable

from pandas._typing import TypeT

def register_dataframe_accessor(name: str) -> Callable[[TypeT], TypeT]: ...
def register_series_accessor(name: str) -> Callable[[TypeT], TypeT]: ...
def register_index_accessor(name: str) -> Callable[[TypeT], TypeT]: ...
