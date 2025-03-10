from .offsets import BaseOffset

class PeriodDtypeBase:
    def __init__(self, code: int) -> None: ...
    def __eq__(self, other) -> bool: ...
    @property
    def date_offset(self) -> BaseOffset: ...
    @classmethod
    def from_date_offset(cls, offset: BaseOffset) -> PeriodDtypeBase: ...

class FreqGroup:
    FR_ANN: int
    FR_QTR: int
    FR_MTH: int
    FR_WK: int
    FR_BUS: int
    FR_DAY: int
    FR_HR: int
    FR_MIN: int
    FR_SEC: int
    FR_MS: int
    FR_US: int
    FR_NS: int
    FR_UND: int

    @staticmethod
    def get_freq_group(code: int) -> int: ...
