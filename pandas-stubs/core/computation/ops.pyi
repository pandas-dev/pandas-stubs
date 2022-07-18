from __future__ import annotations

import numpy as np

class UndefinedVariableError(NameError):
    def __init__(self, name, is_local: bool) -> None: ...

class Term:
    def __new__(cls, name, env, side=..., encoding=...): ...
    is_local: bool
    env = ...
    side = ...
    encoding = ...
    def __init__(self, name, env, side=..., encoding=...) -> None: ...
    @property
    def local_name(self) -> str: ...
    def __call__(self, *args, **kwargs): ...
    def evaluate(self, *args, **kwargs): ...
    def update(self, value) -> None: ...
    @property
    def is_scalar(self) -> bool: ...
    @property
    def type(self): ...
    return_type = ...
    @property
    def raw(self) -> str: ...
    @property
    def is_datetime(self) -> bool: ...
    @property
    def value(self): ...
    @value.setter
    def value(self, new_value) -> None: ...
    @property
    def name(self): ...
    @property
    def ndim(self) -> int: ...

class Constant(Term):
    def __init__(self, value, env, side=..., encoding=...) -> None: ...
    @property
    def name(self): ...

class Op:
    op: str
    operands = ...
    encoding = ...
    def __init__(self, op: str, operands, *args, **kwargs) -> None: ...
    def __iter__(self): ...
    @property
    def return_type(self): ...
    @property
    def has_invalid_return_type(self) -> bool: ...
    @property
    def operand_types(self): ...
    @property
    def is_scalar(self) -> bool: ...
    @property
    def is_datetime(self) -> bool: ...

def is_term(obj) -> bool: ...

class BinOp(Op):
    lhs = ...
    rhs = ...
    func = ...
    def __init__(self, op: str, lhs, rhs, **kwargs) -> None: ...
    def __call__(self, env): ...
    def evaluate(self, env, engine: str, parser, term_type, eval_in_python): ...
    def convert_values(self): ...

def isnumeric(dtype) -> bool: ...

class Div(BinOp):
    def __init__(self, lhs, rhs, **kwargs) -> None: ...

class UnaryOp(Op):
    operand = ...
    func = ...
    def __init__(self, op: str, operand) -> None: ...
    def __call__(self, env): ...
    @property
    def return_type(self) -> np.dtype: ...

class MathCall(Op):
    func = ...
    def __init__(self, func, args) -> None: ...
    def __call__(self, env): ...

class FuncNode:
    name = ...
    func = ...
    def __init__(self, name: str) -> None: ...
    def __call__(self, *args): ...
