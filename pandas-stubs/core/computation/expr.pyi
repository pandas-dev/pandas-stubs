from __future__ import annotations

import ast

from pandas.core.computation.ops import Term as Term
from pandas.core.computation.scope import Scope as Scope

intersection = ...

def disallow(nodes): ...
def add_ops(op_classes): ...

class BaseExprVisitor(ast.NodeVisitor):
    const_type = ...
    term_type = ...
    binary_ops = ...
    binary_op_nodes = ...
    binary_op_nodes_map = ...
    unary_ops = ...
    unary_op_nodes = ...
    unary_op_nodes_map = ...
    rewrite_map = ...
    env = ...
    engine = ...
    parser = ...
    preparser = ...
    assigner = ...
    def __init__(self, env, engine, parser, preparser=...) -> None: ...
    def visit(self, node, **kwargs): ...
    def visit_Module(self, node, **kwargs): ...
    def visit_Expr(self, node, **kwargs): ...
    def visit_BinOp(self, node, **kwargs): ...
    def visit_Div(self, node, **kwargs): ...
    def visit_UnaryOp(self, node, **kwargs): ...
    def visit_Name(self, node, **kwargs): ...
    def visit_NameConstant(self, node, **kwargs): ...
    def visit_Num(self, node, **kwargs): ...
    def visit_Constant(self, node, **kwargs): ...
    def visit_Str(self, node, **kwargs): ...
    def visit_List(self, node, **kwargs): ...
    def visit_Index(self, node, **kwargs): ...
    def visit_Subscript(self, node, **kwargs): ...
    def visit_Slice(self, node, **kwargs): ...
    def visit_Assign(self, node, **kwargs): ...
    def visit_Attribute(self, node, **kwargs): ...
    def visit_Call(self, node, side=..., **kwargs): ...
    def translate_In(self, op): ...
    def visit_Compare(self, node, **kwargs): ...
    def visit_BoolOp(self, node, **kwargs): ...

class PandasExprVisitor(BaseExprVisitor):
    def __init__(self, env, engine, parser, preparser=...) -> None: ...

class PythonExprVisitor(BaseExprVisitor):
    def __init__(self, env, engine, parser, preparser=...): ...

class Expr:
    env: Scope
    engine: str
    parser: str
    expr = ...
    terms = ...
    def __init__(
        self,
        expr,
        engine: str = ...,
        parser: str = ...,
        env: Scope | None = ...,
        level: int = ...,
    ) -> None: ...
    @property
    def assigner(self): ...
    def __call__(self): ...
    def __len__(self) -> int: ...
    def parse(self): ...
    @property
    def names(self): ...
