import ast
from pathlib import Path


def get_args(item):
    args = (
        item.posonlyargs
        + item.args
        + ([item.vararg] if item.vararg else [])
        + item.kwonlyargs
        + ([item.kwarg] if item.kwarg else [])
    )
    return {arg.arg: dispatch(arg.annotation) for arg in args}


def get_class(item):
    result = {}
    for x in item.body:
        result.update(dispatch(x, only_dict=True))
    return {item.name: result}


def parse(path: Path):
    result = {}
    for item in ast.parse(
        path.read_text(), filename=path, type_comments=True, feature_version=(3, 10)
    ).body:
        result.update(dispatch(item, only_dict=True))
    return result


def dispatch(item, only_dict=False):
    if isinstance(
        item,
        (
            ast.Import,
            ast.ImportFrom,
            ast.For,
            ast.Raise,
            ast.Delete,
            ast.Try,
            ast.Call,
            ast.With,
            ast.AugAssign,
            ast.Pass,
            ast.Assert,
        ),
    ):
        return {}
    elif isinstance(item, ast.Assign):
        result = {}
        for x in item.targets:
            result.update({dispatch(x): {}})
        return result
    elif isinstance(item, ast.AnnAssign):
        return {item.target.id: dispatch(item.annotation)}
    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {item.name: {"in": get_args(item.args), "out": dispatch(item.returns)}}
    elif isinstance(item, ast.ClassDef):
        return get_class(item)
    elif item is None:
        return {}
    elif isinstance(item, ast.Expr):
        return dispatch(item.value, only_dict=only_dict)
    elif isinstance(item, ast.If):
        result = {}
        for x in item.body:
            result.update(dispatch(x, only_dict=True))
        return result
    elif only_dict:
        return {}
    elif isinstance(item, ast.Name):
        return f"{item.id}"
    elif isinstance(item, ast.Constant):
        return f"{item.value}"
    elif isinstance(item, ast.Subscript):
        return f"{dispatch(item.value)}[{dispatch(item.slice)}]"
    elif isinstance(item, ast.BinOp):
        return f"{dispatch(item.op)}({dispatch(item.left)}, {dispatch(item.right)})"
    elif isinstance(item, ast.Tuple):
        return f"[{','.join([dispatch(x) for x in item.elts])}]]"
    elif isinstance(item, ast.Attribute):
        return f"{dispatch(item.value)}.{item.attr}"
    elif isinstance(item, ast.List):
        return f"List[{''.join([dispatch(x) for x in item.elts])}]"
    elif not [x for x in dir(item) if not x.startswith("_")]:
        name = item.__class__.__name__
        if name == "BitOr":
            name = "Union"
        return name
    else:
        assert only_dict


def parse_module(root, extension):
    result = {}
    for pyi in root.glob(f"**/*.{extension}"):
        result.update({str(pyi.relative_to(root).with_suffix("")): parse(pyi)})
    return result


def pandas_stubs():
    stubs = parse_module(Path("pandas-stubs"), "pyi")
    pandas = parse_module(Path("/var/home/twoertwein/pandas/pandas"), "pyi")
    pandas.update(parse_module(Path("/var/home/twoertwein/pandas/pandas"), "py"))

    for key, values in pandas.items():
        if key not in stubs:
            continue
        compare(key, values, stubs[key])


def compare(prefix, pandas, stubs):
    if isinstance(pandas, dict) and pandas:
        if isinstance(stubs, str):
            if False:
                print("stubs function", prefix, pandas)
            return

        for key, values in pandas.items():
            if key not in stubs:
                continue
            compare(f"{prefix}.{key}", values, stubs[key])
    elif isinstance(pandas, str) and not stubs:
        if False:
            print("missing in stubs", prefix, pandas)
    elif isinstance(pandas, str) and isinstance(stubs, str):
        if pandas != stubs and False:
            print("different", prefix, pandas, stubs)
    elif stubs and True:
        print("missing in pandas", prefix, stubs)


pandas_stubs()
