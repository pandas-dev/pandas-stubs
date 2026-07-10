import dataclasses
from functools import partial
from typing import (
    TYPE_CHECKING,
    Literal,
    get_args,
)

from scripts._job import run_job
from scripts.test import _step

if TYPE_CHECKING:
    from scripts._job import Step

_SRC_STEPS = [
    _step.ty_src,
    _step.pyrefly_src,
    _step.mypy_src,
    _step.pyright_src,
    _step.pytest,
    _step.style,
]
_DIST_STEPS = [
    _step.build_dist,
    _step.install_dist,
    _step.rename_src,
    _step.ty_dist,
    # _step.pyrefly_dist,  TODO: pandas-dev/pandas-stubs#1801
    _step.pyright_dist,
    _step.mypy_dist,
]

TypeChecker = Literal["mypy", "pyright", "pyrefly", "ty"]


def run_tests(
    src: bool = False,
    dist: bool = False,
    type_checker: TypeChecker | None = None,
) -> None:
    steps: list[Step] = []
    if src:
        steps.extend(_SRC_STEPS)

    if dist:
        steps.extend(_DIST_STEPS)

    if type_checker:
        steps = [
            s
            for s in steps
            if type_checker in s.name
            or not any(t in s.name for t in get_args(TypeChecker))
        ]

    run_job(steps)


def stubtest(allowlist: str, check_missing: bool, nightly: bool) -> None:
    stubtest = dataclasses.replace(
        _step.stubtest,
        run=partial(
            _step.stubtest.run, allowlist=allowlist, check_missing=check_missing
        ),
    )
    steps = _DIST_STEPS[:2]
    if nightly:
        steps.append(_step.nightly)
    run_job([*steps, stubtest])


def pytest(nightly: bool) -> None:
    setup_steps = []
    pytest_step = _step.pytest
    if nightly:
        setup_steps = [_step.nightly]
    run_job([*setup_steps, pytest_step])


def mypy_src(mypy_nightly: bool) -> None:
    steps = [_step.mypy_nightly] if mypy_nightly else []
    run_job([*steps, _step.mypy_src])
