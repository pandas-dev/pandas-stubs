from typing import Literal

from scripts._job import run_job
from scripts.test import _step

_SRC_STEPS = [_step.mypy_src, _step.pyright_src, _step.pytest, _step.style]
_DIST_STEPS = [
    _step.build_dist,
    _step.install_dist,
    _step.rename_src,
    _step.mypy_dist,
    _step.pyright_dist,
    _step.uninstall_dist,
    _step.restore_src,
]


def test(
    src: bool = False,
    dist: bool = False,
    type_checker: Literal["", "mypy", "pyright"] = "",
):
    steps = []
    if src:
        steps.extend(_SRC_STEPS)

    if dist:
        steps.extend(_DIST_STEPS)

    if type_checker:
        # either pyright or mypy
        remove = "mypy" if type_checker == "pyright" else "pyright"
        steps = [step for step in steps if remove not in step.name]

    run_job(steps)
