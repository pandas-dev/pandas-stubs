from scripts._job import run_job
from scripts.test import _step

_CACHE_STEPS = [_step.clean_mypy_cache, _step.clean_pytest_cache]
_SRC_STEPS = [_step.mypy_src, _step.pyright_src, _step.pytest_src, _step.style_src]
_DIST_STEPS = [
    _step.build_dist,
    _step.install_dist,
    _step.rename_src,
    _step.mypy_dist,
    _step.pyright_dist,
    _step.uninstall_dist,
    _step.restore_src,
]


def test(clean_cache: bool = False, src: bool = False, dist: bool = False):
    steps = []
    if clean_cache:
        steps.extend(_CACHE_STEPS)

    if src:
        steps.extend(_SRC_STEPS)

    if dist:
        steps.extend(_DIST_STEPS)

    run_job(steps)
