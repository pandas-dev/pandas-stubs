from scripts._job import run_job
from scripts.test import _step


def test_src(profile: str, clean_cache: bool = False):
    """ """
    steps = []
    if clean_cache:
        steps.extend(
            [
                _step.clean_mypy_cache,
                _step.clean_pytest_cache,
            ]
        )

    if profile in (None, "", "default"):
        steps.extend([_step.mypy_src, _step.pyright_src])
    elif profile == "pytest":
        steps.extend([_step.pytest_src])
    elif profile == "full":
        steps.extend([_step.mypy_src, _step.pyright_src, _step.pytest_src])
    else:
        raise ModuleNotFoundError("Profile not found!")

    run_job(steps)


def test_dist(clean_cache: bool = False):
    """ """
    steps = []
    if clean_cache:
        steps.extend(
            [
                _step.clean_mypy_cache,
                _step.clean_pytest_cache,
            ]
        )

    steps.extend(
        [
            _step.build_dist,
            _step.install_dist,
            _step.rename_src,
            _step.mypy_dist,
            _step.pyright_src,
            _step.uninstall_dist,
            _step.restore_src,
        ]
    )

    run_job(steps)


def test_all(clean_cache: bool = False):
    """ """
    steps = []
    if clean_cache:
        steps.extend(
            [
                _step.clean_mypy_cache,
                _step.clean_pytest_cache,
            ]
        )

    steps.extend(
        [
            _step.mypy_src,
            _step.pyright_src,
            _step.pytest_src,
            _step.build_dist,
            _step.install_dist,
            _step.rename_src,
            _step.mypy_dist,
            _step.pyright_src,
            _step.uninstall_dist,
            _step.restore_src,
        ]
    )

    run_job(steps)
