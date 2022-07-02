from scripts._job import run_job
from scripts.style import _step


def check_style():
    steps = [_step.check_pyupgrade, _step.check_black, _step.check_isort]

    run_job(steps)


def format_style():
    steps = [_step.format_pyupgrade, _step.format_black, _step.format_isort]

    run_job(steps)
