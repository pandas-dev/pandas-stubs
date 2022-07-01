from scripts._job import Step
from scripts.style import run

check_black = Step(name="Check Black Style Code", run=run.check_black)
check_isort = Step(name="Check Isort Style Code", run=run.check_isort)
format_black = Step(name="Format Black Style Code", run=run.format_black)
format_isort = Step(name="Format Isort Style Code", run=run.format_isort)
