import time
from dataclasses import dataclass
from subprocess import CalledProcessError
from typing import Callable, List, Union, Optional

from loguru import logger


@dataclass
class Step:
    name: str
    run: Callable[[], None]
    rollback: Optional[Callable[[], None]] = None


def __rollback_job(steps: List[Step]):
    """
    Resposible to run rollback of steps.
    """

    if len(steps) > 0:
        for step in steps[-1:]:
            logger.warning(f"Undoing step: {step.name}")
            if step.rollback is not None:
                step.rollback()
                logger.warning(f"Undoing step: {step.name}")

    logger.success(f"End of rollback with success")


def run_job(steps: List[Step]) -> None:
    """
    Responsible to run steps with logs.
    """

    rollback_steps: List[Step] = []

    for step in steps:
        start = time.perf_counter()
        logger.info(f"Beginning: '{step.name}'")

        try:

            if step.rollback is not None:
                rollback_steps.append(step)

            step.run()

        except CalledProcessError:

            logger.error(f"'{step.name}' failed!")
            __rollback_job(rollback_steps)
            
            break

        end = time.perf_counter()
        logger.success(f"End: '{step.name}', runtime: {end - start:.3f} seconds.")

