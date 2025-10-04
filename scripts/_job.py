from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import sys
import time

from loguru import logger


@dataclass
class Step:
    name: str
    run: Callable[[], None]
    rollback: Callable[[], None] | None = None


def __rollback_job(steps: deque[Step]) -> None:
    """
    Responsible to run rollback of steps.
    """

    while steps:
        step = steps.pop()
        if step.rollback is not None:
            logger.warning(f"Undoing: {step.name}")
            try:
                step.rollback()
            except Exception:
                logger.error(
                    f"Rollback of Step: '{step.name}' failed! The project could be in a unstable mode."
                )


def run_job(steps: list[Step]) -> None:
    """
    Responsible to run steps with logs.
    """

    rollback_steps = deque[Step]()
    failed = False

    for step in steps:
        start = time.perf_counter()
        logger.info(f"Beginning: '{step.name}'")

        try:
            rollback_steps.append(step)
            step.run()

        except Exception:
            logger.error(f"Step: '{step.name}' failed!")
            __rollback_job(rollback_steps)
            failed = True

            break

        end = time.perf_counter()
        logger.success(f"End: '{step.name}', runtime: {end - start:.3f} seconds.")

    if not failed:
        __rollback_job(rollback_steps)

    if failed:
        sys.exit(1)
