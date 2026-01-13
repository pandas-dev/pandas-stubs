from collections.abc import (
    Generator,
    Iterable,
)
from itertools import (
    chain,
    combinations,
)

from typing_extensions import TypeVar

T = TypeVar("T")


def powerset(
    iterable: Iterable[T], minimal_size: int = 0
) -> Generator[tuple[T, ...], None, None]:
    """Subsequences of the iterable from shortest to longest.

    powerset([1,2,3], 0) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    yield from chain.from_iterable(
        combinations(s, r) for r in range(minimal_size, len(s) + 1)
    )
