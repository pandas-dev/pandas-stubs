from collections.abc import Generator
import gc

import pytest


@pytest.fixture
def mpl_cleanup() -> Generator[None, None, None]:
    """
    Ensure Matplotlib is cleaned up around a test.

    Before a test is run:

    1) Set the backend to "template" to avoid requiring a GUI.

    After a test is run:

    1) Reset units registry
    2) Reset rc_context
    3) Close all figures

    See matplotlib/testing/decorators.py#L24.
    """
    mpl = pytest.importorskip("matplotlib")
    mpl_units = pytest.importorskip("matplotlib.units")
    plt = pytest.importorskip("matplotlib.pyplot")
    orig_units_registry = mpl_units.registry.copy()
    try:
        with mpl.rc_context():
            mpl.use("template")
            yield
    finally:
        mpl_units.registry.clear()
        mpl_units.registry.update(orig_units_registry)
        plt.close("all")
        # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.6.0.html#garbage-collection-is-no-longer-run-on-figure-close
        gc.collect(1)
