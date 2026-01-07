import pytest


def test_no_default_alias() -> None:
    from pandas._libs.lib import no_default

    assert no_default

    msg = r"cannot import name 'NoDefaultDoNotUse' from 'pandas._libs.lib'"
    with pytest.raises(ImportError, match=msg):
        from pandas._libs.lib import (  # isort: skip
            NoDefaultDoNotUse as NoDefaultDoNotUse,  # noqa: PLC0414
        )
