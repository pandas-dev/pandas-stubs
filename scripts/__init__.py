import sys
from typing import Any

from loguru import logger

# Config the format of log message
config: dict[str, Any] = {
    "handlers": [
        {
            "sink": sys.stderr,
            "format": (
                "<level>\n"
                "===========================================\n"
                "{message}\n"
                "===========================================\n"
                "</level>"
            ),
        }
    ]
}

logger.configure(**config)
