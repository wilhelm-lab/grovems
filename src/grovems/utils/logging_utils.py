from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for a CLI entry-point script.

    Args:
        level: Root logging level (default ``logging.INFO``).
    """
    logging.basicConfig(level=level, stream=sys.stdout, format=LOG_FORMAT)
