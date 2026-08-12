from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def configure_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure root logging for a CLI entry-point script; log_file (if given) is appended to alongside stdout."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers)
