"""Logging setup for MarceLLo training runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """Configure logging with rich console output and optional file logging."""
    logger = logging.getLogger("marcello")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
        )
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
