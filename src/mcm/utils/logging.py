"""Shared console logging."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
_CONFIGURED = False

# These emit a line per HTTP request. During a model load or a dataset download
# that is hundreds of lines of 404s and redirects burying the output that matters.
_NOISY = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub.file_download")


def get_logger(name: str = "mcm", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        )
        for noisy in _NOISY:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
