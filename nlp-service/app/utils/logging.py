"""
Shared logging configuration for the NLP service.

Every module should call `get_logger(__name__)` instead of configuring its
own handlers, so log formatting stays consistent across the service.
"""
import logging
import sys

from app.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the root logger once, based on NLP_LOG_LEVEL."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
