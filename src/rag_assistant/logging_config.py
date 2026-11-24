"""Structured application logging configuration."""

from __future__ import annotations

import logging
import logging.config


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent console logging for the application and quiet noisy libraries."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "rag_assistant": {
                    "handlers": ["console"],
                    "level": level,
                    "propagate": False,
                },
                "httpx": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
                "chromadb.telemetry": {"level": "ERROR"},
            },
            "root": {
                "handlers": ["console"],
                "level": "WARNING",
            },
        }
    )
