# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Standard-library logging helpers for the Accuracy Debugger."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Final

__all__ = [
    "LOG_FILE_SUFFIX",
    "LOG_FORMAT",
    "configure_worker_logging",
    "get_logger",
    "resolve_log_level",
]

LOG_FORMAT: Final = "%(name)s: %(asctime)s | %(levelname)s | %(message)s"
LOG_FILE_SUFFIX: Final = "user_info.log"

_MAX_LOG_BYTES: Final = 10485760
_LOG_BACKUP_COUNT: Final = 40
_HANDLER_MARK: Final = "_adbg_handler"
_VERBOSE: Final = "VERBOSE"
_LEVEL_NAMES: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

# Holds "VERBOSE" once the non-standard level warning has been printed (no `global` needed).
_VERBOSE_WARNED: Final[set[str]] = set()


def resolve_log_level(level: str | int) -> int:
    """Resolves a log level name or number to a ``logging`` level.

    ``"VERBOSE"`` is mapped to ``DEBUG`` and a warning is printed the first time. ``"WARN"`` maps to ``WARNING``.
    Names are case-insensitive.

    Args:
        level: Level name (e.g. ``"INFO"``) or integer level.

    Returns:
        The integer logging level.

    Raises:
        ValueError: If the name is not a known level.
    """
    if isinstance(level, bool) or not isinstance(level, (str, int)):
        raise ValueError(f"Invalid log level: {level!r}")
    if isinstance(level, int):
        return int(level)
    name = level.strip().upper()
    if name == _VERBOSE:
        if _VERBOSE not in _VERBOSE_WARNED:
            print("WARNING!: 'VERBOSE' is not a standard log level. Using 'DEBUG' instead.\n")
            _VERBOSE_WARNED.add(_VERBOSE)
        return logging.DEBUG
    try:
        return _LEVEL_NAMES[name]
    except KeyError:
        raise ValueError(f"Invalid log level: {level!r}") from None


def _remove_marked_handlers(logger: logging.Logger) -> None:
    """Closes and removes every handler previously attached by this module."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()


def _attach(logger: logging.Logger, handler: logging.Handler) -> None:
    """Formats, marks and attaches a handler."""
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)


def get_logger(
    logger_name: str,
    log_file_path: Path,
    log_file_name: str = "accuracy_debugger",
    level: str | int = "INFO",
) -> logging.Logger:
    """Returns a named logger that writes to stderr and to a rotating log file.

    The file is ``<log_file_path>/<log_file_name>_user_info.log``; ``log_file_path`` is created if missing. Handlers
    attached by a previous call are closed and replaced, so a second run never logs into the first run's file.

    Args:
        logger_name: Logger name.
        log_file_path: Directory of the log file.
        log_file_name: Log file name prefix.
        level: Log level name or number.

    Returns:
        The configured logger.
    """
    resolved = resolve_log_level(level)
    log_file_path = Path(log_file_path)
    log_file_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(resolved)
    _remove_marked_handlers(logger)

    _attach(logger, logging.StreamHandler())
    _attach(
        logger,
        logging.handlers.RotatingFileHandler(
            log_file_path / f"{log_file_name}_{LOG_FILE_SUFFIX}",
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf8",
        ),
    )
    print(f"File logs for {logger_name} will be stored at: {log_file_path}\n")
    return logger


def configure_worker_logging(logger_name: str, level: int, log_file: Path | None) -> None:
    """Configures logging in a spawned worker process (``ProcessPoolExecutor`` initializer).

    Idempotent: handlers attached by an earlier call are closed and replaced. The worker logs to stderr and, when
    ``log_file`` is given, appends to that file (opened lazily, no rotation).

    Args:
        logger_name: Name of the logger used by the worker.
        level: Integer logging level.
        log_file: Log file shared with the parent process, or None.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    _remove_marked_handlers(logger)
    _attach(logger, logging.StreamHandler())
    if log_file is not None:
        _attach(logger, logging.FileHandler(log_file, mode="a", encoding="utf8", delay=True))
