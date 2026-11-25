"""Runtime Logging and Crash Diagnostic Module for Onmyoji Tactical Helper.

Configures rotating file logging and optional global unhandled exception hooks for both
the main process and background worker threads without import-time side effects.
"""

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Any

from onmyoji_helper.infra.paths import get_app_base_dir

LOGGER_NAME: str = "OnmyojiHelper"


def setup_runtime_logger(
    log_dir: str = "logs",
    log_filename: str = "app_runtime.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    log_level: int = logging.INFO,
    install_hooks: bool = True,
) -> logging.Logger:
    """Initializes the rotating file logger and optionally installs global exception hooks.

    Args:
        log_dir: Relative or absolute directory where log files are stored.
        log_filename: Name of the rotating log file.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of rotated backup files to retain.
        log_level: Logging severity threshold.
        install_hooks: Whether to attach `sys.excepthook` and `threading.excepthook`
            (keep False during unit tests so pytest manages exception capture).

    Returns:
        Configured application logger instance.
    """
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(get_app_base_dir(), log_dir)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except Exception as exc:
            if sys.stdout is not None:
                print(f"⚠️ Failed to create rotating file handler: {exc}")

        if sys.stdout is not None:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)

    if install_hooks:
        def _global_excepthook(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.critical(
                "💥 [FATAL CRASH] Uncaught exception in main process:",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

        sys.excepthook = _global_excepthook

        if hasattr(threading, "excepthook"):
            def _thread_excepthook(args: Any) -> None:
                if issubclass(args.exc_type, KeyboardInterrupt):
                    return
                thread_name = args.thread.name if args.thread else "UnknownThread"
                logger.critical(
                    f"💥 [THREAD CRASH] Uncaught exception in thread [{thread_name}]:",
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )

            threading.excepthook = _thread_excepthook

    return logger
