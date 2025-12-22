from __future__ import annotations

import logging
import os
from typing import Any, Iterable


def setup_logging(log_level: str, log_file: str | None = None) -> None:
    log_level = log_level.upper()
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(log_level)
        if log_file:
            log_file = os.path.abspath(log_file)
            if not any(
                isinstance(handler, logging.FileHandler)
                and getattr(handler, "baseFilename", None) == log_file
                for handler in root_logger.handlers
            ):
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
                )
                root_logger.addHandler(file_handler)
        return

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )


def log_kv(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    kv_pairs: Iterable[str] = (
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    kv_text = " ".join(kv_pairs)
    if kv_text:
        message = f"{message} {kv_text}"
    logger.log(level, message)
