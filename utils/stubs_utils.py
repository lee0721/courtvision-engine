"""
A module for caching and retrieving computational results to disk.

This module provides utility functions to save and load intermediate processing results,
which helps avoid redundant computations and speeds up development iterations.
"""

from __future__ import annotations

import logging
import os 
import pickle
from typing import Any

from utils.logging_utils import log_kv

logger = logging.getLogger("courtvision.stubs")

def save_stub(
    stub_path: str | None,
    object: Any,
    job_id: str | None = None,
    stage: str | None = None,
) -> None:
    """
    Save a Python object to disk at the specified path.

    Creates necessary directories if they don't exist and serializes the object using pickle.

    Args:
        stub_path (str): File path where the object should be saved.
        object: Any Python object that can be pickled.
    """
    if stub_path is None:
        return

    dir_path = os.path.dirname(stub_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    try:
        with open(stub_path, "wb") as f:
            pickle.dump(object, f)
        log_kv(
            logger,
            logging.DEBUG,
            "stub_write",
            job_id=job_id,
            stage=stage,
            stub_path=stub_path,
        )
    except Exception as exc:  # pragma: no cover - filesystem issues
        log_kv(
            logger,
            logging.ERROR,
            "stub_write_failed",
            job_id=job_id,
            stage=stage,
            stub_path=stub_path,
            error=str(exc),
        )
        raise

def read_stub(
    read_from_stub: bool,
    stub_path: str | None,
    job_id: str | None = None,
    stage: str | None = None,
) -> Any | None:
    """
    Read a previously saved Python object from disk if available.

    Args:
        read_from_stub (bool): Whether to attempt reading from disk.
        stub_path (str): File path where the object was saved.

    Returns:
        object: The loaded Python object if successful, None otherwise.
    """
    if not read_from_stub or stub_path is None:
        return None

    if not os.path.exists(stub_path):
        log_kv(
            logger,
            logging.INFO,
            "stub_miss",
            job_id=job_id,
            stage=stage,
            cache_hit=False,
            stub_path=stub_path,
        )
        return None

    try:
        with open(stub_path, "rb") as f:
            object = pickle.load(f)
        log_kv(
            logger,
            logging.INFO,
            "stub_hit",
            job_id=job_id,
            stage=stage,
            cache_hit=True,
            stub_path=stub_path,
        )
        return object
    except Exception as exc:  # pragma: no cover - corrupt cache
        log_kv(
            logger,
            logging.ERROR,
            "stub_read_failed",
            job_id=job_id,
            stage=stage,
            stub_path=stub_path,
            error=str(exc),
        )
        raise
    return None
    
