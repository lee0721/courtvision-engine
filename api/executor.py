from __future__ import annotations

import json
import logging
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from video_analysis.video_analysis import VideoAnalysis
from utils.logging_utils import log_kv

from .jobs import JobStore
from .schemas import AnalysisRequest, JobStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEOS_DIR = REPO_ROOT / "input_videos"
logger = logging.getLogger("courtvision.executor")

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _download_input_video(input_url: str, job_id: str) -> Path:
    INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(input_url)
    filename = Path(parsed.path).name or f"{job_id}.mp4"
    if not Path(filename).suffix:
        filename = f"{filename}.mp4"
    destination = INPUT_VIDEOS_DIR / filename
    logger.info("job_id=%s downloading input_url=%s destination=%s", job_id, input_url, destination)
    try:
        urlretrieve(input_url, destination)
    except Exception as exc:  # pragma: no cover - network IO
        log_kv(
            logger,
            logging.ERROR,
            "input_download_failed",
            job_id=job_id,
            input_url=input_url,
            destination=str(destination),
            error=str(exc),
        )
        raise
    return destination


def _to_serializable(value):
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(data)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, ensure_ascii=True)
    except Exception as exc:  # pragma: no cover - filesystem issues
        log_kv(
            logger,
            logging.ERROR,
            "result_write_failed",
            result_json_path=str(path),
            error=str(exc),
        )
        raise


class BackgroundExecutor:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store

    def submit(
        self,
        job_id: str,
        request: AnalysisRequest,
        output_video_path: Path,
        stub_path: Path,
        result_json_path: Path,
    ) -> None:
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, request, output_video_path, stub_path, result_json_path),
            daemon=True,
        )
        thread.start()

    def _run_job(
        self,
        job_id: str,
        request: AnalysisRequest,
        output_video_path: Path,
        stub_path: Path,
        result_json_path: Path,
    ) -> None:
        start_time = time.perf_counter()
        worker_host = socket.gethostname()
        self.job_store.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            worker_host=worker_host,
        )
        logger.info("job_id=%s status=running output_video=%s", job_id, output_video_path)
        try:
            if request.input_video_path:
                input_path = _resolve_path(request.input_video_path)
                logger.info("job_id=%s input_path=%s", job_id, input_path)
            elif request.input_video_url:
                input_path = _download_input_video(request.input_video_url, job_id)
            else:
                raise ValueError("input_video_path or input_video_url is required")

            stub_path.mkdir(parents=True, exist_ok=True)

            analyzer = VideoAnalysis(
                input_path=str(input_path),
                output_path=str(output_video_path),
                stub_path=str(stub_path),
                use_stubs=request.use_stubs,
                job_id=job_id,
            )
            results = analyzer.run()
            results = results or {}
            results.setdefault("job_id", job_id)
            results.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
            _write_json(result_json_path, results)

            completed_at = datetime.utcnow()
            elapsed_s = time.perf_counter() - start_time
            self.job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                output_video_path=str(output_video_path),
                result_json_path=str(result_json_path),
                completed_at=completed_at,
                runtime_ms=elapsed_s * 1000,
            )
            logger.info(
                "job_id=%s status=completed elapsed_s=%.2f result_json=%s",
                job_id,
                elapsed_s,
                result_json_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            elapsed_s = time.perf_counter() - start_time
            self.job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                error_message=str(exc),
                completed_at=datetime.utcnow(),
                runtime_ms=elapsed_s * 1000,
            )
            logger.exception("job_id=%s status=failed error=%s", job_id, exc)
