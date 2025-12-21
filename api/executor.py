from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

from video_analysis.video_analysis import VideoAnalysis

from .jobs import JobStore
from .schemas import AnalysisRequest, JobStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEOS_DIR = REPO_ROOT / "input_videos"


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
    urlretrieve(input_url, destination)
    return destination


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


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
        self.job_store.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        try:
            if request.input_video_path:
                input_path = _resolve_path(request.input_video_path)
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
            )
            results = analyzer.run()
            results = results or {}
            results.setdefault("job_id", job_id)
            results.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
            _write_json(result_json_path, results)

            self.job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                output_video_path=str(output_video_path),
                result_json_path=str(result_json_path),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                error_message=str(exc),
            )
