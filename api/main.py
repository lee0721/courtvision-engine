from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from .executor import BackgroundExecutor
from .jobs import JobRecord, JobStore
from .schemas import (
    AnalysisRequest,
    AnalysisResponse,
    JobStatus,
    ResultsResponse,
    StatusResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output_videos"
DEFAULT_STUB_DIR = REPO_ROOT / "stubs"
JOB_DB_PATH = DEFAULT_STUB_DIR / "jobs.db"
LOG_LEVEL = os.getenv("COURTVISION_LOG_LEVEL", "INFO").upper()

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

TAGS_METADATA = [
    {"name": "Analysis", "description": "Submit analysis jobs."},
    {"name": "Jobs", "description": "Check job status and results."},
]

app = FastAPI(
    title="CourtVision Engine API",
    version="0.1.0",
    openapi_tags=TAGS_METADATA,
)

job_store = JobStore(JOB_DB_PATH)
executor = BackgroundExecutor(job_store)
logger = logging.getLogger("courtvision.api")
logger.setLevel(LOG_LEVEL)


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return jsonable_encoder(model)


@app.on_event("startup")
def mark_interrupted_jobs() -> None:
    count = job_store.mark_running_as_failed(
        "Interrupted by API restart. Retry with POST /jobs/{job_id}/retry."
    )
    if count:
        logger.info("Marked %s running job(s) as failed after restart.", count)


@app.exception_handler(HTTPException)
def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "HTTP %s path=%s detail=%s",
        exc.status_code,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error path=%s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/analysis", response_model=AnalysisResponse, status_code=202, tags=["Analysis"])
def submit_analysis(request: AnalysisRequest) -> AnalysisResponse:
    if not request.input_video_path and not request.input_video_url:
        logger.warning("Rejecting analysis request: missing input_video_path/input_video_url")
        raise HTTPException(status_code=400, detail="input_video_path or input_video_url is required")

    if request.input_video_path:
        input_path = _resolve_path(request.input_video_path)
        if not input_path.exists():
            logger.warning(
                "Rejecting analysis request: input_video_path missing path=%s",
                request.input_video_path,
            )
            raise HTTPException(
                status_code=400,
                detail=f"input_video_path does not exist: {request.input_video_path}",
            )

    job_id = str(uuid4())
    submitted_at = datetime.utcnow()

    if request.output_video:
        output_video_path = _resolve_path(request.output_video)
    else:
        output_video_path = DEFAULT_OUTPUT_DIR / f"{job_id}.mp4"

    if request.stub_path:
        stub_path = _resolve_path(request.stub_path)
    else:
        stub_path = DEFAULT_STUB_DIR / job_id

    result_json_path = output_video_path.with_suffix(".json")

    job = JobRecord(
        job_id=job_id,
        status=JobStatus.QUEUED,
        input_video_path=request.input_video_path,
        input_video_url=request.input_video_url,
        output_video_path=str(output_video_path),
        stub_path=str(stub_path),
        use_stubs=request.use_stubs,
        result_json_path=str(result_json_path),
        submitted_at=submitted_at,
    )
    job_store.create_job(job)

    executor.submit(job_id, request, output_video_path, stub_path, result_json_path)
    logger.info(
        "job_id=%s status=queued input_path=%s input_url=%s output_video=%s use_stubs=%s",
        job_id,
        request.input_video_path,
        request.input_video_url,
        output_video_path,
        request.use_stubs,
    )

    return AnalysisResponse(job_id=job_id, status=JobStatus.QUEUED, submitted_at=submitted_at)


@app.get("/status/{job_id}", response_model=StatusResponse, tags=["Jobs"])
def get_status(job_id: str) -> StatusResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    logger.debug("job_id=%s status=%s", job_id, job.status)
    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        updated_at=job.updated_at,
        progress=job.progress,
        error_message=job.error_message,
    )


@app.get("/results/{job_id}", response_model=Union[ResultsResponse, StatusResponse], tags=["Jobs"])
def get_results(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    if job.status != JobStatus.COMPLETED:
        logger.debug("job_id=%s status=%s results not ready", job_id, job.status)
        payload = StatusResponse(
            job_id=job.job_id,
            status=job.status,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            updated_at=job.updated_at,
            progress=job.progress,
            error_message=job.error_message,
        )
        return JSONResponse(status_code=202, content=_model_dump(payload))

    return ResultsResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        updated_at=job.updated_at,
        progress=job.progress,
        error_message=job.error_message,
        output_video_path=job.output_video_path,
        result_json_path=job.result_json_path,
    )


@app.post("/jobs/{job_id}/retry", response_model=StatusResponse, tags=["Jobs"])
def retry_job(job_id: str) -> StatusResponse:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="job already completed")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="job is currently running")

    if job.input_video_path:
        input_path = _resolve_path(job.input_video_path)
        if not input_path.exists():
            logger.warning(
                "Rejecting retry request: input_video_path missing path=%s",
                job.input_video_path,
            )
            raise HTTPException(
                status_code=400,
                detail=f"input_video_path does not exist: {job.input_video_path}",
            )

    output_video_path = (
        _resolve_path(job.output_video_path)
        if job.output_video_path
        else DEFAULT_OUTPUT_DIR / f"{job_id}.mp4"
    )
    stub_path = (
        _resolve_path(job.stub_path)
        if job.stub_path
        else DEFAULT_STUB_DIR / job_id
    )
    result_json_path = (
        _resolve_path(job.result_json_path)
        if job.result_json_path
        else output_video_path.with_suffix(".json")
    )

    request = AnalysisRequest(
        input_video_path=job.input_video_path,
        input_video_url=job.input_video_url,
        output_video=str(output_video_path),
        stub_path=str(stub_path),
        use_stubs=job.use_stubs if job.use_stubs is not None else True,
    )

    updated_job = job_store.update_job(
        job_id,
        status=JobStatus.QUEUED,
        error_message=None,
        progress=None,
        started_at=None,
    )
    executor.submit(job_id, request, output_video_path, stub_path, result_json_path)
    logger.info("job_id=%s status=queued retry=true", job_id)

    return StatusResponse(
        job_id=updated_job.job_id,
        status=updated_job.status,
        submitted_at=updated_job.submitted_at,
        started_at=updated_job.started_at,
        updated_at=updated_job.updated_at,
        progress=updated_job.progress,
        error_message=updated_job.error_message,
    )
