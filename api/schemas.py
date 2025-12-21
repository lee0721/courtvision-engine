from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    input_video_path: Optional[str] = Field(
        default=None,
        description="Local path to an input video file."
    )
    input_video_url: Optional[str] = Field(
        default=None,
        description="URL to an input video file."
    )
    output_video: Optional[str] = Field(
        default=None,
        description="Output video path. Defaults to output_videos/<job_id>.mp4"
    )
    stub_path: Optional[str] = Field(
        default=None,
        description="Stub cache directory. Defaults to stubs/<job_id>"
    )
    use_stubs: bool = Field(
        default=True,
        description="Whether to read/write cached stub files."
    )


class AnalysisResponse(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    runtime_ms: Optional[float] = None
    error_message: Optional[str] = None
    input_video_path: Optional[str] = None
    input_video_url: Optional[str] = None
    output_video_path: Optional[str] = None


class JobsResponse(BaseModel):
    count: int
    jobs: list[JobSummary]


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    progress: Optional[float] = None
    runtime_ms: Optional[float] = None
    worker_host: Optional[str] = None
    error_message: Optional[str] = None
    input_video_path: Optional[str] = None
    input_video_url: Optional[str] = None
    output_video_path: Optional[str] = None
    result_json_path: Optional[str] = None


class ResultsResponse(StatusResponse):
    pass
