from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from pydantic import BaseModel

from .schemas import JobStatus


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    input_video_path: Optional[str] = None
    input_video_url: Optional[str] = None
    output_video_path: Optional[str] = None
    stub_path: Optional[str] = None
    use_stubs: Optional[bool] = None
    result_json_path: Optional[str] = None
    request_payload: Optional[dict] = None
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    progress: Optional[float] = None
    runtime_ms: Optional[float] = None
    worker_host: Optional[str] = None
    error_message: Optional[str] = None


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def create_job(self, job: JobRecord) -> JobRecord:
        payload = self._serialize_job(job)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id,
                    status,
                    input_video_path,
                    input_video_url,
                    output_video_path,
                    stub_path,
                    use_stubs,
                    result_json_path,
                    request_payload,
                    submitted_at,
                    started_at,
                    completed_at,
                    updated_at,
                    progress,
                    runtime_ms,
                    worker_host,
                    error_message
                )
                VALUES (
                    :job_id,
                    :status,
                    :input_video_path,
                    :input_video_url,
                    :output_video_path,
                    :stub_path,
                    :use_stubs,
                    :result_json_path,
                    :request_payload,
                    :submitted_at,
                    :started_at,
                    :completed_at,
                    :updated_at,
                    :progress,
                    :runtime_ms,
                    :worker_host,
                    :error_message
                )
                """,
                payload,
            )
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return self._deserialize_job(row)

    def update_job(self, job_id: str, **updates) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"job_id not found: {job_id}")

        if "updated_at" not in updates:
            updates["updated_at"] = datetime.utcnow()

        updated_job = self._merge_job(job, updates)
        payload = self._serialize_job(updated_job)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = :status,
                    input_video_path = :input_video_path,
                    input_video_url = :input_video_url,
                    output_video_path = :output_video_path,
                    stub_path = :stub_path,
                    use_stubs = :use_stubs,
                    result_json_path = :result_json_path,
                    request_payload = :request_payload,
                    submitted_at = :submitted_at,
                    started_at = :started_at,
                    completed_at = :completed_at,
                    updated_at = :updated_at,
                    progress = :progress,
                    runtime_ms = :runtime_ms,
                    worker_host = :worker_host,
                    error_message = :error_message
                WHERE job_id = :job_id
                """,
                payload,
            )
        return updated_job

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input_video_path TEXT,
                    input_video_url TEXT,
                    output_video_path TEXT,
                    stub_path TEXT,
                    use_stubs INTEGER,
                    result_json_path TEXT,
                    request_payload TEXT,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT,
                    progress REAL,
                    runtime_ms REAL,
                    worker_host TEXT,
                    error_message TEXT
                )
                """
            )
            self._ensure_columns(conn)

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> list[JobRecord]:
        query = "SELECT * FROM jobs"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._deserialize_job(row) for row in rows]

    def mark_running_as_failed(self, error_message: str) -> int:
        now = datetime.utcnow()
        updated_at = now.isoformat()
        completed_at = now.isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, error_message = ?, updated_at = ?, completed_at = ?
                WHERE status = ?
                """,
                (
                    JobStatus.FAILED.value,
                    error_message,
                    updated_at,
                    completed_at,
                    JobStatus.RUNNING.value,
                ),
            )
            return cursor.rowcount

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        columns = {
            "stub_path": "TEXT",
            "use_stubs": "INTEGER",
            "request_payload": "TEXT",
            "runtime_ms": "REAL",
            "worker_host": "TEXT",
            "completed_at": "TEXT",
        }
        for column_name, column_type in columns.items():
            if column_name not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} {column_type}")

    def _merge_job(self, job: JobRecord, updates: dict) -> JobRecord:
        data = self._dump_job(job)
        data.update(updates)
        return JobRecord(**data)

    def _serialize_job(self, job: JobRecord) -> dict:
        data = self._dump_job(job)
        return {
            **data,
            "status": job.status.value,
            "use_stubs": self._serialize_bool(job.use_stubs),
            "request_payload": self._serialize_json(job.request_payload),
            "submitted_at": self._serialize_datetime(job.submitted_at),
            "started_at": self._serialize_datetime(job.started_at),
            "completed_at": self._serialize_datetime(job.completed_at),
            "updated_at": self._serialize_datetime(job.updated_at),
        }

    def _deserialize_job(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            input_video_path=row["input_video_path"],
            input_video_url=row["input_video_url"],
            output_video_path=row["output_video_path"],
            stub_path=row["stub_path"],
            use_stubs=self._deserialize_bool(row["use_stubs"]),
            result_json_path=row["result_json_path"],
            request_payload=self._deserialize_json(row["request_payload"]),
            submitted_at=self._deserialize_datetime(row["submitted_at"]),
            started_at=self._deserialize_datetime(row["started_at"]),
            completed_at=self._deserialize_datetime(row["completed_at"]),
            updated_at=self._deserialize_datetime(row["updated_at"]),
            progress=row["progress"],
            runtime_ms=row["runtime_ms"],
            worker_host=row["worker_host"],
            error_message=row["error_message"],
        )

    def _serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value else None

    def _deserialize_datetime(self, value: Optional[str]) -> Optional[datetime]:
        return datetime.fromisoformat(value) if value else None

    def _deserialize_bool(self, value: Optional[int]) -> Optional[bool]:
        if value is None:
            return None
        return bool(value)

    def _serialize_bool(self, value: Optional[bool]) -> Optional[int]:
        if value is None:
            return None
        return 1 if value else 0

    def _deserialize_json(self, value: Optional[str]) -> Optional[dict]:
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def _serialize_json(self, value: Optional[dict]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=True)

    def _dump_job(self, job: JobRecord) -> dict:
        if hasattr(job, "model_dump"):
            return job.model_dump()
        return job.dict()
