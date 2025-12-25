from datetime import datetime, timedelta

from api.jobs import JobRecord, JobStore
from api.schemas import JobStatus


def _make_job(job_id: str, status: JobStatus = JobStatus.QUEUED, submitted_at: datetime | None = None) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        status=status,
        submitted_at=submitted_at or datetime.utcnow(),
    )


def test_job_store_accepts_directory_path(tmp_path):
    db_dir = tmp_path / "dbdir"
    db_dir.mkdir()

    store = JobStore(db_dir)

    expected_path = (db_dir / "jobs.db").resolve()
    assert store._db_path == expected_path
    assert expected_path.exists()


def test_job_store_create_get_update_and_mark_failed(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    job = _make_job("job-1")
    store.create_job(job)

    fetched = store.get_job(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id
    assert fetched.status == JobStatus.QUEUED

    updated = store.update_job(job.job_id, status=JobStatus.RUNNING, started_at=job.submitted_at)
    assert updated.status == JobStatus.RUNNING

    count = store.mark_running_as_failed("boom")
    assert count == 1

    failed = store.get_job(job.job_id)
    assert failed is not None
    assert failed.status == JobStatus.FAILED
    assert failed.error_message == "boom"
    assert failed.completed_at is not None
    assert failed.updated_at is not None


def test_list_jobs_sorted_by_submitted_at(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    older = _make_job("older", submitted_at=datetime.utcnow() - timedelta(seconds=1))
    newer = _make_job("newer", submitted_at=datetime.utcnow())

    store.create_job(older)
    store.create_job(newer)

    jobs = store.list_jobs()

    assert [job.job_id for job in jobs] == ["newer", "older"]
