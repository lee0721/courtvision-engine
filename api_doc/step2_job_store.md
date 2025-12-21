# Step 2: Job Metadata Store (SQLite)

## Overview
Step 2 adds persistent job metadata storage so jobs survive API restarts and can be queried later. A lightweight SQLite database is used to keep the system simple while demonstrating backend patterns (state persistence, queryability, traceability).

## Data Model
Jobs are stored in a single `jobs` table with fields that capture inputs, status, timing, and errors:
- Core fields: `job_id`, `status`, `input_video_path`, `input_video_url`, `output_video_path`, `stub_path`, `use_stubs`, `result_json_path`.
- Metadata: `request_payload` (JSON string), `submitted_at`, `started_at`, `completed_at`, `updated_at`.
- Execution: `progress`, `runtime_ms`, `worker_host`, `error_message`.

## Storage Layer
The API uses a small repository that:
- Ensures the `jobs` table exists and adds new columns if needed.
- Creates job records at submission time.
- Updates status transitions and timing fields.
- Lists jobs by status with a limit for dashboards or admin tools.

## API Integration
The service layer now reads and writes jobs from the database:
- `POST /analysis` creates a job record (including `request_payload`).
- `GET /status/{job_id}` and `GET /results/{job_id}` read status from SQLite.
- `GET /jobs?status=...&limit=...` lists recent jobs.
- `POST /jobs/{job_id}/retry` resets timing fields and requeues when allowed.

## Persistence Location
The database lives at:
- `data/jobs.db`

This path keeps runtime artifacts separate from source code and outputs.

## Validation (CREATE HPC)
Verified with:
- Schema check: `sqlite3 data/jobs.db ".schema jobs"`.
- API list endpoint: `GET /jobs?status=completed&limit=5`.
- DB query: `select job_id, status, runtime_ms, worker_host, request_payload from jobs`.
