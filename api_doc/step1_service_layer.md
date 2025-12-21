# Step 1: Service Layer (FastAPI)

## Overview
Step 1 adds a minimal service layer on top of the existing CourtVision pipeline so the system looks and behaves like a backend service rather than a local script. The core analysis flow is unchanged; it is simply wrapped by an API.

## API Contract
Endpoints are defined with Pydantic models and exposed via OpenAPI:
- `POST /analysis` submits a job and returns a `job_id` immediately.
- `GET /status/{job_id}` returns job state (`queued`, `running`, `completed`, `failed`).
- `GET /results/{job_id}` returns output paths when complete, or a 202 status if still running.

The request/response schema is self-documented via `/docs` and `/openapi.json`.

## Execution Model
Jobs are executed asynchronously in a background thread:
- The API accepts a request and enqueues work with a lightweight executor.
- `VideoAnalysis.run()` is invoked inside the background task.
- Output artifacts are written without modifying the core pipeline logic.

## Observability and Errors
Basic logging and error handling are included:
- Structured logs include `job_id`, status transitions, and elapsed time.
- Invalid inputs return 4xx with clear messages.
- Unhandled errors return 500 with a stable response shape.

## Output Artifacts
Each job produces:
- A processed video (`*.mp4`).
- A structured JSON result (`*.json`) including events, possession, and player metrics.

Example artifacts produced on CREATE HPC:
- `output_videos/test_2_api.mp4`
- `output_videos/test_2_api.json`

## HPC Validation
The API was validated on CREATE HPC using a GPU node with SSH tunneling:
- FastAPI service started on the compute node.
- Requests were submitted from the local machine via SSH tunnel.
- Jobs completed with status updates and generated output artifacts.

## Why This Matters for the MSc
This step demonstrates:
- A clean API surface that supports collaboration with frontend/product teams.
- Non-blocking job execution suitable for long-running workloads.
- Engineering practices (documentation, error handling, logging) aligned with backend SWE expectations.
