# Step 4: Logging + Observability

## Overview
Step 4 adds structured logging, stage timing, and safer I/O error handling across the pipeline.
It does not change analysis results. It makes execution observable, debuggable, and easier to operate.

## What Was Added
Logging setup:
- `utils/logging_utils.py` (centralized logging + key/value helper)
- `configs/settings.py` and `configs/configs.py` (log level + optional log file)

Structured fields (key=value):
- `job_id`
- `stage`
- `frame_id` (when relevant)
- `cache_hit`
- `duration_ms`

Pipeline timing:
- `VideoAnalysis.run()` logs stage start/end and a final `analysis_complete` summary.

I/O error handling:
- Video open/read/write failures log clearly and raise.
- Stub cache read/write failures log clearly and raise.
- Model load failures log clearly and raise.

Job ID propagation:
- API jobs pass `job_id` into core modules.
- CLI uses `job_id=cli`.

## Stage Names
The pipeline logs these stages:
- `read_video`
- `init_models`
- `tracking`
- `arena_marks`
- `ball_processing`
- `team_assignment`
- `ball_acquisition`
- `ball_events`
- `tactical_view`
- `trajectory_metrics`
- `action_recognition`
- `drawing`
- `save_video`
- `build_result`

## Logging Configuration
Environment variables:
- `COURTVISION_LOG_LEVEL` (default `INFO`)
- `COURTVISION_LOG_FILE` (optional, e.g. `logs/step4-api.log`)

Example:
```bash
export COURTVISION_LOG_LEVEL=INFO
export COURTVISION_LOG_FILE=logs/step4-cli.log
```

## Validation (CREATE HPC)
Note: `rg` is optional. If it is not installed, use `grep`.

### 1) CLI: baseline run (stub miss)
```bash
conda activate bball
export COURTVISION_LOG_LEVEL=INFO
export COURTVISION_LOG_FILE=logs/step4-cli.log
mkdir -p logs

python main.py input_videos/test_2.mp4 \
  --output_video output_videos/step4_test.mp4 \
  --stub_path stubs/step4_test_run
```

Check key logs:
```bash
grep -n "stage_start\|stage_end\|analysis_complete\|stub_" logs/step4-cli.log
```

### 2) CLI: cache hit (stub hit)
```bash
python main.py input_videos/test_2.mp4 \
  --output_video output_videos/step4_test.mp4 \
  --stub_path stubs/step4_test_run
```

```bash
grep -n "stub_hit\|stub_miss" logs/step4-cli.log
```

### 3) DEBUG logs (logger replaces print)
```bash
export COURTVISION_LOG_LEVEL=DEBUG
export COURTVISION_LOG_FILE=logs/step4-cli-debug.log

python main.py input_videos/test_2.mp4 \
  --output_video output_videos/step4_test_debug.mp4 \
  --stub_path stubs/step4_test_run_debug
```

```bash
grep -n "action_clip_summary\|action_clip_count\|action_clip_predictions" logs/step4-cli-debug.log
```

### 4) I/O error handling (intentional failure)
```bash
export COURTVISION_LOG_LEVEL=INFO
export COURTVISION_LOG_FILE=logs/step4-cli.log

python main.py input_videos/not_exist.mp4 \
  --output_video output_videos/step4_bad.mp4 \
  --stub_path stubs/step4_bad_run
```

```bash
grep -n "video_open_failed\|analysis_failed" logs/step4-cli.log
```

### 5) API job_id propagation
Terminal A (API server):
```bash
export COURTVISION_LOG_LEVEL=INFO
export COURTVISION_LOG_FILE=logs/step4-api.log
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Terminal B (same GPU job):
```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "input_video_path": "input_videos/test_2.mp4",
    "output_video": "output_videos/step4_api.mp4",
    "stub_path": "stubs/step4_api_run",
    "use_stubs": true
  }'
```

```bash
curl http://127.0.0.1:8000/status/<job_id>
```

```bash
grep -n "job_id=.*stage_start\|job_id=.*analysis_complete" logs/step4-api.log
```

### 6) Output artifacts
```bash
ls -l output_videos/step4_api.mp4 output_videos/step4_api.json
```

## Notes
- The pipeline should log a final `analysis_complete` entry for CLI and API runs.
- If CUDA is unavailable, a warning may appear, but logging still works.
- Training notebooks are not part of the Step 4 logging scope.
