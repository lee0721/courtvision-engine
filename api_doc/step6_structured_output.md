# Step 6: Structured Output (analysis_result.json)

## Overview
Step 6 produces a structured JSON result alongside the output video and exposes it
through the API. This makes the analysis consumable by frontend and downstream systems.

## What Was Added
- CLI writes `<output_video>.json` after analysis completes.
- API executor writes `result_json_path` and persists the JSON.
- `GET /results/{job_id}` includes the JSON under `result`.

## Result Schema
Core fields:
- `input_video` (str)
- `output_video` (str)
- `frame_count` (int)
- `events` (object)
  - `passes` (list[int])
  - `interceptions` (list[int])
- `ball_possession` (list[int])
- `team_ball_control_ratio` (object)
  - `team_1` (float)
  - `team_2` (float)
  - `none` (float)
- `player_metrics` (object keyed by player_id)
  - `total_distance_m` (float)
  - `avg_speed_kmh` (float)
  - `max_speed_kmh` (float)

API-only metadata (added by executor):
- `job_id`
- `generated_at` (ISO8601)

## Files and Paths
- CLI: `output_videos/<name>.mp4` + `output_videos/<name>.json`
- API: `output_videos/<name>.mp4` + `output_videos/<name>.json`
- Job status includes `result_json_path`.

## Validation (CREATE HPC)
Note: `rg` is optional. If it is not installed, use `grep`.

### 1) CLI: JSON output file
```bash
conda activate bball
export COURTVISION_LOG_LEVEL=INFO
export COURTVISION_LOG_FILE=logs/step6-cli.log
mkdir -p logs

python main.py input_videos/test_2.mp4 \
  --output_video output_videos/step6_cli.mp4 \
  --stub_path stubs/step6_cli_run
```

Check artifacts:
```bash
ls -l output_videos/step6_cli.mp4 output_videos/step6_cli.json
```

Check keys:
```bash
python - <<'PY'
import json
with open("output_videos/step6_cli.json") as f:
    data = json.load(f)
print(data.keys())
print(data["events"].keys())
print("players:", len(data["player_metrics"]))
PY
```

### 2) API: /results returns JSON
Terminal A (API server):
```bash
export COURTVISION_LOG_LEVEL=INFO
export COURTVISION_LOG_FILE=logs/step6-api.log
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Terminal B (same GPU job):
```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "input_video_path": "input_videos/test_2.mp4",
    "output_video": "output_videos/step6_api.mp4",
    "stub_path": "stubs/step6_cli_run",
    "use_stubs": true
  }'
```

```bash
curl http://127.0.0.1:8000/status/<job_id>
```

```bash
curl http://127.0.0.1:8000/results/<job_id> | python -m json.tool | head -n 40
ls -l output_videos/step6_api.json output_videos/step6_api.mp4
```

### 3) Consistency check (optional)
```bash
python - <<'PY'
import json
with open("output_videos/step6_api.json") as f:
    file_result = json.load(f)
print(file_result.keys())
PY
```

## Notes
- If JSON serialization fails, make sure you pulled the latest Step 6 changes.
- Reusing `stub_path` makes reruns much faster.
