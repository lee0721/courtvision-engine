# CourtVision Engine 🏀

CourtVision Engine is a computer-vision toolkit for breaking down half-court basketball footage. The
pipeline combines multi-object tracking, jersey-based team classification,
ball-possession heuristics, tactical-view projection, and action recognition to
produce an annotated video that surfaces passes, interceptions, and per-player
movement metrics.

## Project Overview 🎯
[🎥 Explanation video](https://drive.google.com/file/d/1-hqNvQog2tOV4v0bxwxdczk2CtINs8SM/view?usp=sharing)

## Features ✨
- **Player & ball tracking** – YOLOv8 detections paired with ByteTrack keep
  persistent IDs for every player while a separate detector follows the ball,
  filters false positives, and interpolates missing frames.
- **Team classification with CLIP** – automatically assigns each tracked player
  to a jersey label (e.g., `dark blue shirt` vs `white shirt`) so downstream
  overlays can stay colour consistent.
- **Possession and event detection** – heuristics determine who controls the
  ball, then emit pass and interception events in real time.
- **Court keypoint extraction** – detects court markings, validates them, and
  builds the homography needed to project activity onto a tactical top-down
  map.
- **Trajectory kinetics** – converts projected positions into distance (m) and
  speed (km/h) metrics per player over time.
- **Action recognition** – R(2+1)D model predicts player actions on cropped
  clips for richer context.
- **Rich overlays** – drawers layer in tracks, team colours, frame numbers,
  ball-control banners, tactical insets, kinetics charts, and action labels for
  review-friendly output videos.
- **Stub caching** – every heavy module can persist intermediate results
  (detections, classifications, predictions) as pickled stubs so subsequent runs
  iterate quickly.

## Pipeline at a Glance 🧭
```
Video Frames
   │
   ├── YOLO / ByteTrack → player_tracks
   ├── YOLO (ball)      → ball_tracks → filtering → interpolation
   │
   ├── CLIP classifier      → team assignments
   ├── Ball possession      → passes / interceptions
   ├── Court mark detector  → homography / tactical positions
   ├── R(2+1)D action model → action labels
   └── Trajectory kinetics  → per-player distance & speed
         ↓
    Drawers overlay all artefacts → rendered analysis video
```

## Service Flow (API) 🔌
```
Client → POST /analysis → JobStore (SQLite) → BackgroundExecutor → VideoAnalysis
   → output_videos/*.mp4 + output_videos/*.json → GET /results/{job_id}
```

## Performance (Benchmarks) ⚡
- End-to-end pipeline: cold 536299.55 ms → warm 10995.06 ms (48.78x faster, 97.9% time saved).
- Throughput: cold 0.34 FPS, warm 16.37 FPS (frame_count=180).
- API latency: `POST /analysis` enqueue/response `time_total` avg 0.0419 s (5 requests, 202 Accepted; not full pipeline time).

## Repository Layout 🗂️
- `main.py` – CLI entry point for running a full analysis on a source video.
- `video_analysis/` – orchestrates the end-to-end pipeline.
- `trackers/` – YOLO-based player and ball detectors plus ByteTrack wrappers.
- `team_classifier/` – CLIP-powered jersey classification and colour mapping.
- `ball_aquisition/` & `ball_event_detector/` – possession, passes, and interceptions.
- `arena_mark_detector/` & `perspective_transformer/` – keypoint detection and tactical projection.
- `trajectory_kinetics_analyzer/` – distance and speed calculations.
- `action_recognition/` – R(2+1)D model wrapper for player action inference.
- `drawers/` – visualization components that paint tracks, scoreboards, tactical insets, etc.
- `utils/` – video IO, bbox helpers, and the stub caching utilities.
- `training_notebook/` – exploratory notebook for model experimentation.

## Getting Started

### Prerequisites
- Python 3.11 (tested); GPU acceleration is recommended for inference speed.
- ffmpeg installed on your system (required by OpenCV when encoding MP4).
- Model weights placed under `models/`:
  - `player_detector.pt` – YOLOv8 weights trained for player detection.
  - `ball_detector_model.pt` – YOLOv8 weights for basketball detection.
  - `arena_mark_detector.pt` – YOLO keypoint detector for court markings.
  - `action_r2plus1d_best.pt` – fine-tuned R(2+1)D action recognition checkpoint.

> Swap in your own weights if you have different training artefacts—paths are
> loaded from `configs/settings.py` and can be overridden via `COURTVISION_*`
> environment variables.

### Installation
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run an Analysis
```bash
python main.py input_videos/sample.mp4 \
  --output_video output_videos/sample_annotated.mp4 \
  --stub_path stubs
```

Flags:
- `--output_video` (optional) chooses where the rendered video is written.
- `--stub_path` (optional) points to a directory for cached intermediate
  results. On the first pass detections are saved; subsequent runs will reuse
  them so that you can iterate on downstream logic without rerunning YOLO.

Outputs:
- `output_videos/…` – annotated game film with all overlays.
- `output_videos/… .json` – structured analysis result for downstream use.
- `stubs/…` – cached pickle files (player tracks, ball tracks, team assignments,
  action predictions, etc.) to accelerate future runs.

### Structured Output (analysis_result.json)
Each run writes a JSON file next to the output video:
- `input_video`
- `output_video`
- `frame_count`
- `events` (`passes`, `interceptions`)
- `ball_possession`
- `team_ball_control_ratio`
- `player_metrics` (distance and speed per player)

### API Service (FastAPI)
Start the service:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /analysis` – submit a job
- `GET /status/{job_id}` – job status
- `GET /results/{job_id}` – job status + JSON result (when ready)
- `GET /jobs` – list recent jobs
- `POST /jobs/{job_id}/retry` – retry a failed job

Example request:
```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "input_video_path": "input_videos/sample.mp4",
    "output_video": "output_videos/sample_api.mp4",
    "stub_path": "stubs/sample_api_run",
    "use_stubs": true
  }'
```

Example results:
```bash
curl http://127.0.0.1:8000/status/<job_id>
curl http://127.0.0.1:8000/results/<job_id>
```

Swagger UI: `http://127.0.0.1:8000/docs`

### Stub Cache Layout
Stub files are stored under the `stub_path` directory:
- `player_track_stubs.pkl`
- `ball_track_stubs.pkl`
- `court_key_points_stub.pkl`
- `player_assignment_stub.pkl`
- `action_recognition_predictions.pkl`

### Testing
```bash
python -m pytest
```

### Configuration (Environment Variables)
All settings are read from `configs/settings.py` with the `COURTVISION_` prefix.
Common overrides:
- `COURTVISION_OUTPUT_DIR`, `COURTVISION_STUBS_DIR`, `COURTVISION_DATA_DIR`
- `COURTVISION_JOBS_DB_PATH`
- `COURTVISION_PLAYER_DETECTOR_PATH`, `COURTVISION_BALL_DETECTOR_PATH`
- `COURTVISION_ARENA_MARK_DETECTOR_PATH`, `COURTVISION_ACTION_RECOGNITION_MODEL_PATH`
- `COURTVISION_DETECTION_BATCH_SIZE`, `COURTVISION_DETECTION_CONFIDENCE`
- `COURTVISION_ACTION_CLIP_LEN`, `COURTVISION_ACTION_STRIDE`
- `COURTVISION_BALL_POSSESSION_MIN_FRAMES`, `COURTVISION_BALL_POSSESSION_THRESHOLD_PX`
- `COURTVISION_SPEED_WINDOW_SIZE`, `COURTVISION_ANALYSIS_FPS`
- `COURTVISION_LOG_LEVEL`, `COURTVISION_LOG_FILE`, `COURTVISION_ACTION_DEVICE`

## Extending the Project
- **Training** – use the notebooks under `training_notebook/` or create new ones
  to refine detectors and the action model.
- **New events** – extend `ball_event_detector` to recognise screens, rebounds,
  or turnovers; all team and possession data is available per frame.
- **Analytics exports** – hook into the per-frame dictionaries before the
  drawers run and dump JSON or CSV summaries for BI dashboards.
- **Alternative sports** – swap model weights, update court dimensions in the
  perspective transformer, and adjust event heuristics to adapt to other
  invasion sports.

## Responsible Usage
The repo processes broadcast footage and generates advanced tracking data. If
you apply it to competitions, make sure usage complies with local league rules,
player privacy requirements, and data governance guidelines.
