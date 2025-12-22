# Step 3: Config + Shared Types + Type Hints

## Overview
Step 3 formalizes configuration and data contracts across the pipeline. It centralizes runtime settings, defines shared dataclasses for analysis outputs, and adds type hints to key modules to make the system easier to maintain and review.

## Configuration (Pydantic Settings)
Runtime settings live in:
- `configs/settings.py` (Pydantic `BaseSettings`)
- `configs/configs.py` (module-level exports)

Key settings include:
- Model paths, batch sizes, detection confidence
- Output and stub directories
- Court overlay image path (`COURT_IMAGE_PATH`)
- Action recognition clip settings, possession thresholds, tactical view dimensions, and output FPS
- Environment variable overrides via `COURTVISION_*`

## Shared Data Models
Structured results are defined in:
- `video_analysis/types.py`

Dataclasses added:
- `EventTimeline`
- `TeamControlRatio`
- `PlayerMetric`
- `AnalysisResult`

These serve as the shared schema between pipeline stages and output JSON.

## Type Hints Coverage
Type hints were added to inputs/outputs for:
- `trackers/*`
- `ball_aquisition/ball_aquisition_detector.py`
- `ball_event_detector/ball_event_detector.py`
- `arena_mark_detector/arena_mark_detector.py`
- `team_classifier/team_classifier.py`
- `action_recognition/action_recognition.py`
- `perspective_transformer/perspective_transformer.py`
- `drawers/*`
- `utils/*`
- `video_analysis/video_analysis.py`

## Validation (CREATE HPC)
Settings and overrides:
```bash
python - <<'PY'
from configs.settings import get_settings
s = get_settings()
print("court_image_path:", s.court_image_path)
print("stubs_dir:", s.stubs_dir)
print("output_dir:", s.output_dir)
print("detection_batch_size:", s.detection_batch_size)
print("detection_confidence:", s.detection_confidence)
PY
```

```bash
COURTVISION_COURT_IMAGE_PATH=images/basketball_court.png \
COURTVISION_DETECTION_BATCH_SIZE=7 \
python - <<'PY'
from configs.settings import get_settings
s = get_settings()
print("court_image_path:", s.court_image_path)
print("detection_batch_size:", s.detection_batch_size)
PY
```

Import sanity check:
```bash
python - <<'PY'
import importlib
modules = [
    'configs.settings',
    'configs.configs',
    'arena_mark_detector.arena_mark_detector',
    'ball_event_detector.ball_event_detector',
    'ball_aquisition.ball_aquisition_detector',
    'trackers.ball_tracker',
    'trackers.player_tracker',
    'trackers.deepsort_player_tracker',
    'team_classifier.team_classifier',
    'action_recognition.action_recognition',
    'perspective_transformer.perspective_transformer',
    'drawers.player_tracks_drawer',
    'drawers.ball_tracks_drawer',
    'drawers.ball_event_drawer',
    'drawers.perspective_overlay_drawer',
    'drawers.arena_mark_drawer',
    'drawers.trajectory_kinetics_drawer',
    'drawers.frame_number_drawer',
    'drawers.action_recognition_drawer',
    'drawers.team_ball_control_drawer',
    'utils.bbox_utils',
    'utils.video_utils',
    'utils.stubs_utils',
]
for name in modules:
    importlib.import_module(name)
print("ok")
PY
```

End-to-end check (API job runs to completion and produces outputs):
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H 'Content-Type: application/json' \
  -d '{
    "input_video_path": "input_videos/test_2.mp4",
    "output_video": "output_videos/test_result_2.mp4",
    "stub_path": "stubs/test_2_run",
    "use_stubs": true
  }'
```

```bash
curl http://127.0.0.1:8000/status/<job_id>
curl http://127.0.0.1:8000/results/<job_id>
ls -l output_videos/test_result_2.mp4
ls -l output_videos/test_result_2.json
```

## Notes
The settings module depends on `pydantic-settings`. Ensure it is installed in the runtime environment.
