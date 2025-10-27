# Basketball Analysis Pipeline

Computer-vision toolkit for breaking down half-court basketball footage. The
pipeline combines multi-object tracking, jersey-based team classification,
ball-possession heuristics, tactical-view projection, and action recognition to
produce an annotated video that surfaces passes, interceptions, and per-player
movement metrics.

## Features
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

## Pipeline at a Glance
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

## Repository Layout
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

> Swap in your own weights if you have different training artefacts—just update
> the paths inside `configs/configs.py`.

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
- `stubs/…` – cached pickle files (player tracks, ball tracks, team assignments,
  action predictions, etc.) to accelerate future runs.

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

