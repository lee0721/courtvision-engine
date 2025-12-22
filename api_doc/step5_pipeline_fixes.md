# Step 5: Pipeline Correctness Fixes

## Overview
Step 5 focuses on correctness, performance, and stability in the core pipeline. It does not
change model outputs. It removes redundant work, centralizes inference parameters, and
ensures safe video I/O handling.

## What Was Addressed
1) Remove duplicate player tracking calls
- Goal: avoid running YOLO twice on the same frames.
- Status: `VideoAnalysis.run()` calls `player_tracker.get_object_tracks()` once.

2) Parameterize batch size and confidence
- Goal: move inference settings out of hard-coded values.
- Status: `DETECTION_BATCH_SIZE` and `DETECTION_CONFIDENCE` are loaded from
  `configs/settings.py` via `configs/configs.py` and used in:
  - `trackers/player_tracker.py`
  - `trackers/ball_tracker.py`
  - `trackers/deepsort_player_tracker.py`
  - `arena_mark_detector/arena_mark_detector.py`

3) Safe video I/O (read, empty frame handling, release)
- Goal: fail fast with clear errors and always release resources.
- Status: `utils/video_utils.py` checks open/read/write, logs errors,
  handles empty frames, and releases resources.

## Quick Verification (Optional)
Check tracking is called once:
```bash
rg "player_tracker\.get_object_tracks" video_analysis/video_analysis.py
```

Check parameterized settings usage:
```bash
rg "DETECTION_BATCH_SIZE|DETECTION_CONFIDENCE" trackers arena_mark_detector
```

Check video I/O safety:
```bash
rg "video_open_failed|empty_frame|video_write_failed" utils/video_utils.py
```

## Notes
- Training notebooks are not part of the Step 5 scope.
- These fixes build on Step 3 (config) and Step 4 (logging/observability).
