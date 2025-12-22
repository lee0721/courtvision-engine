# Step 7: Tests (pytest + fixtures)

## Overview
Step 7 adds a minimal test suite to validate core logic and a stubbed integration flow.
This improves reliability and maintainability without requiring full model execution.

## What Was Added
- `pytest.ini` to configure test discovery (`tests/`).
- Shared fixtures in `tests/conftest.py`.
- Unit tests for:
  - Ball possession logic (`ball_aquisition`).
  - Pass/interception detection (`ball_event_detector`).
  - Distance/speed metrics (`trajectory_kinetics_analyzer`).
- A stubbed integration test for `VideoAnalysis.run()` that verifies output keys and file output.

## Test Files
- `tests/test_ball_aquisition_detector.py`
- `tests/test_ball_event_detector.py`
- `tests/test_trajectory_kinetics_analyzer.py`
- `tests/test_video_analysis_integration.py`

## How to Run
```bash
python -m pytest
```

Optional:
```bash
python -m pytest -q
python -m pytest tests/test_video_analysis_integration.py -q
```

If pytest is missing:
```bash
python -m pip install pytest
```

## Expected Result
You should see output like:
```
5 passed in <time>
```

## Notes
- The integration test is stubbed and does not run real models.
- If heavy dependencies are missing, the integration test may be skipped
  (this is acceptable for Step 7).
- These tests focus on logic correctness and regression prevention, not model accuracy.
