import importlib
from pathlib import Path

import pytest


def test_video_analysis_run_with_stubs(
    monkeypatch,
    stub_frames,
    stub_player_tracks,
    stub_ball_tracks,
    stub_team_assignment,
    tmp_path,
):
    try:
        va_module = importlib.import_module("video_analysis.video_analysis")
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing dependency: {exc.name}")
    except ImportError as exc:
        pytest.skip(f"Cannot import video_analysis: {exc}")

    class DummyPlayerTracker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_object_tracks(self, frames, **kwargs):
            return stub_player_tracks

    class DummyBallTracker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_object_tracks(self, frames, **kwargs):
            return stub_ball_tracks

        def remove_wrong_detections(self, tracks):
            return tracks

        def interpolate_ball_positions(self, tracks):
            return tracks

    class DummyArenaMarkDetector:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def extract_marks(self, frames, **kwargs):
            return [None] * len(frames)

    class DummyTeamClassifier:
        def __init__(self, *args, **kwargs) -> None:
            self.team_1_color_rgb = (255, 0, 0)
            self.team_2_color_rgb = (0, 0, 255)

        def get_player_teams_across_frames(self, frames, tracks, **kwargs):
            return stub_team_assignment

    class DummyActionRecognitionModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, frames, tracks, **kwargs):
            return {}

    class DummyPerspectiveTransformer:
        def __init__(self, court_image_path: str) -> None:
            self.court_image_path = court_image_path
            self.width = 100
            self.height = 50
            self.actual_width_in_meters = 10.0
            self.actual_height_in_meters = 5.0
            self.key_points = []

        def validate_keypoints(self, arena_marks):
            return arena_marks

        def transform_players_to_tactical_view(self, arena_marks, player_tracks):
            output = []
            for frame_tracks in player_tracks:
                frame_positions = {}
                for player_id, info in frame_tracks.items():
                    x1, y1, x2, y2 = info["bbox"]
                    frame_positions[player_id] = ((x1 + x2) / 2, (y1 + y2) / 2)
                output.append(frame_positions)
            return output

    class DummyDrawer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def draw(self, frames, *args, **kwargs):
            return frames

    class DummyActionRecognitionDrawer(DummyDrawer):
        def set_predictions(self, predictions):
            self.predictions = predictions

    def dummy_read_video(path):
        return stub_frames

    def dummy_save_video(frames, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")

    monkeypatch.setattr(va_module, "PlayerTracker", DummyPlayerTracker)
    monkeypatch.setattr(va_module, "BallTracker", DummyBallTracker)
    monkeypatch.setattr(va_module, "ArenaMarkDetector", DummyArenaMarkDetector)
    monkeypatch.setattr(va_module, "TeamClassifier", DummyTeamClassifier)
    monkeypatch.setattr(va_module, "ActionRecognitionModel", DummyActionRecognitionModel)
    monkeypatch.setattr(va_module, "PerspectiveTransformer", DummyPerspectiveTransformer)
    monkeypatch.setattr(va_module, "PlayerTracksDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "BallTracksDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "ArenaMarkDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "TeamBallControlDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "FrameNumberDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "BallEventDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "PerspectiveOverlayDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "TrajectoryKineticsDrawer", DummyDrawer)
    monkeypatch.setattr(va_module, "ActionRecognitionDrawer", DummyActionRecognitionDrawer)
    monkeypatch.setattr(va_module, "read_video", dummy_read_video)
    monkeypatch.setattr(va_module, "save_video", dummy_save_video)

    output_path = tmp_path / "out.mp4"
    analysis = va_module.VideoAnalysis(
        input_path="input.mp4",
        output_path=str(output_path),
        stub_path=str(tmp_path / "stubs"),
        use_stubs=True,
        job_id="test-job",
    )
    result = analysis.run()

    expected_keys = {
        "input_video",
        "output_video",
        "frame_count",
        "events",
        "ball_possession",
        "team_ball_control_ratio",
        "player_metrics",
    }
    assert expected_keys.issubset(result.keys())
    assert result["frame_count"] == len(stub_frames)
    assert output_path.exists()
