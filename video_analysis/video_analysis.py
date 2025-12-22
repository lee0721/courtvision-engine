from __future__ import annotations

import os

from action_recognition import ActionRecognitionModel
from arena_mark_detector import ArenaMarkDetector
from ball_aquisition import BallAquisitionDetector
from ball_event_detector import BallEventDetector
from configs import (
    ACTION_DEVICE,
    ACTION_RECOGNITION_MODEL_PATH,
    ARENA_MARK_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_IMAGE_PATH,
    PLAYER_DETECTOR_PATH,
    TEAM_1_CLASS_NAME,
    TEAM_2_CLASS_NAME,
)
from drawers import (
    ActionRecognitionDrawer,
    ArenaMarkDrawer,
    BallEventDrawer,
    BallTracksDrawer,
    FrameNumberDrawer,
    PerspectiveOverlayDrawer,
    PlayerTracksDrawer,
    TeamBallControlDrawer,
    TrajectoryKineticsDrawer,
)
from perspective_transformer import PerspectiveTransformer
from team_classifier import TeamClassifier
from trackers import BallTracker, DeepSortPlayerTracker, PlayerTracker
from trajectory_kinetics_analyzer import TrajectoryKineticsAnalyzer
from utils import read_video, save_video

from .types import (
    AnalysisResult,
    BallPossession,
    EventTimeline,
    PerFrameFloat,
    PlayerAssignments,
    PlayerMetric,
    TeamControlRatio,
)


class VideoAnalysis:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        stub_path: str,
        use_stubs: bool = True,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.stub_path = stub_path
        self.use_stubs = use_stubs

    def run(self) -> dict[str, object]:
        # Read video
        video_frames = read_video(self.input_path)
        if not video_frames:
            raise ValueError(f"No frames read from input video: {self.input_path}")

        # Initialize trackers
        # player_tracker = DeepSortPlayerTracker(PLAYER_DETECTOR_PATH)
        player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
        ball_tracker = BallTracker(BALL_DETECTOR_PATH)

        # Initialize keypoint detector
        mark_detector = ArenaMarkDetector(ARENA_MARK_DETECTOR_PATH)

        # Initialize action recognition model
        action_recognition_model = ActionRecognitionModel(
            ACTION_RECOGNITION_MODEL_PATH,
            device=ACTION_DEVICE,
        )

        # Run detectors
        player_tracks = player_tracker.get_object_tracks(
            video_frames,
            read_from_stub=self.use_stubs,
            stub_path=os.path.join(self.stub_path, "player_track_stubs.pkl"),
        )
        ball_tracks = ball_tracker.get_object_tracks(
            video_frames,
            read_from_stub=self.use_stubs,
            stub_path=os.path.join(self.stub_path, "ball_track_stubs.pkl"),
        )

        # Run keypoint extractor
        arena_marks_per_frame = mark_detector.extract_marks(
            video_frames,
            read_from_stub=self.use_stubs,
            stub_path=os.path.join(self.stub_path, "court_key_points_stub.pkl"),
        )

        # Remove wrong ball detections
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
        # Interpolate ball tracks
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

        # Assign player teams
        team_classifier = TeamClassifier(
            team_1_class_name=TEAM_1_CLASS_NAME,
            team_2_class_name=TEAM_2_CLASS_NAME,
        )
        player_assignment = team_classifier.get_player_teams_across_frames(
            video_frames,
            player_tracks,
            read_from_stub=self.use_stubs,
            stub_path=os.path.join(self.stub_path, "player_assignment_stub.pkl"),
        )

        # Ball acquisition
        ball_aquisition_detector = BallAquisitionDetector()
        ball_aquisition = ball_aquisition_detector.detect_ball_possession(
            player_tracks,
            ball_tracks,
        )

        # Detect passes
        ball_event_detector = BallEventDetector()
        passes = ball_event_detector.detect_passes(ball_aquisition, player_assignment)
        interceptions = ball_event_detector.detect_interceptions(
            ball_aquisition,
            player_assignment,
        )

        # Tactical view
        perspective_transformer = PerspectiveTransformer(
            court_image_path=COURT_IMAGE_PATH,
        )

        arena_marks_per_frame = perspective_transformer.validate_keypoints(
            arena_marks_per_frame,
        )
        tactical_player_positions = perspective_transformer.transform_players_to_tactical_view(
            arena_marks_per_frame,
            player_tracks,
        )

        # Speed and distance calculator
        trajectory_kinetics_analyzer = TrajectoryKineticsAnalyzer(
            perspective_transformer.width,
            perspective_transformer.height,
            perspective_transformer.actual_width_in_meters,
            perspective_transformer.actual_height_in_meters,
        )
        player_distances_per_frame = trajectory_kinetics_analyzer.calculate_distance(
            tactical_player_positions,
        )
        player_speed_per_frame = trajectory_kinetics_analyzer.calculate_speed(
            player_distances_per_frame,
        )

        # Run action recognition
        action_predictions = action_recognition_model.predict(
            video_frames,
            player_tracks,
            read_from_stub=self.use_stubs,
            stub_path=os.path.join(
                self.stub_path,
                "action_recognition_predictions.pkl",
            ),
        )

        # Initialize drawers
        player_tracks_drawer = PlayerTracksDrawer(
            team_1_color=team_classifier.team_1_color_rgb,
            team_2_color=team_classifier.team_2_color_rgb,
        )
        ball_tracks_drawer = BallTracksDrawer()
        arena_mark_drawer = ArenaMarkDrawer()
        team_ball_control_drawer = TeamBallControlDrawer()
        frame_number_drawer = FrameNumberDrawer()
        ball_event_drawer = BallEventDrawer()
        perspective_drawer = PerspectiveOverlayDrawer(
            team_1_color=team_classifier.team_1_color_rgb,
            team_2_color=team_classifier.team_2_color_rgb,
        )
        trajectory_kinetics_drawer = TrajectoryKineticsDrawer()

        # Initialize ActionRecognitionDrawer and set predictions
        action_recognition_drawer = ActionRecognitionDrawer()
        action_recognition_drawer.set_predictions(action_predictions)

        # Draw object tracks
        output_video_frames = player_tracks_drawer.draw(
            video_frames,
            player_tracks,
            player_assignment,
            ball_aquisition,
        )
        output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

        # Draw team ball control
        output_video_frames = team_ball_control_drawer.draw(
            output_video_frames,
            player_assignment,
            ball_aquisition,
        )

        # Draw passes and interceptions
        output_video_frames = ball_event_drawer.draw(
            output_video_frames,
            passes,
            interceptions,
        )

        # Draw keypoints
        output_video_frames = arena_mark_drawer.draw(
            output_video_frames,
            arena_marks_per_frame,
        )

        # Draw tactical view
        output_video_frames = perspective_drawer.draw(
            output_video_frames,
            perspective_transformer.court_image_path,
            perspective_transformer.width,
            perspective_transformer.height,
            perspective_transformer.key_points,
            tactical_player_positions,
            player_assignment,
            ball_aquisition,
        )

        # Draw frame number
        output_video_frames = frame_number_drawer.draw(output_video_frames)

        # Speed and distance drawer
        output_video_frames = trajectory_kinetics_drawer.draw(
            output_video_frames,
            player_tracks,
            player_distances_per_frame,
            player_speed_per_frame,
        )

        # Draw action recognition results
        output_video_frames = action_recognition_drawer.draw(
            output_video_frames,
            player_tracks,
        )

        # Save video
        save_video(output_video_frames, self.output_path)

        analysis_result = self._build_analysis_result(
            passes,
            interceptions,
            ball_aquisition,
            player_assignment,
            player_distances_per_frame,
            player_speed_per_frame,
            len(video_frames),
        )
        return analysis_result.to_dict()

    def _build_analysis_result(
        self,
        passes: list[int],
        interceptions: list[int],
        ball_aquisition: BallPossession,
        player_assignment: PlayerAssignments,
        player_distances_per_frame: PerFrameFloat,
        player_speed_per_frame: PerFrameFloat,
        frame_count: int,
    ) -> AnalysisResult:
        team_control: list[int] = []
        for assignment, ball_player in zip(player_assignment, ball_aquisition):
            if ball_player == -1 or ball_player not in assignment:
                team_control.append(-1)
            else:
                team_control.append(assignment[ball_player])

        team_1_frames = sum(1 for team_id in team_control if team_id == 1)
        team_2_frames = sum(1 for team_id in team_control if team_id == 2)
        none_frames = sum(1 for team_id in team_control if team_id == -1)

        total_frames = frame_count or 1
        team_control_ratio = TeamControlRatio(
            team_1=team_1_frames / total_frames,
            team_2=team_2_frames / total_frames,
            none=none_frames / total_frames,
        )

        distance_totals: dict[int, float] = {}
        for frame_distances in player_distances_per_frame:
            for player_id, distance in frame_distances.items():
                distance_totals[player_id] = distance_totals.get(player_id, 0.0) + float(
                    distance,
                )

        speed_stats: dict[int, dict[str, float]] = {}
        for frame_speeds in player_speed_per_frame:
            for player_id, speed in frame_speeds.items():
                stats = speed_stats.setdefault(
                    player_id,
                    {"sum": 0.0, "count": 0.0, "max": 0.0},
                )
                speed_value = float(speed)
                if speed_value > 0:
                    stats["sum"] += speed_value
                    stats["count"] += 1
                    stats["max"] = max(stats["max"], speed_value)

        player_metrics: dict[str, PlayerMetric] = {}
        for player_id in set(distance_totals) | set(speed_stats):
            stats = speed_stats.get(player_id, {"sum": 0.0, "count": 0.0, "max": 0.0})
            avg_speed = stats["sum"] / stats["count"] if stats["count"] else 0.0
            player_metrics[str(player_id)] = PlayerMetric(
                total_distance_m=round(distance_totals.get(player_id, 0.0), 4),
                avg_speed_kmh=round(avg_speed, 4),
                max_speed_kmh=round(stats["max"], 4),
            )

        return AnalysisResult(
            input_video=self.input_path,
            output_video=self.output_path,
            frame_count=frame_count,
            events=EventTimeline(passes=passes, interceptions=interceptions),
            ball_possession=ball_aquisition,
            team_ball_control_ratio=team_control_ratio,
            player_metrics=player_metrics,
        )
