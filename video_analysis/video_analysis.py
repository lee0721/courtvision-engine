import os
from utils import read_video, save_video
from trackers import DeepSortPlayerTracker, BallTracker, PlayerTracker
from team_classifier import TeamClassifier
from arena_mark_detector import ArenaMarkDetector
from ball_aquisition import BallAquisitionDetector
from ball_event_detector import BallEventDetector
from perspective_transformer import PerspectiveTransformer
from trajectory_kinetics_analyzer import TrajectoryKineticsAnalyzer
from action_recognition import ActionRecognitionModel 
from drawers import (
    PlayerTracksDrawer, 
    BallTracksDrawer,
    ArenaMarkDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    BallEventDrawer,
    PerspectiveOverlayDrawer,
    TrajectoryKineticsDrawer,
    ActionRecognitionDrawer 
)
from configs import(
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    ARENA_MARK_DETECTOR_PATH,
    ACTION_RECOGNITION_MODEL_PATH,
    TEAM_1_CLASS_NAME, 
    TEAM_2_CLASS_NAME
)

class VideoAnalysis:
    def __init__(self, input_path, output_path, stub_path, use_stubs=True):
        self.input_path = input_path
        self.output_path = output_path
        self.stub_path = stub_path
        self.use_stubs = use_stubs

    def run(self):
        # Read Video
        video_frames = read_video(self.input_path)
        if not video_frames:
            raise ValueError(f"No frames read from input video: {self.input_path}")
        
        ## Initialize Tracker
        #player_tracker = DeepSortPlayerTracker(PLAYER_DETECTOR_PATH)
        player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
        ball_tracker = BallTracker(BALL_DETECTOR_PATH)

        ## Initialize Keypoint Detector
        mark_detector = ArenaMarkDetector(ARENA_MARK_DETECTOR_PATH)

        # Initialize Action Recognition Model
        action_recognition_model = ActionRecognitionModel(ACTION_RECOGNITION_MODEL_PATH) 

        # Run Detectors
        player_tracks = player_tracker.get_object_tracks(
                                        video_frames,
                                        read_from_stub=self.use_stubs,
                                        stub_path=os.path.join(self.stub_path, 'player_track_stubs.pkl')
                                        )
        
        ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                    read_from_stub=self.use_stubs,
                                                    stub_path=os.path.join(self.stub_path, 'ball_track_stubs.pkl')
                                                    )
        ## Run KeyPoint Extractor
        arena_marks_per_frame = mark_detector.extract_marks(video_frames,
                                                                        read_from_stub=self.use_stubs,
                                                                        stub_path=os.path.join(self.stub_path, 'court_key_points_stub.pkl')
                                                                        )

        # Remove Wrong Ball Detections
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
        # Interpolate Ball Tracks
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    

        # Assign Player Teams
        team_classifier = TeamClassifier( team_1_class_name=TEAM_1_CLASS_NAME, team_2_class_name=TEAM_2_CLASS_NAME)
        player_assignment = team_classifier.get_player_teams_across_frames(video_frames,
                                                                        player_tracks,
                                                                        read_from_stub=self.use_stubs,
                                                                        stub_path=os.path.join(self.stub_path, 'player_assignment_stub.pkl')
                                                                        )

        # Ball Acquisition
        ball_aquisition_detector = BallAquisitionDetector()
        ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

        # Detect Passes
        ball_event_detector = BallEventDetector()
        passes = ball_event_detector.detect_passes(ball_aquisition,player_assignment)
        interceptions = ball_event_detector.detect_interceptions(ball_aquisition,player_assignment)

        # Tactical View
        perspective_transformer = PerspectiveTransformer(
            court_image_path="./images/basketball_court.png"
        )

        arena_marks_per_frame = perspective_transformer.validate_keypoints(arena_marks_per_frame)
        tactical_player_positions = perspective_transformer.transform_players_to_tactical_view(arena_marks_per_frame,player_tracks)

        # Speed and Distance Calculator
        trajectory_kinetics_analyzer = TrajectoryKineticsAnalyzer(
            perspective_transformer.width,
            perspective_transformer.height,
            perspective_transformer.actual_width_in_meters,
            perspective_transformer.actual_height_in_meters
        )
        player_distances_per_frame = trajectory_kinetics_analyzer.calculate_distance(tactical_player_positions)
        player_speed_per_frame = trajectory_kinetics_analyzer.calculate_speed(player_distances_per_frame)
        
        # Run Action Recognition
        action_predictions = action_recognition_model.predict(video_frames, player_tracks, read_from_stub=self.use_stubs, 
                                                            stub_path=os.path.join(self.stub_path, 'action_recognition_predictions.pkl'))
        
        # Draw output   
        # Initialize Drawers
        player_tracks_drawer = PlayerTracksDrawer(
            team_1_color=team_classifier.team_1_color_rgb,
            team_2_color=team_classifier.team_2_color_rgb
        )
        ball_tracks_drawer = BallTracksDrawer()
        arena_mark_drawer = ArenaMarkDrawer()
        team_ball_control_drawer = TeamBallControlDrawer()
        frame_number_drawer = FrameNumberDrawer()
        ball_event_drawer = BallEventDrawer()
        perspective_drawer = PerspectiveOverlayDrawer(
            team_1_color=team_classifier.team_1_color_rgb,
            team_2_color=team_classifier.team_2_color_rgb
        )
        trajectory_kinetics_drawer = TrajectoryKineticsDrawer()
        # Initialize ActionRecognitionDrawer and set predictions
        action_recognition_drawer = ActionRecognitionDrawer()
        action_recognition_drawer.set_predictions(action_predictions)  # Set predictions here
        
        ## Draw object Tracks
        output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                        player_tracks,
                                                        player_assignment,
                                                        ball_aquisition)
        output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

        # Draw Team Ball Control
        output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                            player_assignment,
                                                            ball_aquisition)
        # Draw Passes and Interceptions
        output_video_frames = ball_event_drawer.draw(output_video_frames,
                                                                passes,
                                                                interceptions)
        ## Draw KeyPoints
        output_video_frames = arena_mark_drawer.draw(output_video_frames, arena_marks_per_frame)

        ## Draw Tactical View
        output_video_frames = perspective_drawer.draw(output_video_frames,
                                                        perspective_transformer.court_image_path,
                                                        perspective_transformer.width,
                                                        perspective_transformer.height,
                                                        perspective_transformer.key_points,
                                                        tactical_player_positions,
                                                        player_assignment,
                                                        ball_aquisition,
                                                        )
        ## Draw Frame Number
        output_video_frames = frame_number_drawer.draw(output_video_frames)
        
        # Speed and Distance Drawer
        output_video_frames = trajectory_kinetics_drawer.draw(output_video_frames,
                                                            player_tracks,
                                                            player_distances_per_frame,
                                                            player_speed_per_frame
                                                            )
        # Draw action recognition results
        output_video_frames = action_recognition_drawer.draw(output_video_frames, player_tracks) 
        
        # Save video
        save_video(output_video_frames, self.output_path)

        analysis_result = self._build_analysis_result(
            passes,
            interceptions,
            ball_aquisition,
            player_assignment,
            player_distances_per_frame,
            player_speed_per_frame,
            len(video_frames)
        )
        return analysis_result

    def _build_analysis_result(
        self,
        passes,
        interceptions,
        ball_aquisition,
        player_assignment,
        player_distances_per_frame,
        player_speed_per_frame,
        frame_count
    ):
        team_control = []
        for assignment, ball_player in zip(player_assignment, ball_aquisition):
            if ball_player == -1 or ball_player not in assignment:
                team_control.append(-1)
            else:
                team_control.append(assignment[ball_player])

        team_1_frames = sum(1 for team_id in team_control if team_id == 1)
        team_2_frames = sum(1 for team_id in team_control if team_id == 2)
        none_frames = sum(1 for team_id in team_control if team_id == -1)

        total_frames = frame_count or 1
        team_control_ratio = {
            "team_1": team_1_frames / total_frames,
            "team_2": team_2_frames / total_frames,
            "none": none_frames / total_frames,
        }

        distance_totals = {}
        for frame_distances in player_distances_per_frame:
            for player_id, distance in frame_distances.items():
                distance_totals[player_id] = distance_totals.get(player_id, 0.0) + float(distance)

        speed_stats = {}
        for frame_speeds in player_speed_per_frame:
            for player_id, speed in frame_speeds.items():
                stats = speed_stats.setdefault(player_id, {"sum": 0.0, "count": 0, "max": 0.0})
                speed_value = float(speed)
                if speed_value > 0:
                    stats["sum"] += speed_value
                    stats["count"] += 1
                    stats["max"] = max(stats["max"], speed_value)

        player_metrics = {}
        for player_id in set(distance_totals) | set(speed_stats):
            stats = speed_stats.get(player_id, {"sum": 0.0, "count": 0, "max": 0.0})
            avg_speed = stats["sum"] / stats["count"] if stats["count"] else 0.0
            player_metrics[str(player_id)] = {
                "total_distance_m": round(distance_totals.get(player_id, 0.0), 4),
                "avg_speed_kmh": round(avg_speed, 4),
                "max_speed_kmh": round(stats["max"], 4),
            }

        return {
            "input_video": self.input_path,
            "output_video": self.output_path,
            "frame_count": frame_count,
            "events": {
                "passes": passes,
                "interceptions": interceptions,
            },
            "ball_possession": ball_aquisition,
            "team_ball_control_ratio": team_control_ratio,
            "player_metrics": player_metrics,
        }
