from .settings import get_settings

_settings = get_settings()

# Default path to save/load cached results from intermediate modules (used by stub system)
STUBS_DEFAULT_PATH = _settings.stubs_dir

# Paths to detection models
PLAYER_DETECTOR_PATH = _settings.player_detector_path  # Player YOLOv8 detection model
BALL_DETECTOR_PATH = _settings.ball_detector_path  # Ball YOLOv8 detection model
ARENA_MARK_DETECTOR_PATH = _settings.arena_mark_detector_path  # Court keypoint detection model

# Output video path (rendered results will be saved here)
OUTPUT_VIDEO_PATH = f"{_settings.output_dir}/output_video.avi"

# Path to the trained action recognition model (R(2+1)D)
ACTION_RECOGNITION_MODEL_PATH = _settings.action_recognition_model_path
ACTION_DEVICE = _settings.action_device

# Court overlay image path
COURT_IMAGE_PATH = _settings.court_image_path

# Team class names used for classification (these must match YOLO class labels or tracker labels)
TEAM_1_CLASS_NAME = _settings.team_1_class_name
TEAM_2_CLASS_NAME = _settings.team_2_class_name

# Detection defaults
DETECTION_BATCH_SIZE = _settings.detection_batch_size
DETECTION_CONFIDENCE = _settings.detection_confidence

# Video output
OUTPUT_VIDEO_FPS = _settings.output_video_fps

# Action recognition clip settings
ACTION_CLIP_LEN = _settings.action_clip_len
ACTION_STRIDE = _settings.action_stride

# Ball possession thresholds
BALL_POSSESSION_THRESHOLD_PX = _settings.ball_possession_threshold_px
BALL_POSSESSION_MIN_FRAMES = _settings.ball_possession_min_frames
BALL_POSSESSION_CONTAINMENT_THRESHOLD = _settings.ball_possession_containment_threshold

# Tactical view dimensions
TACTICAL_VIEW_WIDTH_PX = _settings.tactical_view_width_px
TACTICAL_VIEW_HEIGHT_PX = _settings.tactical_view_height_px
COURT_WIDTH_M = _settings.court_width_m
COURT_HEIGHT_M = _settings.court_height_m

# Trajectory analysis defaults
SPEED_WINDOW_SIZE = _settings.speed_window_size
ANALYSIS_FPS = _settings.analysis_fps
