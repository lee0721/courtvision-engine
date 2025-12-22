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
