# Default path to save/load cached results from intermediate modules (used by stub system)
STUBS_DEFAULT_PATH = 'stubs'

# Paths to detection models
PLAYER_DETECTOR_PATH = 'models/player_detector.pt'  # Player YOLOv8 detection model
BALL_DETECTOR_PATH = 'models/ball_detector_model.pt'  # Ball YOLOv8 detection model
ARENA_MARK_DETECTOR_PATH = 'models/arena_mark_detector.pt'  # Court keypoint detection model

# Output video path (rendered results will be saved here)
OUTPUT_VIDEO_PATH = 'output_videos/output_video.avi'

# Path to the trained action recognition model (R(2+1)D)
ACTION_RECOGNITION_MODEL_PATH = 'models/action_r2plus1d_best.pt'

# Team class names used for classification (these must match YOLO class labels or tracker labels)
TEAM_1_CLASS_NAME = "white shirt"
TEAM_2_CLASS_NAME = "green shirt"