from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """

    def __init__(self, model_path):
        """
        Initialize the BallTracker with a specified YOLO model.

        Args:
            model_path (str): Path to the trained YOLO model.
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        """
        Detect the ball in a batch of video frames using YOLO model.

        Args:
            frames (list): List of video frames to detect objects in.

        Returns:
            list: A list of detection results from the YOLO model.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Generate ball tracking results from frames with optional stub caching.

        Args:
            frames (list): Video frames to be processed.
            read_from_stub (bool): Whether to load previously saved tracking results.
            stub_path (str): Path to the stub cache file.

        Returns:
            list: A list of dictionaries with tracking results per frame.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = 0

            for det in detection_supervision:
                bbox = det[0].tolist()
                cls_id = det[3]
                confidence = det[2]

                if cls_id == cls_names_inv['basketball']: #test
                    if confidence > max_confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Remove unreasonable ball detections based on frame-to-frame motion.

        Args:
            ball_positions (list): A list of dictionaries containing detected ball positions.

        Returns:
            list: Cleaned list with suspicious detections removed.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            distance = np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2]))
            if distance > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Fill in missing ball positions using linear interpolation.

        Args:
            ball_positions (list): List of dictionaries with ball bbox info.

        Returns:
            list: Interpolated and smoothed ball tracking results.
        """
        bboxes = [frame.get(1, {}).get('bbox', []) for frame in ball_positions]
        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Fill missing values
        df = df.interpolate()
        df = df.bfill()

        interpolated = [{1: {"bbox": row}} for row in df.to_numpy().tolist()]
        return interpolated