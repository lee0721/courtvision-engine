from ultralytics import YOLO
import supervision as sv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class DeepSortPlayerTracker:
    """
    A class for player detection and tracking using YOLO and DeepSORT.

    This class combines YOLO object detection with DeepSORT tracking to maintain consistent
    player identities across frames, processing detections in batches.
    """

    def __init__(self, model_path, max_age=30, n_init=3, nn_budget=100):
        """
        Initialize the DeepSortPlayerTracker with YOLO model and DeepSORT tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
            max_age (int): Max frames to keep a track without updates.
            n_init (int): Number of detections before confirming a track.
            nn_budget (int): Size of appearance feature buffer.
        """
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

    def detect_frames(self, frames):
        """
        Detect players in batches of video frames using YOLO.

        Args:
            frames (list): List of video frames.

        Returns:
            list: Detection results for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Run tracking on input video frames and return consistent player IDs per frame.

        Args:
            frames (list): List of video frames to track.
            read_from_stub (bool): Whether to load existing tracking results from cache.
            stub_path (str): Path to the stub file for caching.

        Returns:
            list: Per-frame list of tracked player IDs and their bounding boxes.
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

            # Convert detections to DeepSORT expected format
            deepsort_detections = []
            for det in detection_supervision:
                bbox = det[0]  # [x1, y1, x2, y2]
                conf = det[2]
                cls_id = det[3]
                if cls_id == cls_names_inv['Player']:
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    deepsort_detections.append(([x1, y1, w, h], conf, cls_id))

            # DeepSORT tracking update
            tracked_objects = self.tracker.update_tracks(deepsort_detections, frame=frames[frame_num])

            tracks.append({})
            for track in tracked_objects:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr()  # returns [x1, y1, x2, y2]
                if cls_names[track.det_class] == 'Player':
                    tracks[frame_num][track_id] = {"bbox": bbox.tolist()}

        save_stub(stub_path, tracks)
        return tracks