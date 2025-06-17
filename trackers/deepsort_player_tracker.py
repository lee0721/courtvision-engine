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
            max_age (int): Maximum number of frames a track can remain inactive before deletion.
            n_init (int): Number of consecutive detections required to confirm a track.
            nn_budget (int): Maximum size of the appearance descriptor gallery.
        """
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Obtain player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                  where each dictionary maps player IDs to their bounding box coordinates.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert detections to DeepSORT format: [[x, y, w, h], conf, class_id]
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

            # Update tracker
            tracked_objects = self.tracker.update_tracks(deepsort_detections, frame=frames[frame_num])

            tracks.append({})
            for track in tracked_objects:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                if cls_names[track.det_class] == 'Player':
                    tracks[frame_num][track_id] = {"bbox": bbox.tolist()}

        save_stub(stub_path, tracks)
        return tracks