from __future__ import annotations

from ultralytics import YOLO
import supervision as sv
import sys 
import logging
from typing import Any, TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np

sys.path.append('../')
from utils.logging_utils import log_kv
from utils import read_stub, save_stub
from configs import DETECTION_BATCH_SIZE, DETECTION_CONFIDENCE

logger = logging.getLogger("courtvision.arena_mark_detector")


class ArenaMarkDetector:
    """
    The ArenaMarkDetector class uses a YOLO model to detect arena (court) keypoints 
    from a batch of video frames. Stub caching is supported to avoid redundant inference.
    """
    def __init__(self, model_path: str) -> None:
        # Load YOLOv8 keypoint detection model
        try:
            self.model = YOLO(model_path)
        except Exception as exc:  # pragma: no cover - model load
            log_kv(
                logger,
                logging.ERROR,
                "arena_mark_model_load_failed",
                model_path=model_path,
                error=str(exc),
            )
            raise
    
    def extract_marks(
        self,
        frames: Sequence["np.ndarray"],
        read_from_stub: bool = False,
        stub_path: str | None = None,
        job_id: str | None = None,
    ) -> list[Any]:
        """
        Detect court keypoints for a sequence of frames using the YOLO model.
        If stub reading is enabled and results are available, cached keypoints are returned.

        Args:
            frames (list of np.ndarray): List of video frames (BGR images) for detection.
            read_from_stub (bool): If True, attempt to load cached keypoints from stub.
            stub_path (str): Path to the cached file.

        Returns:
            list: A list containing keypoints (in model output format) for each input frame.
        """
        # Attempt to load from cache
        arena_marks = read_stub(
            read_from_stub,
            stub_path,
            job_id=job_id,
            stage="arena_mark",
        )
        if arena_marks is not None:
            if len(arena_marks) == len(frames):
                return arena_marks
        
        # Run inference in batches to improve efficiency
        batch_size = DETECTION_BATCH_SIZE
        arena_marks = []
        for i in range(0,len(frames),batch_size):
            # Run YOLO keypoint prediction
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=DETECTION_CONFIDENCE,
            )
            
            # Collect keypoints per frame
            for detection in detections_batch:
                arena_marks.append(detection.keypoints)

        # Save results to cache
        save_stub(
            stub_path,
            arena_marks,
            job_id=job_id,
            stage="arena_mark",
        )
        return arena_marks
