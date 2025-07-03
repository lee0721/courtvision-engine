from ultralytics import YOLO
import supervision as sv
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class ArenaMarkDetector:
    """
    The ArenaMarkDetector class uses a YOLO model to detect arena (court) keypoints 
    from a batch of video frames. Stub caching is supported to avoid redundant inference.
    """
    def __init__(self, model_path):
        # Load YOLOv8 keypoint detection model
        self.model = YOLO(model_path)
    
    def extract_marks(self, frames,read_from_stub=False, stub_path=None):
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
        arena_marks = read_stub(read_from_stub,stub_path)
        if arena_marks is not None:
            if len(arena_marks) == len(frames):
                return arena_marks
        
        # Run inference in batches to improve efficiency
        batch_size=20
        arena_marks = []
        for i in range(0,len(frames),batch_size):
            # Run YOLO keypoint prediction
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            
            # Collect keypoints per frame
            for detection in detections_batch:
                arena_marks.append(detection.keypoints)

        # Save results to cache
        save_stub(stub_path,arena_marks)
        return arena_marks