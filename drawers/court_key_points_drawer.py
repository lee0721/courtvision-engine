import supervision as sv
import numpy as np
import torch

class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints.
    """
    def __init__(self):
        self.keypoint_color = '#ff2c2c'

    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints (dots and labels without rectangular background) on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8)
        
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color(0, 0, 0, alpha=0),  # Transparent background to remove rectangle
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )
        
        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = court_keypoints[index]
            # Draw dots
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints)
            # Draw labels (numbers without background)
            # Convert PyTorch tensor to numpy array
            keypoints_numpy = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy)

            output_frames.append(annotated_frame)

        return output_frames