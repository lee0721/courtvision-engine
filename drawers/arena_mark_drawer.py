import supervision as sv

class ArenaMarkDrawer:
    """
    This class is responsible for drawing court keypoints on video frames using Supervision.

    Attributes:
        keypoint_color (str): Hex color code used to draw the keypoints.
    """
    def __init__(self):
        self.keypoint_color = "#15fc00"

    def draw(self, frames, arena_marks):
        """
        Draw court keypoints on each frame.

        Args:
            frames (list of ndarray): List of video frames (images).
            arena_marks (list): A list of keypoint tensors for each frame. Each element is a tensor of (x, y) points.

        Returns:
            list: List of annotated frames with keypoints drawn.
        """
        # Create annotator for labeling keypoints
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )
        
        output_frames = []
        for index,frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = arena_marks[index]
            
            # Convert PyTorch tensor to NumPy array (in case it comes from YOLOv8 keypoints format)
            keypoints_numpy = keypoints.cpu().numpy()
            # Annotate keypoints on the frame
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy
            )

            output_frames.append(annotated_frame)

        return output_frames