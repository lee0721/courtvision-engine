import supervision as sv
import numpy as np
import cv2

class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames
    with number labels and an inverted triangle below each label.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints and triangles.
        triangle_size (int): Size of the inverted triangle (base width in pixels).
        text_scale (float): Scale of the text for number labels.
        text_thickness (int): Thickness of the text for number labels.
    """
    def __init__(self):
        self.keypoint_color = '#ff2c2c'
        self.triangle_size = 10  # Base width of the inverted triangle
        self.text_scale = 0.5
        self.text_thickness = 1

    def _draw_inverted_triangle(self, frame, x, y):
        """
        Draws an inverted triangle at the specified (x, y) coordinates using OpenCV.

        Args:
            frame (np.ndarray): The frame to draw on.
            x (int): X-coordinate of the triangle's top vertex.
            y (int): Y-coordinate of the triangle's top vertex (adjusted for text).

        Returns:
            np.ndarray: Frame with the triangle drawn.
        """
        # Convert hex color to BGR for OpenCV
        color = sv.Color.from_hex(self.keypoint_color).as_bgr()
        
        # Define triangle vertices: top (x, y), bottom-left, bottom-right
        half_base = self.triangle_size // 2
        vertices = np.array([
            [x, y],                    # Top vertex
            [x - half_base, y + self.triangle_size],  # Bottom-left
            [x + half_base, y + self.triangle_size]   # Bottom-right
        ], dtype=np.int32)
        
        # Draw filled triangle
        cv2.fillPoly(frame, [vertices], color)
        return frame

    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints on a given list of frames as number labels with
        an inverted triangle below each label.

        Args:
            frames (list): A list of frames (as NumPy arrays) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list
                contains the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """
        # Initialize annotator for number labels
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=self.text_scale,
            text_thickness=self.text_thickness
        )
        
        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index]
            
            # Convert PyTorch tensor to NumPy array if necessary
            keypoints_numpy = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
            
            # Draw number labels
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy
            )
            
            # Draw inverted triangles below each keypoint
            for i, (x, y) in enumerate(keypoints_numpy):
                # Adjust y-coordinate to place triangle below the text
                # Approximate text height based on text_scale (empirical adjustment)
                text_height = int(15 * self.text_scale)
                triangle_y = int(y + text_height + self.triangle_size // 2)
                annotated_frame = self._draw_inverted_triangle(annotated_frame, int(x), triangle_y)
            
            output_frames.append(annotated_frame)

        return output_frames