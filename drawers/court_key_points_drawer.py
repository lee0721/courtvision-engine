import supervision as sv

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
        Draws court keypoints as circles on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """
        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index]

            keypoints_numpy = keypoints.cpu().numpy()  # Convert tensor to numpy

            for idx, (x, y) in enumerate(keypoints_numpy):
                x, y = int(x), int(y)
                cv2.circle(annotated_frame, (x, y), self.radius, self.color, -1)
                cv2.putText(annotated_frame, str(idx), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.label_color, 1)

            output_frames.append(annotated_frame)

        return output_frames
