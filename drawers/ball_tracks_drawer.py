from .utils import draw_traingle

class BallTracksDrawer:
    """
    A drawer class responsible for drawing tracked ball positions on video frames.
    Each ball position is visualized using a triangular pointer.

    Attributes:
        ball_pointer_color (tuple): BGR color used to draw the triangle pointer for the ball.
    """

    def __init__(self):
        """
        Initialize the BallTracksDrawer with default green pointer color.
        """
        self.ball_pointer_color = (0, 255, 0)  # Green

    def draw(self, video_frames, tracks):
        """
        Overlay ball tracking results onto each video frame.

        Args:
            video_frames (list of ndarray): The input video frames.
            tracks (list of dict): Frame-wise dictionaries, each containing ball information.
                                   Expected format: {ball_id: {"bbox": [x1, y1, x2, y2], ...}}

        Returns:
            list: A list of frames with ball pointers drawn.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks[frame_num]

            # Draw ball 
            for _, ball in ball_dict.items():
                if ball["bbox"] is None:
                    continue
                frame = draw_traingle(frame, ball["bbox"],self.ball_pointer_color)

            output_video_frames.append(frame)
            
        return output_video_frames