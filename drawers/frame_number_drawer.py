import cv2

class FrameNumberDrawer:
    """
    A drawer class responsible for overlaying frame numbers on video frames.
    This is useful for debugging and aligning frame-level events.

    Methods:
        draw(frames): Draws frame index numbers on each frame.
    """
    def __init__(self):
        pass

    def draw(self, frames):
        """
        Annotates each frame with its index number.

        Args:
            frames (list of ndarray): The input list of video frames.

        Returns:
            list: A list of frames with frame numbers drawn in the bottom-left corner.
        """
        output_frames = []
        for i in range(len(frames)):
            frame = frames[i].copy()
            frame_text = f"{i}"  # Frame index as string

            # Calculate text size for background box
            text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = 50
            text_y = frame.shape[0] - 150  # Adjust this value to move the text vertically

            # Draw white rectangle background
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (255, 255, 255),
                -1
            )

            # Draw black frame number text
            cv2.putText(
                frame,
                frame_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                2
            )

            output_frames.append(frame)

        return output_frames