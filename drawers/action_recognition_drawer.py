import cv2

class ActionRecognitionDrawer:
    """
    Responsible for drawing action recognition results on each frame.
    """
    def __init__(self):
        """
        Initialize the drawer without predictions.
        Predictions should be set later using `set_predictions`.
        """
        self.action_predictions = {}

    def set_predictions(self, action_predictions):
        """
        Set the action recognition results.
        
        Args:
            action_predictions (dict): Predicted actions {frame_index: action_label}
        """
        self.action_predictions = action_predictions

    def draw(self, video_frames, player_tracks):
        """
        Draw action labels on each frame.

        Args:
            video_frames (list): List of frames (images).
            player_tracks (list): Player tracking data for each frame, 
                                  formatted as [{player_id: {'bbox': (x1, y1, x2, y2)}}]
        
        Returns:
            list: Output video frames with action labels drawn.
        """
        output_video_frames = []

        for frame_idx, frame in enumerate(video_frames):
            output_frame = frame.copy()

            # Get the action label for the current frame
            action = self.action_predictions.get(frame_idx, None)

            if action is not None:
                # Draw action label next to each player in the frame
                if frame_idx < len(player_tracks):  # Ensure the frame_idx is within range
                    player_data = player_tracks[frame_idx]
                    for player_id, player_info in player_data.items():
                        bbox = player_info['bbox']
                        x1, y1, x2, y2 = bbox
                        position = [int((x1 + x2) / 2), int(y2) + 40]

                        # Display the action label on the frame
                        cv2.putText(output_frame, f"Action: {action}", (position[0], position[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames