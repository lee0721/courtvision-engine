import cv2
import json

class ActionRecognitionDrawer:
    """
    Responsible for drawing action recognition results on each frame.
    """
    def __init__(self, labels_file='action_recognition/dataset/labels_dict.json'):
        """
        Initialize the drawer without predictions.
        Predictions should be set later using `set_predictions`.
        """
        self.action_predictions = {}

        # Load action labels from the JSON file
        with open(labels_file, 'r') as f:
            self.action_labels = json.load(f)

    def set_predictions(self, action_predictions):
        """
        Set the action recognition results.
        
        Args:
            action_predictions (dict): Predicted actions {frame_index: action_label}
        """
        # Map action labels (numbers) to action names
        self.action_predictions = {
            frame_idx: self.action_labels.get(str(action_label), "Unknown")
            for frame_idx, action_label in action_predictions.items()
        }

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

            # Get the action name for the current frame
            action = self.action_predictions.get(frame_idx, None)

            if action is not None:
                # Draw action label next to each player in the frame
                if frame_idx < len(player_tracks):  # Ensure the frame_idx is within range
                    player_data = player_tracks[frame_idx]
                    for player_id, player_info in player_data.items():
                        bbox = player_info['bbox']
                        x1, y1, x2, y2 = bbox
                        position = [int((x1 + x2) / 2), int(y2) + 40]

                        # Display the action name (e.g., "Action: Block") on the frame
                        cv2.putText(output_frame, f"Action: {action}", (position[0], position[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames