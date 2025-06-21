import cv2
import json

class ActionRecognitionDrawer:
    """
    Responsible for drawing action recognition results on each frame.
    Each player can have their own action prediction per clip.
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
            action_predictions (dict): {player_id: [action_label_1, action_label_2, ...]}
        """
        # Convert action index to label string
        self.action_predictions = {
            player_id: [
                self.action_labels.get(str(action_label), "Unknown")
                for action_label in action_list
            ]
            for player_id, action_list in action_predictions.items()
        }

    def draw(self, video_frames, player_tracks, vid_stride=8):
        """
        Draw action labels for each player in each frame.

        Args:
            video_frames (list): List of frames (images).
            player_tracks (list): Player tracking data for each frame,
                                  formatted as [{player_id: {'bbox': (x1, y1, x2, y2)}}]
            vid_stride (int): The stride used when generating clips for prediction.

        Returns:
            list: Output video frames with action labels drawn.
        """
        output_video_frames = []

        for frame_idx, frame in enumerate(video_frames):
            output_frame = frame.copy()

            if frame_idx < len(player_tracks):
                player_data = player_tracks[frame_idx]
                for player_id, player_info in player_data.items():
                    bbox = player_info['bbox']
                    x1, y1, x2, y2 = bbox
                    position = [int((x1 + x2) / 2), int(y2) + 40]

                    label = ""
                    if player_id in self.action_predictions:
                        label_idx = frame_idx // vid_stride
                        if label_idx < len(self.action_predictions[player_id]):
                            label = self.action_predictions[player_id][label_idx]

                    if label:
                        cv2.putText(output_frame, f"Action: {label}", (position[0], position[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames