import cv2
import json

class ActionRecognitionDrawer:
    """
    This class draws action recognition labels on the video frames,
    based on per-player predictions.
    """
    def __init__(self, labels_file='action_recognition/training_notebooks/dataset/labels_dict.json'):
        # Load the mapping from label index to action label name
        self.action_predictions = {}
        with open(labels_file, 'r') as f:
            self.action_labels = json.load(f)

    def set_predictions(self, action_predictions):
        """
        Store predicted action labels for each player.

        Args:
            action_predictions (dict): {player_id: [label_index1, label_index2, ...]}
        """
        self.action_predictions = {
            player_id: [
                self.action_labels.get(str(label_index), "Unknown") # Translate index to label
                for label_index in label_list
            ]
            for player_id, label_list in action_predictions.items()
        }

    def draw(self, video_frames, player_tracks, ball_control_data, vid_stride=8):
        """
        Draw action labels on video frames.

        Args:
            video_frames (list of ndarray): Original video frames.
            player_tracks (list of dict): List of player detections per frame.
            vid_stride (int): Stride used when generating clips (default 8).
        
        Returns:
            list: Annotated video frames.
        """
        output_video_frames = []

        for frame_idx, frame in enumerate(video_frames):
            output_frame = frame.copy()

            if frame_idx < len(player_tracks):
                player_data = player_tracks[frame_idx]
                for player_id, player_info in player_data.items():
                    bbox = player_info['bbox']
                    x1, y1, x2, y2 = bbox
                    # Label position below the player box
                    position = [int((x1 + x2) / 2)-20, int(y2) + 10]

                    # Determine action label for this player at this frame
                    label = ""
                    if player_id in self.action_predictions:
                        clip_idx = frame_idx // vid_stride
                        player_labels = self.action_predictions[player_id]
                        if 0 <= clip_idx < len(player_labels):
                            label = player_labels[clip_idx]
                            if label:
                                cv2.putText(output_frame, f"Action: {label}", (position[0], position[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            output_video_frames.append(output_frame)

        return output_video_frames