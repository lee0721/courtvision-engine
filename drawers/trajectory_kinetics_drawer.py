import cv2

class TrajectoryKineticsDrawer():
    """
    A drawer class that overlays player speed (km/h) and cumulative distance (m) on video frames.
    """
    def __init__(self):
        pass 
    def draw(self, video_frames,player_tracks,player_distances_per_frame,player_speed_per_frame):
        """
        Draws speed and total distance for each player onto the video frames.

        Args:
            video_frames (list): List of frames (np.ndarray).
            player_tracks (list): List of dicts mapping player_id to bbox per frame.
            player_distances_per_frame (list): List of dicts mapping player_id to distance (m) in that frame.
            player_speed_per_frame (list): List of dicts mapping player_id to speed (km/h) in that frame.

        Returns:
            list: Frames with speed and distance annotations.
        """
        output_video_frames = []
        total_distances = {}  # Store cumulative distance per player

        for frame, track_dict, distance_dict, speed_dict in zip(
            video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame
        ):
            output_frame = frame.copy()

            # Update total distances
            for player_id, dist in distance_dict.items():
                if player_id not in total_distances:
                    total_distances[player_id] = 0.0
                total_distances[player_id] += dist

            # Draw speed and distance for each player
            for player_id, bbox_info in track_dict.items():
                x1, y1, x2, y2 = bbox_info["bbox"]
                text_x = int((x1 + x2) / 2)
                text_y = int(y2) + 40

                # Draw speed
                if player_id in speed_dict:
                    speed = speed_dict[player_id]
                    cv2.putText(output_frame, f"{speed:.2f} km/h", (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Draw total distance
                if player_id in total_distances:
                    distance = total_distances[player_id]
                    cv2.putText(output_frame, f"{distance:.2f} m", (text_x, text_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_video_frames.append(output_frame)

        return output_video_frames