import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import get_foot_position, measure_distance

class PerspectiveTransformer:
    """
    A class to transform player positions from camera perspective to a top-down tactical view.
    """

    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300    # Width of tactical view image in pixels
        self.height = 161   # Height of tactical view image in pixels

        # Actual court dimensions in meters
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        # Fixed keypoints on tactical view based on real-world proportions
        self.key_points = [
            # Left edge
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),

            # Midline
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),

            # Left free throw line
            (int((5.79 / self.actual_width_in_meters) * self.width), int((5.18 / self.actual_height_in_meters) * self.height)),
            (int((5.79 / self.actual_width_in_meters) * self.width), int((10 / self.actual_height_in_meters) * self.height)),

            # Right edge
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0),

            # Right free throw line
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width), int((5.18 / self.actual_height_in_meters) * self.height)),
            (int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width), int((10 / self.actual_height_in_meters) * self.height)),
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Validates court keypoints based on relative distances between them.

        Args:
            keypoints_list (List): A list of frames, each containing 18 keypoints.
                Each keypoint set includes .xy and .xyn tensors.
        
        Returns:
            List: Updated keypoints list with invalid points zeroed out.
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0]

            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []

            for i in detected_indices:
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]

                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')
                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)

        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transforms player bounding box foot positions from video coordinates to tactical view coordinates.

        Args:
            keypoints_list (List): List of court keypoints per frame (from YOLO pose output).
            player_tracks (List): List of player tracking dictionaries per frame.

        Returns:
            List: List of dictionaries per frame mapping player_id -> (x, y) in tactical view.
        """
        tactical_player_positions = []

        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue

            detected_keypoints = frame_keypoints
            valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]

            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue

            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)

            try:
                homography = Homography(source_points, target_points)

                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)])
                    tactical_position = homography.transform_points(player_position)

                    if (0 <= tactical_position[0][0] <= self.width and
                        0 <= tactical_position[0][1] <= self.height):
                        tactical_positions[player_id] = tactical_position[0].tolist()

            except (ValueError, cv2.error):
                pass

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions