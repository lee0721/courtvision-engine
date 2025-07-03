import os
import sys
import pathlib

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))

from utils import measure_distance


class TrajectoryKineticsAnalyzer:
    """
    A class to analyze player movement in tactical view, providing distance (m) and speed (km/h) metrics.

    It converts pixel coordinates to real-world meters based on known court dimensions, and
    calculates cumulative distance and running speed over time.
    """

    def __init__(self, 
                 width_in_pixels,
                 height_in_pixels,
                 width_in_meters,
                 height_in_meters):
        """
        Args:
            width_in_pixels (int): Court width in pixels.
            height_in_pixels (int): Court height in pixels.
            width_in_meters (float): Court width in meters.
            height_in_meters (float): Court height in meters.
        """
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters = height_in_meters

    def calculate_meter_distance(self, previous_pixel_position, current_pixel_position):
        """
        Convert pixel positions to real-world meters and compute distance.

        Args:
            previous_pixel_position (tuple): (x, y) in pixels.
            current_pixel_position (tuple): (x, y) in pixels.

        Returns:
            float: Distance in meters (scaled by 0.4 factor).
        """
        px1, py1 = previous_pixel_position
        px2, py2 = current_pixel_position

        mx1 = px1 * self.width_in_meters / self.width_in_pixels
        my1 = py1 * self.height_in_meters / self.height_in_pixels
        mx2 = px2 * self.width_in_meters / self.width_in_pixels
        my2 = py2 * self.height_in_meters / self.height_in_pixels

        dist = measure_distance((mx1, my1), (mx2, my2)) * 0.4
        return dist

    def calculate_distance(self, tactical_player_positions):
        """
        Calculate distance (in meters) each player travels per frame.

        Args:
            tactical_player_positions (list): List of dicts mapping player_id to (x,y) positions.

        Returns:
            list: List of dicts containing distance per player per frame.
        """
        previous_position = {}
        output_distances = []

        for frame_idx, positions in enumerate(tactical_player_positions):
            output_distances.append({})
            for player_id, curr_pos in positions.items():
                if player_id in previous_position:
                    dist = self.calculate_meter_distance(previous_position[player_id], curr_pos)
                    output_distances[frame_idx][player_id] = dist
                previous_position[player_id] = curr_pos

        return output_distances

    def calculate_speed(self, distances, fps=30):
        """
        Calculate speed (km/h) for each player over time using a sliding window.

        Args:
            distances (list): Output from `calculate_distance()`.
            fps (int): Frame rate of the video.

        Returns:
            list: List of dicts containing player speed in km/h per frame.
        """
        speeds = []
        window_size = 5  # speed averaged over past 5 frames

        for frame_idx in range(len(distances)):
            speeds.append({})
            for player_id in distances[frame_idx].keys():
                start_frame = max(0, frame_idx - window_size * 3 + 1)

                total_distance = 0
                frames_present = 0
                last_frame = None

                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_frame is not None:
                            total_distance += distances[i][player_id]
                            frames_present += 1
                        last_frame = i

                if frames_present >= window_size:
                    time_sec = frames_present / fps
                    time_hr = time_sec / 3600
                    speed = (total_distance / 1000) / time_hr if time_hr > 0 else 0
                    speeds[frame_idx][player_id] = speed
                else:
                    speeds[frame_idx][player_id] = 0

        return speeds