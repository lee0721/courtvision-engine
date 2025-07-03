from .utils import draw_rounded_rectangle
import cv2 
import numpy as np

class TeamBallControlDrawer:
    """
    A drawer class responsible for calculating and visualizing team ball control statistics.
    """
    def __init__(self):
        pass

    def get_team_ball_control(self, player_assignment, ball_aquisition):
        """
        Determine which team controls the ball in each frame.

        Args:
            player_assignment (list): List of dicts, each mapping player_id to team_id for a frame.
            ball_aquisition (list): List indicating the player_id with the ball per frame (-1 if none).

        Returns:
            np.ndarray: Array where each element is 1 (Team 1), 2 (Team 2), or -1 (no control).
        """
        control_array = []

        for assign, ball_player in zip(player_assignment, ball_aquisition):
            if ball_player == -1 or ball_player not in assign:
                control_array.append(-1)
            else:
                control_array.append(assign[ball_player])

        return np.array(control_array)

    def draw(self, video_frames, player_assignment, ball_aquisition):
        """
        Draw accumulated ball control statistics on video frames.

        Args:
            video_frames (list): List of video frames (np.ndarray).
            player_assignment (list): Per-frame player_id to team_id mapping.
            ball_aquisition (list): List of player_ids who possess the ball each frame.

        Returns:
            list: Frames with overlaid ball control statistics.
        """
        team_ball_control = self.get_team_ball_control(player_assignment, ball_aquisition)

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            frame_drawn = self.draw_frame(frame, frame_num, team_ball_control)
            output_video_frames.append(frame_drawn)

        return output_video_frames
    
    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Overlay the statistics box on a single frame.

        Args:
            frame (np.ndarray): Input frame.
            frame_num (int): Current frame number.
            team_ball_control (np.ndarray): Control array for all frames.

        Returns:
            np.ndarray: Frame with overlay.
        """
        h, w = frame.shape[:2]
        box_w = int(w * 0.20)
        box_h = int(h * 0.10)
        margin = 30

        x2 = w - margin - 400
        x1 = x2 - box_w
        y2 = h - margin
        y1 = y2 - box_h

        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), radius=20, color=(255, 255, 255), alpha=0.6)

        font_scale = box_h / 150
        font_thickness = max(1, int(font_scale * 2))

        control_so_far = team_ball_control[:frame_num + 1]
        team_1_ratio = np.sum(control_so_far == 1) / len(control_so_far)
        team_2_ratio = np.sum(control_so_far == 2) / len(control_so_far)

        text1 = f"Team 1 Ball Control: {team_1_ratio * 100:.2f}%"
        text2 = f"Team 2 Ball Control: {team_2_ratio * 100:.2f}%"

        (w1, h1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        (w2, h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        spacing = 10

        y_text1 = center_y - spacing
        y_text2 = center_y + h2 + spacing

        x_text1 = center_x - w1 // 2
        x_text2 = center_x - w2 // 2

        cv2.putText(frame, text1, (x_text1, y_text1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, text2, (x_text2, y_text2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        return frame