from .utils import draw_rounded_rectangle
import cv2 
import numpy as np

class TeamBallControlDrawer:
    """
    A class responsible for calculating and drawing team ball control statistics on video frames.
    """
    def __init__(self):
        pass

    def get_team_ball_control(self,player_assignment,ball_aquisition):
        """
        Calculate which team has ball control for each frame.

        Args:
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            numpy.ndarray: An array indicating which team has ball control for each frame
                (1 for Team 1, 2 for Team 2, -1 for no control).
        """

        team_ball_control = []
        for player_assignment_frame,ball_aquisition_frame in zip(player_assignment,ball_aquisition):
            if ball_aquisition_frame == -1:
                team_ball_control.append(-1)
                continue
            if ball_aquisition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue
            if player_assignment_frame[ball_aquisition_frame] == 1:
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        team_ball_control= np.array(team_ball_control) 
        return team_ball_control

    def draw(self,video_frames,player_assignment,ball_aquisition):
        """
        Draw team ball control statistics on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            list: A list of frames with team ball control statistics drawn on them.
        """
        
        team_ball_control = self.get_team_ball_control(player_assignment,ball_aquisition)

        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue

            frame_drawn = self.draw_frame(frame,frame_num,team_ball_control)
            output_video_frames.append(frame_drawn)
        return output_video_frames
    
    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent overlay of team ball control percentages on a single frame.
        """
        # Overlay Position - smaller and centered lower right
        frame_height, frame_width = frame.shape[:2]
        box_width = int(frame_width * 0.30)
        box_height = int(frame_height * 0.10)
        margin = 30

        rect_x2 = frame_width - margin
        rect_x1 = rect_x2 - box_width
        rect_y2 = frame_height - margin
        rect_y1 = rect_y2 - box_height

        # ✨ 自動根據 box 高度調整文字大小
        font_scale = box_height / 100  # 可調參數（越小字越大）
        font_thickness = max(1, int(font_scale * 2))

        # 繪製圓角框
        draw_rounded_rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2),
                            radius=20, color=(255, 255, 255), alpha=0.6)

        # 計算控球時間百分比
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = len(team_ball_control_till_frame)
        team_1 = team_1_num_frames / total_frames
        team_2 = team_2_num_frames / total_frames

        text1 = f"Team 1 Ball Control: {team_1 * 100:.2f}%"
        text2 = f"Team 2 Ball Control: {team_2 * 100:.2f}%"

        (text_width1, text_height1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        (text_width2, text_height2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        center_x = (rect_x1 + rect_x2) // 2
        center_y = (rect_y1 + rect_y2) // 2
        spacing = 10

        text_y1 = center_y - spacing
        text_y2 = center_y + text_height2 + spacing

        text_x1 = center_x - text_width1 // 2
        text_x2 = center_x - text_width2 // 2

        cv2.putText(frame, text1, (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, text2, (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)

        return frame