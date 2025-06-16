from .utils import draw_rounded_rectangle
import cv2
import numpy as np

class PassInterceptionDrawer:
    """
    A class responsible for calculating and drawing pass and interception statistics
    on a sequence of video frames.
    """
    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        """
        Calculate the number of passes and interceptions for Team 1 and Team 2.

        Args:
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)

        Returns:
            tuple: A tuple of four integers (team1_pass_total, team2_pass_total,
                team1_interception_total, team2_interception_total) indicating the total
                number of passes and interceptions for both teams.
        """
        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_num, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):
            if pass_frame == 1:
                team1_passes.append(frame_num)
            elif pass_frame == 2:
                team2_passes.append(frame_num)
                
            if interception_frame == 1:
                team1_interceptions.append(frame_num)
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)
                
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)

    def draw(self, video_frames, passes, interceptions):
        """
        Draw pass and interception statistics on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)

        Returns:
            list: A list of frames with pass and interception statistics drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions)
            output_video_frames.append(frame_drawn)
        return output_video_frames
    
    def draw_frame(self, frame, frame_num, passes, interceptions):
        frame_height, frame_width = frame.shape[:2]
        rect_width = int(frame_width * 0.30)
        rect_height = int(frame_height * 0.10)

        rect_x2 = frame_width - 20
        rect_y2 = frame_height - 20
        rect_x1 = rect_x2 - rect_width
        rect_y1 = rect_y2 - rect_height

        draw_rounded_rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2),
                            radius=20, color=(255, 255, 255), alpha=0.6)

        # ✨ 自動根據 box 高度調整文字大小
        font_scale = rect_height / 150  # 控制縮放比例（數字越小字越大）
        font_thickness = max(1, int(font_scale * 2))

        passes_till_frame = passes[:frame_num + 1]
        interceptions_till_frame = interceptions[:frame_num + 1]

        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame,
            interceptions_till_frame
        )

        text1 = f"Team 1 - Passes: {team1_passes} Interceptions: {team1_interceptions}"
        text2 = f"Team 2 - Passes: {team2_passes} Interceptions: {team2_interceptions}"

        text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

        center_x = (rect_x1 + rect_x2) // 2
        center_y = (rect_y1 + rect_y2) // 2
        spacing = 10

        text_y1 = center_y - spacing
        text_y2 = center_y + text2_size[1] + spacing

        text_x1 = center_x - text1_size[0] // 2
        text_x2 = center_x - text2_size[0] // 2

        cv2.putText(frame, text1, (text_x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, text2, (text_x2, text_y2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness)

        return frame