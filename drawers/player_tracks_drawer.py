from .utils import draw_ellipse,draw_traingle

class PlayerTracksDrawer:
    """
    Drawer module for rendering player tracks and ball possession indicators on each frame.

    Attributes:
        default_player_team_id (int): Default team ID assigned if not specified.
        team_1_color (list): RGB color for Team 1 players.
        team_2_color (list): RGB color for Team 2 players.
    """
    def __init__(self,team_1_color=[255, 245, 238],team_2_color=[128, 0, 128]):
        """
        Initialize with colors for both teams.

        Args:
            team_1_color (list): RGB color list for Team 1.
            team_2_color (list): RGB color list for Team 2.
        """
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self,video_frames,tracks,player_assignment,ball_aquisition):
        """
        Draw player bounding ellipses and triangle pointers for ball possession on each frame.

        Args:
            video_frames (list): List of video frames (np.ndarray).
            tracks (list): Per-frame dictionary {player_id: {'bbox': [x1, y1, x2, y2]}}.
            player_assignment (list): Per-frame dictionary {player_id: team_id}.
            ball_aquisition (list): List containing player ID with ball in each frame.

        Returns:
            list: Video frames with visualized player tracking and ball possession.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]
            assignment = player_assignment[frame_num]
            player_id_with_ball = ball_aquisition[frame_num]

            for track_id, player in player_dict.items():
                team_id = assignment.get(track_id, self.default_player_team_id)
                color = self.team_1_color[::-1] if team_id == 1 else self.team_2_color[::-1]

                # Draw ellipse for player bounding box
                frame = draw_ellipse(frame, player["bbox"], color, track_id)

                # Draw red triangle if player has the ball
                if track_id == player_id_with_ball:
                    frame = draw_traingle(frame, player["bbox"], (0, 0, 255))

            output_video_frames.append(frame)

        return output_video_frames
        
