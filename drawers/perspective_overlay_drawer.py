import cv2 

class PerspectiveOverlayDrawer:
    """
    A drawer class responsible for rendering a tactical overlay view on each frame.
    This includes the court background, court keypoints, player positions, team assignments,
    and ball possession indication.

    Attributes:
        start_x (int): X offset for placing the tactical overlay.
        start_y (int): Y offset for placing the tactical overlay.
        team_1_color (list): BGR color for Team 1.
        team_2_color (list): BGR color for Team 2.
    """
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.start_x = 130
        self.start_y = 120
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, 
             video_frames, 
             court_image_path, 
             width,
             height,
             tactical_court_keypoints,
             tactical_player_positions=None,
             player_assignment=None,
             ball_acquisition=None):
        """
        Draws the tactical (top-down) overlay onto each video frame.

        Args:
            video_frames (list): Original video frames.
            court_image_path (str): File path to the court background image.
            width (int): Width of the overlay court image.
            height (int): Height of the overlay court image.
            tactical_court_keypoints (list): List of (x,y) keypoints for the court.
            tactical_player_positions (list, optional): Per-frame player positions in tactical view.
            player_assignment (list, optional): Per-frame team assignments for each player ID.
            ball_acquisition (list, optional): Per-frame ID of player possessing the ball.

        Returns:
            list: Video frames with tactical overlay view applied.
        """
        frame_height, frame_width = video_frames[0].shape[:2]
        # Adjust overlay size to not exceed original frame
        max_width = frame_width - self.start_x
        max_height = frame_height - self.start_y
        width = min(width, max_width)
        height = min(height, max_height)
        
        # Load and resize tactical court image
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            frame = frame.copy()

            # Define overlay position
            y1 = self.start_y
            y2 = self.start_y+height
            x1 = self.start_x
            x2 = self.start_x+width
            
            # Overlay court image with transparency
            alpha = 0.6  
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])
            
            # Draw court keypoints
            for keypoint_index, keypoint in enumerate(tactical_court_keypoints):
                x, y = keypoint
                x += self.start_x
                y += self.start_y
                cv2.circle(frame, (x, y), 5, (0, 252, 21)[::-1], -1)
                cv2.putText(frame, str(keypoint_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0)[::-1], 2)
            
            # Draw player positions if available
            if tactical_player_positions and player_assignment and frame_idx < len(tactical_player_positions):
                frame_positions = tactical_player_positions[frame_idx]
                frame_assignments = player_assignment[frame_idx] if frame_idx < len(player_assignment) else {}
                player_with_ball = ball_acquisition[frame_idx] if ball_acquisition and frame_idx < len(ball_acquisition) else -1
                
                for player_id, position in frame_positions.items():
                    team_id = frame_assignments.get(player_id, 1)
                    color = self.team_1_color[::-1] if team_id == 1 else self.team_2_color[::-1]

                    x = int(position[0]) + self.start_x
                    y = int(position[1]) + self.start_y

                    # Draw player circle
                    player_radius = 8
                    cv2.circle(frame, (x, y), player_radius, color, -1)

                    # Label with ID
                    cv2.putText(frame, str(player_id), (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                    # Highlight ball owner
                    if player_id == player_with_ball:
                        cv2.circle(frame, (x, y), player_radius + 3, (0, 0, 255), 2)
            
            output_video_frames.append(frame)

        return output_video_frames
