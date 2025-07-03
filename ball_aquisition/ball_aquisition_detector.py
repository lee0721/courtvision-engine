import sys 
sys.path.append('../')
from utils.bbox_utils import measure_distance, get_center_of_bbox

class BallAquisitionDetector:
    """
    Detects ball acquisition (possession) by players in a basketball game.

    This class determines which player is most likely in possession of the ball
    based on bounding boxes of both the ball and players. It uses a combination of 
    spatial proximity and bounding box containment heuristics.
    """

    def __init__(self):
        """
        Initialize thresholds and parameters for possession detection.

        Attributes:
            possession_threshold (int): Max distance (in pixels) between player and ball 
                for possible possession.
            min_frames (int): Minimum number of consecutive frames required to confirm possession.
            containment_threshold (float): Threshold for how much of the ball must be inside 
                a player’s bbox to be considered possessed.
        """
        self.possession_threshold = 50
        self.min_frames = 11
        self.containment_threshold = 0.8
        
    def get_key_basketball_player_assignment_points(self, player_bbox,ball_center):
        """
        Generate key anchor points around a player's bounding box for distance checks.

        Args:
            player_bbox (tuple): (x1, y1, x2, y2) bounding box of a player.
            ball_center (tuple): (x, y) center point of the ball.

        Returns:
            list of tuple: Key spatial points (x, y) around the bbox.
        """
        ball_center_x, ball_center_y = ball_center
        x1, y1, x2, y2 = player_bbox
        width, height = x2 - x1, y2 - y1

        output_points = []    
        
        # Horizontal or vertical alignment with ball
        if y1 < ball_center_y < y2:
            output_points += [(x1, ball_center_y), (x2, ball_center_y)]
        if x1 < ball_center_x < x2:
            output_points += [(ball_center_x, y1), (ball_center_x, y2)]

        # Additional surrounding key points
        output_points += [
            (x1 + width//2, y1),          # top center
            (x2, y1),                      # top right
            (x1, y1),                      # top left
            (x2, y1 + height//2),          # center right
            (x1, y1 + height//2),          # center left
            (x1 + width//2, y1 + height//2), # center point
            (x2, y2),                      # bottom right
            (x1, y2),                      # bottom left
            (x1 + width//2, y2),          # bottom center
            (x1 + width//2, y1 + height//3), # mid-top center
        ]
        return output_points
    
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        """
        Compute the overlap ratio between ball and player bbox.

        Args:
            player_bbox (tuple): (x1, y1, x2, y2)
            ball_bbox (tuple): (x1, y1, x2, y2)

        Returns:
            float: Ratio of the ball bbox area inside the player bbox (0~1).
        """
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox
        
        # Compute overlap area
        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)
        
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0
            
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        ball_area = (bx2 - bx1) * (by2 - by1)
        
        return intersection_area / ball_area
    
    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """
        Get the closest distance from ball to key points on player's bbox.

        Args:
            ball_center (tuple): (x, y)
            player_bbox (tuple): (x1, y1, x2, y2)

        Returns:
            float: Closest Euclidean distance (in pixels).
        """
        key_points = self.get_key_basketball_player_assignment_points(player_bbox,ball_center)
        return min(measure_distance(ball_center, point) for point in key_points)
    
    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        """
        Determine which player (if any) is most likely to possess the ball.

        Args:
            ball_center (tuple): (x, y)
            player_tracks_frame (dict): player_id → {'bbox': (x1, y1, x2, y2)}
            ball_bbox (tuple): Ball bounding box

        Returns:
            int: player_id with best evidence of possession, or -1 if no valid candidate.
        """
        high_containment_players = []
        regular_distance_players = []
        
        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('bbox', [])
            if not player_bbox:
                continue
                
            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        # First priority: players with high containment
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]
            
        # Second priority: players within distance threshold
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]
                
        return -1
    
    def detect_ball_possession(self, player_tracks, ball_tracks):
        """
        Detect ball possession for each frame based on player and ball tracks.

        Args:
            player_tracks (list): List of dicts per frame: player_id → info with 'bbox'
            ball_tracks (list): List of dicts per frame: ball_id → info with 'bbox'

        Returns:
            list: One player_id per frame (or -1 if undetermined).
        """
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        consecutive_possession_count = {}
        
        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue
                
            ball_bbox = ball_info.get('bbox', [])
            if not ball_bbox:
                continue
                
            ball_center = get_center_of_bbox(ball_bbox)
            
            best_player_id = self.find_best_candidate_for_possession(
                ball_center, 
                player_tracks[frame_num], 
                ball_bbox
            )

            if best_player_id != -1:
                number_of_consecutive_frames = consecutive_possession_count.get(best_player_id, 0) + 1
                consecutive_possession_count = {best_player_id: number_of_consecutive_frames} 

                if consecutive_possession_count[best_player_id] >= self.min_frames:
                    possession_list[frame_num] = best_player_id
            else:
                consecutive_possession_count ={}
    
        return possession_list