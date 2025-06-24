from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel

import sys 
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their
    appearance. It maintains a consistent team assignment for each player across frames.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """
    def __init__(self,
                 team_1_class_name= "yellow shirt",
                 team_2_class_name= "blue shirt",
                 ):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args: 
            team_1_class_name (str): Description of Team 1's jersey appearance.
            team_2_class_name (str): Description of Team 2's jersey appearance.
        """
        self.team_colors = {}
        self.player_team_dict = {}        
    
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        
        self.color_name_to_rgb = {
            "white shirt": [255, 255, 255],
            "dark blue shirt": [0, 0, 139],
            "red shirt": [255, 0, 0],
            "dark red shirt": [176, 23, 31],
            "blue shirt": [0, 0, 255],
            "yellow shirt": [255, 255, 0],
            "black shirt": [0, 0, 0],
            "green shirt": [0, 128, 0],
        }

        self.team_1_color_rgb = self.color_name_to_rgb.get(self.team_1_class_name, [255, 255, 255])
        self.team_2_color_rgb = self.color_name_to_rgb.get(self.team_2_class_name, [0, 0, 139])

    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): Bounding box coordinates of the player.

        Returns:
            str: The classified jersey color/description.
        """
        h, w, _ = frame.shape
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return self.team_2_class_name  # fallback default

        image = frame[y1:y2, x1:x2]

        if image is None or image.size == 0:
            return self.team_2_class_name  # fallback default

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name

    def get_player_team(self,frame,player_bbox,player_id):
        """
        Gets the team assignment for a player, using cached results if available.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            player_bbox (tuple): Bounding box coordinates of the player.
            player_id (int): Unique identifier for the player.

        Returns:
            int: Team ID (1 or 2) assigned to the player.
        """
        if player_id in self.player_team_dict:
          return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id=2
        if player_color==self.team_1_class_name:
            team_id=1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self,video_frames,player_tracks,read_from_stub=False, stub_path=None):
        """
        Processes all video frames to assign teams to players, with optional caching.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team assignments for each frame.
        """
        
        player_assignment = read_stub(read_from_stub,stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        player_assignment=[]
        for frame_num, player_track in enumerate(player_tracks):        
            player_assignment.append({})
            
            if frame_num %50 ==0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                player_assignment[frame_num][player_id] = team
        
        save_stub(stub_path,player_assignment)

        return player_assignment