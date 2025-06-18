import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from PIL import Image
from trackers import DeepSortPlayerTracker  # 請確保 DeepSortPlayerTracker 可以使用 player_detector.pt
from utils.stubs_utils import save_stub  # 用於保存預測結果

class ActionRecognitionModel:
    def __init__(self, model_path):
        """
        Initialize the action recognition model by loading the pre-trained weights.
        
        Args:
            model_path (str): Path to the pre-trained model.
        """
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        
        # Load the model state dict (weights)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        
        self.model = model.eval()  # Set the model to evaluation mode

    def predict(self, video_frames):
        """
        Perform action recognition on a list of video frames.
        
        Args:
            video_frames (list): A list of video frames (numpy.ndarray format).
        
        Returns:
            dict: A dictionary of action predictions for each frame in the video.
        """
        actions = {}

        transform = Compose([
            Resize((112, 112)),  # Resize the frame to match the input size of the model
            ToTensor(),  # Convert the frame to a tensor
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the frame
        ])
        
        clip_len = 16  # 每 16 張為一段影片
        for start in range(0, len(video_frames) - clip_len + 1):
            clip = video_frames[start:start+clip_len]
            clip_tensor = []

            for frame in clip:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_rgb.shape[2] == 1:
                    frame_rgb = np.repeat(frame_rgb, 3, axis=2)  # Repeat grayscale to make 3 channels
                
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = transform(frame_pil)
                clip_tensor.append(frame_tensor)

            clip_tensor = torch.stack(clip_tensor).permute(1, 0, 2, 3).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = self.model(clip_tensor)
                prediction = torch.argmax(output, dim=1).item()  # Get the predicted action class
            
            for i in range(clip_len):
                actions[start + i] = prediction
        
        return actions

def test_action_recognition_on_video(model_path, video_path, output_path, player_detector_path):
    # Load the action recognition model
    action_model = ActionRecognitionModel(model_path)
    
    # Initialize the player tracker with YOLOv8 model for player detection
    player_tracker = DeepSortPlayerTracker(player_detector_path)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Predict actions for the frames
    action_predictions = action_model.predict(frames)

    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))

    # For each frame, track players and draw action labels
    for frame_idx, frame in enumerate(frames):
        player_data = player_tracker.get_player_tracks(frame)  # get player positions and IDs
        action = action_predictions.get(frame_idx, None)

        if action is not None:
            # Draw action label on the frame
            cv2.putText(frame, f"Action: {action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw player tracking (for each player in the frame)
        for player_id, player_info in player_data.items():
            bbox = player_info['bbox']
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0)  # Set player color to green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Write the frame to the output video
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Output video saved to {output_path}")

# Example usage:
model_path = "models/action_r2plus1d_best.pt"  # 預訓練模型的路徑
video_path = "input_videos/video_1.mp4"  # 測試影片路徑
output_path = "output_videos/test_output.mp4"  # 輸出影片的路徑
player_detector_path = "models/player_detector.pt"  # 載入 YOLOv8 訓練好的球員檢測模型

test_action_recognition_on_video(model_path, video_path, output_path, player_detector_path)