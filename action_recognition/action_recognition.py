import torch
import cv2
import numpy as np
from utils.stubs_utils import read_stub, save_stub
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from PIL import Image
import json
import os
# Load label dictionary for action classes
LABELS_DICT_PATH = "labels_dict.json"
with open(LABELS_DICT_PATH, "r") as f:
    LABELS_DICT = json.load(f)
    
class ActionRecognitionModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained R(2+1)D model with custom classification head for 10 classes
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(model.fc.in_features, 10)
        )
        
        # Load model weights (support both raw and checkpoint formats)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
        self.model = model.to(self.device).eval()
        
        # Define transformation pipeline for input frames
        self.transform = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Number of frames per clip
        self.clip_len = 16 
        # Temporal stride between clips
        self.stride = 8

    def predict(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        # Try loading results from cache if available
        predictions = read_stub(read_from_stub, stub_path)
        if predictions is not None:
            return predictions

        # Step 1: Extract player clips from tracking bounding boxes
        player_clips = {}
        for frame_idx in range(len(video_frames)):
            if frame_idx >= len(player_tracks):
                continue
            for player_id, player_info in player_tracks[frame_idx].items():
                x1, y1, x2, y2 = map(int, player_info['bbox'])
                crop = video_frames[frame_idx][y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (112, 112))
                if player_id not in player_clips:
                    player_clips[player_id] = []
                player_clips[player_id].append(crop_resized)

        # Step 2: Perform inference for each player's clips
        results = {}
        for player_id, frames in player_clips.items():
            clips = []
            print(f"[DEBUG] player_id: {player_id}, total_frames: {len(frames)}")
            for i in range(0, len(frames) - self.clip_len + 1, self.stride):
                clips.append(frames[i:i + self.clip_len])
            print(f"[DEBUG] player_id: {player_id}, num_clips: {len(clips)}")

            if not clips:
                continue

            # Convert clips into tensor format
            input_tensor = self._preprocess_clips(clips)
            
            # Run model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
                label_names = [LABELS_DICT[str(p)] for p in preds]
                print(f"[DEBUG] player_id: {player_id}, preds: {preds}, labels: {label_names}")
            results[player_id] = preds

        # Save results to cache 
        save_stub(stub_path, results)
        return results

    def _preprocess_clips(self, clips):
        # Apply transform pipeline to convert clips into input tensor
        tensor_batch = []
        for clip in clips:
            clip_tensor = []
            for frame in clip:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[2] == 1:
                    frame_rgb = np.repeat(frame_rgb, 3, axis=2)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.transform(frame_pil)
                clip_tensor.append(frame_tensor)
            clip_tensor = torch.stack(clip_tensor).permute(1, 0, 2, 3)  # C, T, H, W
            tensor_batch.append(clip_tensor)
        return torch.stack(tensor_batch).to(self.device)