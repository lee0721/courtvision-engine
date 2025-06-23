import torch
import cv2
import numpy as np
from utils.stubs_utils import read_stub, save_stub
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from PIL import Image
import json
import os

LABELS_DICT_PATH = os.path.join(os.path.dirname(__file__), "dataset", "labels_dict.json")
with open(LABELS_DICT_PATH, "r") as f:
    LABELS_DICT = json.load(f)
    
class ActionRecognitionModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(model.fc.in_features, 10)
        )
        checkpoint = torch.load(model_path, map_location=self.device)

        # 如果是完整 checkpoint（有 'state_dict'、'epoch' 等 key）
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint  # 就是純 model weights

        model.load_state_dict(state_dict, strict=False)
        
        self.model = model.to(self.device).eval()
        self.transform = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.clip_len = 16
        self.stride = 8

    def predict(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        predictions = read_stub(read_from_stub, stub_path)
        if predictions is not None:
            return predictions

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

        results = {}
        for player_id, frames in player_clips.items():
            clips = []
            for i in range(0, len(frames), self.stride):
                if i + self.clip_len <= len(frames):
                    clips.append(frames[i:i + self.clip_len])
                else:
                    remaining = frames[i:]
                    pad_count = self.clip_len - len(remaining)
                    pad_frames = [remaining[-1].copy() for _ in range(pad_count)]
                    padded_clip = remaining + pad_frames
                    clips.append(padded_clip)

            if not clips:
                continue

            input_tensor = self._preprocess_clips(clips)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
                label_names = [LABELS_DICT[str(p)] for p in preds]
                print(f"[DEBUG] player_id: {player_id}, preds: {preds}, labels: {label_names}")
            results[player_id] = preds

        save_stub(stub_path, results)
        return results

    def _preprocess_clips(self, clips):
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