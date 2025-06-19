import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from trackers import DeepSortPlayerTracker
from utils.video_utils import read_video, save_video
from utils.stubs_utils import read_stub, save_stub

import json

# 讀取 label 名稱
LABELS_DICT_PATH = "action_recognition/dataset/labels_dict.json"
if os.path.exists(LABELS_DICT_PATH):
    with open(LABELS_DICT_PATH, 'r') as f:
        LABELS = json.load(f)
else:
    LABELS = {str(i): str(i) for i in range(11)}  # fallback

class ActionRecognitionModel:
    def __init__(self, model_path):
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # class 0~9
        state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(state_dict)
        self.model = model.eval()

    def predict(self, frames, read_from_stub=False, stub_path="stubs/action_predictions.pkl"):
        predictions = read_stub(read_from_stub, stub_path)
        if predictions is not None:
            return predictions

        actions = {}
        transform = Compose([
            Resize((112, 112)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        clip_len = 16
        for start in range(0, len(frames) - clip_len + 1):
            clip = frames[start:start+clip_len]
            clip_tensor = []
            for frame in clip:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[2] == 1:
                    frame_rgb = np.repeat(frame_rgb, 3, axis=2)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = transform(frame_pil)
                clip_tensor.append(frame_tensor)
            clip_tensor = torch.stack(clip_tensor).permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad():
                output = self.model(clip_tensor)
                prediction = torch.argmax(output, dim=1).item()
            for i in range(clip_len):
                actions[start + i] = prediction

        save_stub(stub_path, actions)
        return actions

def test_action_recognition_on_video(model_path, video_path, output_path, player_detector_path):
    # 載入影片
    frames = read_video(video_path)
    print(f"Total frames: {len(frames)}")

    # 載入模型與追蹤器
    action_model = ActionRecognitionModel(model_path)
    player_tracker = DeepSortPlayerTracker(player_detector_path)

    # 動作辨識
    action_predictions = action_model.predict(frames, read_from_stub=True)

    # 球員追蹤
    player_tracks = player_tracker.get_object_tracks(
        frames, 
        read_from_stub=False, 
        stub_path="stubs/player_tracks_tmp.pkl"
    )

    # 建立輸出影片列表
    output_frames = []

    for idx, frame in enumerate(frames):
        frame = frame.copy()
        action_id = action_predictions.get(idx)
        label = LABELS.get(str(action_id), str(action_id))
        cv2.putText(frame, f"Action: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        if idx < len(player_tracks):
            for player_id, player_info in player_tracks[idx].items():
                # 修正：將座標轉為 int
                x1, y1, x2, y2 = map(int, player_info['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        output_frames.append(frame)

    # 儲存輸出影片
    save_video(output_frames, output_path)
    print(f"[Done] Video saved to {output_path}")

# --------- 執行入口 ---------
if __name__ == "__main__":
    model_path = "models/action_r2plus1d_best.pt"
    video_path = "input_videos/video_1.mp4"
    output_path = "output_videos/test_output.mp4"
    player_detector_path = "models/player_detector.pt"
    test_action_recognition_on_video(model_path, video_path, output_path, player_detector_path)